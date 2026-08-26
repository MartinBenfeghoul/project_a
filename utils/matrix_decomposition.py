import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import torch

from utils.turboquant import (
    CompressorParams,
    dequantise_factor,
    get_turboquant_compressor,
    is_quantised_factor,
    quantise_factor,
)


def _full_svd_single(M: torch.Tensor, dtype: torch.dtype):
    """Compute a full SVD for one matrix after moving it to CPU."""
    M = M.to(device="cpu", dtype=dtype).contiguous()
    return torch.linalg.svd(M, full_matrices=False)


def find_rank_wrt_cr(r, m, n):
    """Find the rank k to use for low-rank approximation of a (m x n) matrix
    such that the compression ratio is ~r.
    """
    k = m * n / (r * (m + n))
    k = min(math.floor(k + 0.5), min(m, n))
    if k < 1:
        warnings.warn(
            f"Target compression ratio {r} is too high for matrix of shape ({m}, {n}). Using rank 1."
        )
        k = 1
    return k


def find_rank_wrt_energy(S, energy_threshold):
    """
    Args:
        S: singular values tensor, shape (..., r). Assumed non-negative and sorted descending along last dim.
        energy_threshold: float in (0, 1], or tensor broadcastable to S.shape[:-1]

    Returns:
        k: smallest integer satisfying energy > energy_threshold for all dimensions
           (i.e., k is a Python int; rank is in [1, r] unless r==0).
    """
    if S.numel() == 0:
        return 0

    S2 = S**2
    denom = S2.sum(-1, keepdim=True).clamp_min(torch.finfo(S2.dtype).tiny)
    energy = S2.cumsum(-1) / denom
    meets = energy >= energy_threshold
    first_idx = meets.float().argmax(dim=-1)
    has_any = meets.any(dim=-1)

    k_per = torch.where(
        has_any, first_idx + 1, torch.full_like(first_idx, S.shape[-1])
    )
    return int(k_per.max().item())


@torch.no_grad()
def _xkv_cholqr(Y: torch.Tensor) -> torch.Tensor:
    """Cholesky-QR low-precision randomised SVD"""
    Y32 = Y.float()
    gram = torch.matmul(Y32.mH, Y32)
    gram = 0.5 * (gram + gram.mH)
    scale = (
        torch.diagonal(gram, dim1=-2, dim2=-1)
        .mean(dim=-1, keepdim=True)
        .clamp_min_(1e-12)[..., None]
    )
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    eps = 1e-5
    for _ in range(6):
        R, info = torch.linalg.cholesky_ex(gram + eps * scale * eye, upper=True)
        if (info == 0).all():
            return torch.linalg.solve_triangular(
                R, Y32, upper=True, left=False
            ).to(Y.dtype)
        eps *= 10
    return torch.linalg.qr(Y32, mode="reduced")[0].to(Y.dtype)


@torch.no_grad()
def randomised_svd(
    tensor: torch.Tensor,
    rank: int,
    n_iter: int = 2,
    oversample: int = 4,
):
    """Randomised SVD with Cholesky-QR orthogonalisation.

    Adapted from abdelfattah-lab/xKV, efficiency/svd/random_cholesky_v6.py
    """
    X = tensor.to(torch.bfloat16)
    m, n = X.shape[-2:]
    is_wide = m < n
    if is_wide:
        X = X.mH
        m, n = n, m

    q = min(rank + oversample, m, n)
    random = torch.randn(*X.shape[:-2], n, q, device=X.device, dtype=X.dtype)
    Q = _xkv_cholqr(torch.matmul(X, random))
    for _ in range(n_iter):
        Q = _xkv_cholqr(torch.matmul(X.mH, Q))
        Q = _xkv_cholqr(torch.matmul(X, Q))

    projected = torch.matmul(Q.mH, X)
    U_small, S, Vh = torch.linalg.svd(projected.float(), full_matrices=False)
    U = torch.matmul(Q, U_small.to(Q.dtype))
    if is_wide:
        U, Vh = Vh.mH, U.mH
    return U[..., :rank], S[..., :rank], Vh[..., :rank, :]


def _truncate_svd_factors(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    batch_shape: tuple[int, ...],
    m: int,
    n: int,
    rank_selection: str,
    cr: float = 2.0,
    energy_threshold: float = 0.95,
):
    """Truncate full-SVD outputs and restore the original leading shape.

    The inputs are stacked full-SVD results for one segment group. The
    outputs are the low-rank factors `(A, B)` after rank selection.
    """
    r = S.shape[-1]

    if rank_selection == "comp_ratio":
        k = find_rank_wrt_cr(cr, m, n)
    elif rank_selection == "energy":
        k = find_rank_wrt_energy(S.reshape(*batch_shape, r), energy_threshold)
    else:
        raise ValueError(
            f"Invalid rank_selection {rank_selection!r}. "
            "Expected 'comp_ratio' or 'energy'."
        )

    k = max(1, min(int(k), r))

    U = U.reshape(*batch_shape, m, r)[..., :k]
    S = S.reshape(*batch_shape, r)[..., :k]
    Vh = Vh.reshape(*batch_shape, r, n)[..., :k, :]

    US = U * S.unsqueeze(-2)
    return US, Vh


def _parallel_svd_segments(
    segments: list[torch.Tensor],
    rank_selection: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
    dtype: torch.dtype = torch.float32,
    cr: float = 2.0,
    energy_threshold: float = 0.95,
    **kwargs,
):
    """Run threaded CPU SVD for a list of tensor segments.

    Each segment may still live on the original device. The worker that owns a
    matrix moves it to CPU, computes its SVD, and the resulting factor pair is
    moved back to `target_device` in `target_dtype`.
    """
    if not segments:
        return []

    specs = []
    flat_matrices = []
    for segment in segments:
        batch_shape = segment.shape[:-2]
        m, n = segment.shape[-2:]
        flat_segment = segment.reshape(-1, m, n)
        specs.append(
            (
                batch_shape,
                m,
                n,
                flat_segment.size(0),
            )
        )
        flat_matrices.extend(flat_segment.unbind(0))

    max_workers = min(len(flat_matrices), torch.get_num_threads())
    max_workers = max(1, max_workers)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            svds = list(
                executor.map(_full_svd_single, flat_matrices, repeat(dtype))
            )
    finally:
        torch.set_num_threads(prev_threads)

    factor_pairs = []
    offset = 0
    for batch_shape, m, n, count in specs:
        segment_svds = svds[offset : offset + count]
        offset += count

        U = torch.stack([u for u, _, _ in segment_svds], dim=0)
        S = torch.stack([s for _, s, _ in segment_svds], dim=0)
        Vh = torch.stack([vh for _, _, vh in segment_svds], dim=0)

        US, Vh = _truncate_svd_factors(
            U,
            S,
            Vh,
            batch_shape,
            m,
            n,
            rank_selection,
            cr=cr,
            energy_threshold=energy_threshold,
        )
        factor_pairs.append(
            (
                US.to(device=target_device, dtype=target_dtype),
                Vh.to(device=target_device, dtype=target_dtype),
            )
        )

    return factor_pairs


def svd(
    M: torch.Tensor,
    rank_selection: str,
    cr: float = 2.0,
    energy_threshold: float = 0.95,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    """Apply the threaded CPU SVD path to one tensor.

    Each flattened matrix is handled by a worker that moves it to CPU for the
    SVD, then returns the truncated factors on the original device.
    """
    US, Vh = _parallel_svd_segments(
        [M],
        rank_selection,
        target_device=M.device,
        target_dtype=M.dtype,
        dtype=dtype,
        cr=cr,
        energy_threshold=energy_threshold,
        **kwargs,
    )[0]
    return US, Vh


def decompose_grouped_xkv_to_segment_store(
    tensor: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
    rank_selection: str,
    cr: float = 2.0,
    energy_threshold: float = 0.95,
    dtype: torch.dtype = torch.float32,
    svd_backend: str = "linalg",
    n_iter: int = 4,
    quantise_a: bool = False,
    quantise_b: bool = False,
    compressor_bits: int = 4,
    **kwargs,
):
    """Decompose grouped xKV segments directly on the tensor's current device."""
    del kwargs

    layer_segments = [[] for _ in range(tensor.size(0))]
    for batch_idx, batch_ranges in enumerate(segment_ranges):
        for start_idx, end_idx in batch_ranges:
            segment = tensor[batch_idx, ..., start_idx:end_idx, :]
            if svd_backend == "cholqr":
                rank = find_rank_wrt_cr(cr, segment.size(-2), segment.size(-1))
                U, S, Vh = randomised_svd(
                    segment.contiguous(), rank, n_iter=n_iter
                )
            else:
                U, S, Vh = torch.linalg.svd(
                    segment.to(dtype=dtype).contiguous(),
                    full_matrices=False,
                )
            US, Vh = _truncate_svd_factors(
                U,
                S,
                Vh,
                segment.shape[:-2],
                segment.size(-2),
                segment.size(-1),
                rank_selection,
                cr=cr,
                energy_threshold=energy_threshold,
            )
            layer_segments[batch_idx].append(
                {
                    "range": (start_idx, end_idx),
                    "factors": (
                        (
                            quantise_factor(
                                US.to(dtype=segment.dtype),
                                compressor_bits,
                            )
                            if quantise_a
                            else US.to(dtype=segment.dtype)
                        ),
                        (
                            quantise_factor(
                                Vh.to(dtype=segment.dtype),
                                compressor_bits,
                            )
                            if quantise_b
                            else Vh.to(dtype=segment.dtype)
                        ),
                    ),
                }
            )
    return layer_segments


def _batch_decode_quant_factors(layer_segments):
    """Decode all TurboQuantFactor instances across all segments in one call."""
    groups: dict[int, list] = {}
    for b_idx, batch_segs in enumerate(layer_segments):
        for s_idx, seg in enumerate(batch_segs):
            for f_idx, factor in enumerate(seg["factors"]):
                if not is_quantised_factor(factor):
                    continue
                p = factor.params
                groups.setdefault(p.shape[-1], []).append(
                    (b_idx, s_idx, f_idx, factor)
                )

    decoded = {}
    for dim, entries in groups.items():
        bits = entries[0][3].bits
        device = entries[0][3].params.indices.device
        compressor = get_turboquant_compressor(dim, bits, device)
        params_list = [e[3].params for e in entries]
        n_each = [math.prod(p.shape[:-1]) for p in params_list]
        batched = CompressorParams(
            indices=torch.cat(
                [p.indices.reshape(n, -1) for p, n in zip(params_list, n_each)]
            ),
            norms=torch.cat(
                [p.norms.reshape(n) for p, n in zip(params_list, n_each)]
            ),
            shape=(sum(n_each), dim),
            dtype=params_list[0].dtype,
            bits=bits,
            idx_pad=params_list[0].idx_pad,
        )
        flat = compressor.decode(batched)
        offset = 0
        for (b_idx, s_idx, f_idx, factor), n in zip(entries, n_each):
            decoded[(b_idx, s_idx, f_idx)] = flat[offset : offset + n].reshape(
                factor.params.shape
            )
            offset += n
    return decoded


def reconstruct_segments(
    layer_segments: list[list[dict[str, tuple]]],
    suffix_tensor: torch.Tensor,
):
    """Reconstruct batched keys from stored factors and a live suffix.

    `layer_segments` contains per-segment factor pairs, and `suffix_tensor`
    provides any uncompressed tail that should be appended back on.
    """
    pre_decoded = _batch_decode_quant_factors(layer_segments)

    recon_batches = []
    for batch_idx, batch_segments in enumerate(layer_segments):
        recon_pieces = []
        for seg_idx, segment in enumerate(batch_segments):
            A_raw, B_raw = segment["factors"]
            A_key = (batch_idx, seg_idx, 0)
            B_key = (batch_idx, seg_idx, 1)
            A = (
                pre_decoded[A_key]
                if A_key in pre_decoded
                else dequantise_factor(A_raw)
            )
            B = (
                pre_decoded[B_key]
                if B_key in pre_decoded
                else dequantise_factor(B_raw)
            )
            recon_pieces.append(A @ B)
        if suffix_tensor[batch_idx].size(-2) > 0 or not recon_pieces:
            recon_pieces.append(suffix_tensor[batch_idx])
        recon_batches.append(torch.cat(recon_pieces, dim=-2))
    return torch.stack(recon_batches, dim=0)
