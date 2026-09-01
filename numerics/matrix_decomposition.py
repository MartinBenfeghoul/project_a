import math
import warnings
from dataclasses import dataclass

import torch

from numerics.quantisation import (
    CompressorParams,
    TurboQuantFactor,
    dequantise_factor,
    get_turboquant_compressor,
    is_quantised_factor,
    quantise_factor,
)

Factor = torch.Tensor | TurboQuantFactor


@dataclass(frozen=True)
class SVDDecompositionConfig:
    """Settings controlling how a segment store is decomposed."""

    compression_ratio: float = 2.0
    svd_backend: str = "linalg"
    dtype: torch.dtype = torch.float32
    n_iter: int = 4
    quantise_a: bool = False
    quantise_b: bool = False
    compressor_bits: int = 4


@dataclass
class FactorSegment:
    """A single stored factor covering `token_range`."""

    token_range: tuple[int, int]
    factor: Factor


@dataclass
class FactorPairSegment:
    """The `(left, right)` factor pair covering `token_range`."""

    token_range: tuple[int, int]
    factors: tuple[Factor, Factor]


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
    cr: float = 2.0,
):
    """Truncate full-SVD outputs and restore the original leading shape.

    The inputs are stacked full-SVD results for one segment group. The
    outputs are the low-rank factors `(A, B)` after rank selection.
    """
    r = S.shape[-1]

    k = find_rank_wrt_cr(cr, m, n)
    k = max(1, min(int(k), r))

    U = U.reshape(*batch_shape, m, r)[..., :k]
    S = S.reshape(*batch_shape, r)[..., :k]
    Vh = Vh.reshape(*batch_shape, r, n)[..., :k, :]

    US = U * S.unsqueeze(-2)
    return US, Vh


def _maybe_quantise(
    factor: torch.Tensor,
    quantise: bool,
    compressor_bits: int,
) -> Factor:
    return quantise_factor(factor, compressor_bits) if quantise else factor


def decompose_grouped_xkv_to_segment_store(
    tensor: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
    config: SVDDecompositionConfig,
) -> list[list[FactorPairSegment]]:
    """Decompose grouped xKV segments directly on the tensor's current device."""
    cr = config.compression_ratio

    layer_segments = [[] for _ in range(tensor.size(0))]
    for batch_idx, batch_ranges in enumerate(segment_ranges):
        for start_idx, end_idx in batch_ranges:
            segment = tensor[batch_idx, ..., start_idx:end_idx, :]
            if config.svd_backend == "cholqr":
                rank = find_rank_wrt_cr(cr, segment.size(-2), segment.size(-1))
                U, S, Vh = randomised_svd(
                    segment.contiguous(), rank, n_iter=config.n_iter
                )
            else:
                U, S, Vh = torch.linalg.svd(
                    segment.to(dtype=config.dtype).contiguous(),
                    full_matrices=False,
                )
            US, Vh = _truncate_svd_factors(
                U,
                S,
                Vh,
                segment.shape[:-2],
                segment.size(-2),
                segment.size(-1),
                cr=cr,
            )
            layer_segments[batch_idx].append(
                FactorPairSegment(
                    token_range=(start_idx, end_idx),
                    factors=(
                        _maybe_quantise(
                            US.to(dtype=segment.dtype),
                            config.quantise_a,
                            config.compressor_bits,
                        ),
                        _maybe_quantise(
                            Vh.to(dtype=segment.dtype),
                            config.quantise_b,
                            config.compressor_bits,
                        ),
                    ),
                )
            )
    return layer_segments


def _batch_decode_quant_factors(layer_segments):
    """Decode all TurboQuantFactor instances across all segments in one call."""
    groups: dict[int, list] = {}
    for b_idx, batch_segs in enumerate(layer_segments):
        for s_idx, seg in enumerate(batch_segs):
            for f_idx, factor in enumerate(seg.factors):
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
    layer_segments: list[list[FactorPairSegment]],
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
            A_raw, B_raw = segment.factors
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
