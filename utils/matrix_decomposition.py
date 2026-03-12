import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch


def _full_svd_single(M: torch.Tensor):
    """Compute a full SVD for one CPU matrix.

    Expects a single 2D matrix already on CPU and returns the
    `(U, S, Vh)` triple from `torch.linalg.svd`.
    """
    return torch.linalg.svd(M, full_matrices=False)


def find_rank_wrt_cr(r, m, n):
    """Find the rank k to use for low-rank approximation of a (m x n) matrix
    such that the compression ratio is ~r.
    """
    k = m * n / (r * (m + n))
    return int(round(k))


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
        k = find_rank_wrt_energy(
            S.reshape(*batch_shape, r), energy_threshold
        )
    else:
        raise ValueError(
            f"rank_selection set to {rank_selection}.",
            "Try either 'comp_ratio' or 'energy_threshold'",
        )

    k = max(1, min(int(k), r))

    U = U.reshape(*batch_shape, m, r)[..., :k]
    S = S.reshape(*batch_shape, r)[..., :k]
    Vh = Vh.reshape(*batch_shape, r, n)[..., :k, :]

    US = U * S.unsqueeze(-2)
    return US, Vh


def _parallel_svd_segments(
    host_segments: list[torch.Tensor],
    rank_selection: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
    cr: float = 2.0,
    energy_threshold: float = 0.95,
    **kwargs,
):
    """Run threaded CPU SVD for segments that already live on CPU.

    `host_segments` holds the matrices to decompose. Each resulting
    `(A, B)` pair is restored to `target_device` and `target_dtype`.
    """
    if not host_segments:
        return []

    specs = []
    flat_matrices = []
    for host_segment in host_segments:
        batch_shape = host_segment.shape[:-2]
        m, n = host_segment.shape[-2:]
        flat_segment = host_segment.reshape(-1, m, n)
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
            svds = list(executor.map(_full_svd_single, flat_matrices))
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

    The input tensor is copied to CPU once, decomposed there, and the
    resulting factor pair `(A, B)` is returned on the original device.
    """
    M_host = M.to(device="cpu", dtype=dtype).contiguous()
    US, Vh = _parallel_svd_segments(
        [M_host],
        rank_selection,
        target_device=M.device,
        target_dtype=M.dtype,
        cr=cr,
        energy_threshold=energy_threshold,
        **kwargs,
    )[0]
    return US, Vh


def decompose_to_segment_store(
    tensor: torch.Tensor,
    decompose_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    segment_ranges: list[list[tuple[int, int]]] | None = None,
    **decompose_kwargs,
):
    """Package decomposed factors into the cache segment-store format.

    When `segment_ranges` is omitted, the full tensor is decomposed. When it
    is provided, only those `[start, end)` ranges are decomposed. Returns the
    per-batch segment records plus the shared compressed-prefix boundary.
    """
    if segment_ranges is None:
        A, B = decompose_fn(tensor, **decompose_kwargs)
        seq_len = tensor.size(-2)
        layer_segments = [
            [{"range": (0, seq_len), "factors": (A[b], B[b])}]
            for b in range(tensor.size(0))
        ]
        return layer_segments, seq_len

    suffix_starts = [
        ranges[-1][1] if ranges else 0 for ranges in segment_ranges
    ]
    suffix_start = suffix_starts[0] if suffix_starts else 0
    if any(start != suffix_start for start in suffix_starts[1:]):
        raise ValueError(
            "All batches must share the same compressed prefix length."
        )

    specs = []
    for batch_idx, batch_ranges in enumerate(segment_ranges):
        for start_idx, end_idx in batch_ranges:
            specs.append((batch_idx, start_idx, end_idx))

    if decompose_fn is svd:
        # Copy the compressed prefix once, then slice CPU views per segment.
        prefix_host = tensor[..., :suffix_start, :].to(
            device="cpu",
            dtype=decompose_kwargs.get("dtype", torch.float32),
        ).contiguous()
        host_segments = [
            prefix_host[batch_idx, ..., start_idx:end_idx, :]
            for batch_idx, start_idx, end_idx in specs
        ]
        factor_pairs = _parallel_svd_segments(
            host_segments,
            target_device=tensor.device,
            target_dtype=tensor.dtype,
            **decompose_kwargs,
        )
    else:
        segments = [
            tensor[batch_idx, ..., start_idx:end_idx, :]
            for batch_idx, start_idx, end_idx in specs
        ]
        factor_pairs = [
            decompose_fn(segment, **decompose_kwargs) for segment in segments
        ]

    layer_segments = [[] for _ in range(tensor.size(0))]
    for (batch_idx, start_idx, end_idx), (A, B) in zip(specs, factor_pairs):
        layer_segments[batch_idx].append(
            {"range": (start_idx, end_idx), "factors": (A, B)}
        )
    return layer_segments, suffix_start


def reconstruct_segments(
    layer_segments: list[list[dict[str, tuple]]],
    suffix_tensor: torch.Tensor,
):
    """Reconstruct batched keys from stored factors and a live suffix.

    `layer_segments` is the segment-store output from
    `decompose_to_segment_store`, and `suffix_tensor` provides any
    uncompressed tail that should be appended back on.
    """
    recon_batches = []
    for batch_idx, batch_segments in enumerate(layer_segments):
        # TODO: batch same-length segments together to avoid many small matmuls.
        recon_pieces = [
            A @ B for A, B in (segment["factors"] for segment in batch_segments)
        ]
        if suffix_tensor[batch_idx].size(-2) > 0:
            recon_pieces.append(suffix_tensor[batch_idx])
        if not recon_pieces:
            recon_pieces.append(suffix_tensor[batch_idx])
        recon_batches.append(torch.cat(recon_pieces, dim=-2))
    return torch.stack(recon_batches, dim=0)


def calc_segment_store_compression_ratio(
    segment_store: dict[int, list[list[dict[str, tuple]]]],
    default_ratio: float | None = None,
):
    """Compute the average compression ratio of a segment store.

    Returns `default_ratio` directly when one is provided; otherwise averages
    the realized compression ratio over all stored `(A, B)` segment factors.
    """
    if default_ratio is not None:
        return default_ratio

    crs = 0.0
    num_segments = 0
    for layer_segments in segment_store.values():
        for batch_segments in layer_segments:
            for segment in batch_segments:
                A, B = segment["factors"]
                m, k = A.shape[-2:]
                n = B.size(-1)
                crs += (m * n) / (k * (m + n))
                num_segments += 1
    return crs / num_segments if num_segments > 0 else 0.0


# Learned decomposition methods
def mse(a, b):
    return ((a - b) ** 2).mean()


def init_lora_like(A, B, alpha=None):
    # A: (..., T, r)  (one factor random)
    # B: (..., r, D)  (other factor zero so product starts at 0)
    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    torch.nn.init.zeros_(B)

    r = A.shape[-1]
    scale = (alpha / r) if alpha is not None else 1.0
    return scale


@torch.enable_grad()
def learn_lora_matrix_sgd(M, k, lr=1e-2, n_iter=10, std=0.01):
    b, H, L, T, D = M.shape

    A = torch.randn((b, H, L, T, k), device=M.device, dtype=M.dtype) * std
    B = torch.randn((b, H, L, k, D), device=M.device, dtype=M.dtype) * std

    A, B = A.requires_grad_(True), B.requires_grad_(True)
    losses = []
    for _ in range(n_iter):
        # loss = mse(M, A @ B)
        loss = ((M - A @ B) ** 2).sum() / (b * H * L * T)
        gA, gB = torch.autograd.grad(loss, (A, B))
        assert torch.all(torch.isfinite(gA))

        A = (A - lr * gA).detach().requires_grad_(True)
        B = (B - lr * gB).detach().requires_grad_(True)
        losses.append(loss.detach().cpu().item())

    return A, B, losses


def cosine_with_warmup_scheduler(
    optimizer,
    num_steps,
    warmup_steps=0,
    min_lr_ratio=0.05,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.enable_grad()
def learn_lora_matrix(
    M,
    k,
    init_method="svd",
    lr=1e-2,
    n_iter=10,
    std=1,
    weight_decay=0.0,
    alpha=None,
    warmup_frac=0.05,
    min_lr_ratio=0.05,
    return_loss=False,
    dtype=torch.float32,
    **kwargs,
) -> tuple[
    torch.Tensor, torch.Tensor, list[float] | torch.Tensor, torch.Tensor
]:
    if init_method == "svd":
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        U = U[..., :k]
        S = S[..., :k]

        A = U * S.unsqueeze(-2)

        B = Vh[..., :k, :]
    else:
        shape = M.shape[:-2]
        T, D = M.shape[-2:]
        A = torch.empty((*shape, T, k), device=M.device, dtype=dtype)
        B = torch.zeros((*shape, k, D), device=M.device, dtype=dtype)

        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    scale = (alpha / k) if alpha is not None else 1.0

    A, B = A.requires_grad_(True), B.requires_grad_(True)

    opt = torch.optim.AdamW([A, B], lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=32, min_lr=1e-5
    )
    # warmup_steps = int(warmup_frac * n_iter)
    # sched = cosine_with_warmup_scheduler(
    #     opt,
    #     num_steps=n_iter,
    #     warmup_steps=warmup_steps,
    #     min_lr_ratio=min_lr_ratio,
    # )

    losses = []
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        pred = scale * (A @ B)
        loss = ((M - pred) ** 2).mean()
        loss.backward()
        opt.step()
        l = loss.detach().cpu().item()
        sched.step(l)
        losses.append(l)
    if return_loss:
        return A.detach(), B.detach(), losses
    return A.detach(), B.detach()


def lora_matrix(
    M,
    rank_selection,
    cr=2.0,
    energy_threshold=0.95,
    dtype=torch.float32,
    **kwargs,
):
    og_dtype = M.dtype
    M_ = M.to(dtype).contiguous()
    if rank_selection == "comp_ratio":
        k = find_rank_wrt_cr(cr, M_.size(-2), M_.size(-1))
    elif rank_selection == "energy":
        _, S, _ = torch.linalg.svd(M_, full_matrices=False)
        k = find_rank_wrt_energy(S, energy_threshold)
        del S
    else:
        raise ValueError(
            f"rank_selection set to {rank_selection}.",
            "Try either 'comp_ratio' or 'energy_threshold'",
        )
    A, B = learn_lora_matrix(M_, k, dtype=dtype, **kwargs)
    A, B = (t.to(og_dtype) for t in (A, B))
    return A, B


DECOMP_METHODS = {
    "svd": svd,
    "lora": lora_matrix,
}
