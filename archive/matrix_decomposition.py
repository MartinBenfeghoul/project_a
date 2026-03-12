"""Archived legacy SVD implementations kept for reference."""

import torch


def lowrank_svd(M, k, n_iter=3, dtype=torch.float32, **kwargs):
    """Compute a truncated low-rank SVD using `torch.svd_lowrank`."""
    og_dtype = M.dtype
    M = M.to(dtype).contiguous()
    U, S, V = torch.svd_lowrank(M, q=k, niter=n_iter)
    US = U * S.unsqueeze(-2)
    US, V = (t.to(og_dtype) for t in (US, V))
    return US, V.transpose(-2, -1)


def find_rank_wrt_cr(r, m, n):
    """Find the rank k that gives an approximate compression ratio r."""
    k = m * n / (r * (m + n))
    return int(round(k))


def calc_energy(M):
    """Return cumulative spectral energy for a matrix."""
    S = torch.linalg.svdvals(M)
    return (S**2).cumsum(-1) / (S**2).sum(-1, keepdim=True)


def find_rank_wrt_energy(S, energy_threshold):
    """Find the smallest rank whose cumulative energy crosses the threshold."""
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


def truncated_svd(
    M,
    rank_selection,
    cr=2.0,
    energy_threshold=0.95,
    dtype=torch.float32,
    **kwargs,
):
    """Legacy SVD path that selected between low-rank and full SVD."""
    if rank_selection == "comp_ratio":
        k = find_rank_wrt_cr(cr, M.size(-2), M.size(-1))
        return lowrank_svd(M, k, dtype=dtype, **kwargs)

    if rank_selection == "energy":
        og_dtype = M.dtype
        M_ = M.to(dtype).contiguous()

        U, S, Vh = torch.linalg.svd(M_, full_matrices=False)
        r = S.shape[-1]

        k = find_rank_wrt_energy(S, energy_threshold)
        k = max(1, min(int(k), r))

        U = U[..., :k]
        S = S[..., :k]
        Vh = Vh[..., :k, :]

        US = U * S.unsqueeze(-2)
        US, Vh = (t.to(og_dtype) for t in (US, Vh))
        return US, Vh

    raise ValueError(
        f"rank_selection set to {rank_selection}.",
        "Try either 'comp_ratio' or 'energy_threshold'",
    )


def full_svd(M, dtype=torch.float32):
    """Compute the full SVD of a matrix."""
    og_dtype = M.dtype
    M = M.to(dtype).contiguous()
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    U, S, V = (t.to(og_dtype) for t in (U, S, V))
    return U, S, V
