import torch

# SVD-based decomposition
def lowrank_svd(A, k, niter=3, dtype=torch.float32):
    """Compute the truncated SVD of matrix A using PyTorch's built-in function.
    Args:
        A: (..., m, n) CUDA tensor
        k: number of singular values and vectors to compute
        niter: number of power iterations (default: 3)
        dtype: data type to use for computation (default: torch.float32)
        Returns:
            U: (..., m, k) left singular vectors
            S: (..., k) singular values
            Vh: (..., k, n) right singular vectors (transposed)
    """
    og_dtype = A.dtype
    A = A.to(dtype).contiguous()
    U, S, V = torch.svd_lowrank(A, q=k, niter=niter)  # V: (n, k)
    U, S, V = (t.to(og_dtype) for t in (U, S, V))
    return U, S, V.transpose(-2, -1)  # Vh

def calc_energy(M):
    S = torch.linalg.svdvals(M)  # (d,)
    return (S**2).cumsum(-1) / (S**2).sum(-1, keepdim=True)

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
        return 0  # degenerate

    S2 = S**2
    denom = S2.sum(-1, keepdim=True).clamp_min(torch.finfo(S2.dtype).tiny)
    energy = S2.cumsum(-1) / denom  # (..., r)

    # For each prefix position, does it meet threshold?
    meets = energy >= energy_threshold  # thr  # (..., r)

    # Find first True along last dim. If none, take r.
    first_idx = meets.float().argmax(dim=-1)  # (...,) but 0 if all False too
    has_any = meets.any(dim=-1)               # (...,)

    k_per = torch.where(has_any, first_idx + 1, torch.full_like(first_idx, S.shape[-1]))
    k = int(k_per.max().item())  # smallest k that works for *all* batch dims

    return k

def truncated_svd(A, k=None, energy_threshold=0.95, dtype=torch.float32, **kwargs):
    """Compute a truncated SVD of A.

    Args:
        A: (..., m, n) tensor (CUDA or CPU)
        k: int or None. If None, choose smallest k capturing `energy_threshold`.
        energy_threshold: float in (0, 1], or tensor broadcastable to A.shape[:-2]
        dtype: compute dtype (SVD is typically more stable in float32)

    Returns:
        U:  (..., m, k)
        S:  (..., k)
        Vh: (..., k, n)
    """
    if k is not None:
        # TODO: benchmark this path vs the full then truncated path in terms of compute time
        return lowrank_svd(A, k, dtype=dtype, **kwargs)
    og_dtype = A.dtype
    A_ = A.to(dtype).contiguous()

    U, S, Vh = torch.linalg.svd(A_, full_matrices=False)
    r = S.shape[-1]

    k = find_rank_wrt_energy(S, energy_threshold)

    # safety
    k = max(1, min(int(k), r))

    U = U[..., :k]
    S = S[..., :k]
    Vh = Vh[..., :k, :]

    U, S, Vh = (t.to(og_dtype) for t in (U, S, Vh))
    return U, S, Vh


def full_svd(A, dtype=torch.float32):
    """Compute the full SVD of matrix A using PyTorch's built-in function.
    Args:
        A: (..., m, n) CUDA tensor
        dtype: data type to use for computation (default: torch.float32)
    Returns:
        U: (..., m, min(m, n)) left singular vectors
        S: (..., min(m, n)) singular values
        Vh: (..., n, min(m, n)) right singular vectors (transposed)
    """
    og_dtype = A.dtype
    A = A.to(dtype).contiguous()
    U, S, V = torch.linalg.svd(A, full_matrices=False)  # V: (n, n)
    U, S, V = (t.to(og_dtype) for t in (U, S, V))
    return U, S, V  # Vh.transpose(-2, -1)

# Learned decomposition methods
def mse(a, b):
    return ((a - b) ** 2).mean()

import math
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
    for i in tqdm(range(n_iter)):
        # loss = mse(M, A @ B)
        loss = ((M - A @ B) ** 2).sum() / (b*H*L*T)
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
    M, k, lr=1e-2, n_iter=10, std=1, weight_decay=0.0, alpha=None,
    warmup_frac=0.05, min_lr_ratio=0.05
):
    shape = M.shape[:-2] 
    T, D = M.shape[-2:]

    A = torch.empty((*shape,T,k), device=M.device, dtype=torch.float32)
    B = torch.zeros((*shape,k,D), device=M.device, dtype=torch.float32)

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
    return A.detach(), B.detach(), losses