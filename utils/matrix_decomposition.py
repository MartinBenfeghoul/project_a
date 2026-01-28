import torch


def truncated_svd(A, k, niter=3, dtype=torch.float32):
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