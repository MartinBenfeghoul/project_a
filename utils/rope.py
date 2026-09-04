import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def inverse_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Undo RoPE: x = x_rotated * cos - rotate_half(x_rotated) * sin"""
    return x * cos - rotate_half(x) * sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE: x_rotated = x * cos + rotate_half(x) * sin"""
    return x * cos + rotate_half(x) * sin


def apply_packed_rope(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Apply RoPE from ``[cos_half, sin_half]`` without expanding either."""
    half_dim = x.size(-1) // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    cos_half = packed[..., :half_dim]
    sin_half = packed[..., half_dim:]
    return torch.cat(
        (x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half),
        dim=-1,
    )


def inverse_packed_rope(
    x: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    """Undo RoPE from ``[cos_half, sin_half]`` without expanding either."""
    half_dim = x.size(-1) // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    cos_half = packed[..., :half_dim]
    sin_half = packed[..., half_dim:]
    return torch.cat(
        (x1 * cos_half + x2 * sin_half, x2 * cos_half - x1 * sin_half),
        dim=-1,
    )


def compute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin."""
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def get_rope_theta(model_config) -> float:
    rope_theta = getattr(model_config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(model_config, "rope_parameters", {}) or {}
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise ValueError("Model config does not define rope_theta")
    return float(rope_theta)


def get_rotary_embedding(model):
    """The model's own rotary embedding module."""
    for attribute in ("model", "transformer"):
        inner = getattr(model, attribute, None)
        if inner is not None and hasattr(inner, "rotary_emb"):
            return inner.rotary_emb
    if hasattr(model, "rotary_emb"):
        return model.rotary_emb
    raise ValueError("Model does not expose a rotary embedding module")


def model_rope_cos_sin(
    model,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin exactly as the model applies them."""
    rotary = get_rotary_embedding(model)
    position_ids = torch.arange(seq_len, device=device)[None]
    reference = torch.empty(0, device=device, dtype=dtype)
    with torch.no_grad():
        cos, sin = rotary(reference, position_ids)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)
