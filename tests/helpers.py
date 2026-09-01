import torch

from utils.rope import apply_rope


def rope_cos_sin(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )
    frequencies = torch.outer(positions, inv_freq)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    cos = embeddings.cos().to(dtype).expand(batch_size, -1, -1).contiguous()
    sin = embeddings.sin().to(dtype).expand(batch_size, -1, -1).contiguous()
    return cos, sin


def apply_model_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return apply_rope(tensor, cos.unsqueeze(1), sin.unsqueeze(1))


def gather_tokens(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return tensor.gather(
        2,
        positions[..., None].expand(-1, -1, -1, tensor.size(-1)),
    )


def build_llama(
    num_layers: int = 2,
    *,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 8,
    seed: int = 0,
):
    """A tiny Llama whose attention shapes are convenient for cache tests."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=num_attention_heads * head_dim,
        intermediate_size=2 * num_attention_heads * head_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaForCausalLM(config).eval()


def install_value_importance_hooks(model) -> list:
    """Feed deterministic per-token value importance to the active cache.

    Stands in for the attention-predictor hooks so that eviction can be
    exercised without depending on a trained predictor.
    """
    from model.attention_predictor import (
        _attention_backend_specs,
        _extract_attention_forward_args,
        _get_attention_backend,
    )

    specs = _attention_backend_specs()
    handles = []

    def hook(module, args, kwargs):
        values = _extract_attention_forward_args(args, kwargs)
        hidden_states = values.get("hidden_states")
        cache = values.get("past_key_value")
        if hidden_states is None or hidden_states.shape[1] <= 1:
            return
        if cache is None or not hasattr(cache, "set_value_importance"):
            return
        batch, seq_len, _ = hidden_states.shape
        keys = (
            module.k_proj(hidden_states)
            .view(batch, seq_len, -1, module.head_dim)
            .transpose(1, 2)
        )
        cache.set_value_importance(module.layer_idx, keys.norm(dim=-1))

    for module in model.modules():
        if _get_attention_backend(module, specs) is not None:
            handles.append(
                module.register_forward_pre_hook(hook, with_kwargs=True)
            )
    return handles


def low_rank_keys(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    rank: int,
) -> torch.Tensor:
    """Keys that genuinely live on a `rank`-dimensional subspace."""
    flat_dim = num_heads * head_dim
    left = torch.randn(batch_size, seq_len, rank)
    right = torch.randn(batch_size, rank, flat_dim)
    return (
        torch.bmm(left, right)
        .reshape(batch_size, seq_len, num_heads, head_dim)
        .transpose(1, 2)
        .contiguous()
    )
