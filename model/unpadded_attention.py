"""SDPA that handles left padding without a dense [batch, 1, q, kv] mask."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import flash_attention_mask

ATTN_IMPLEMENTATION = "unpadded_sdpa"


def _causal_padding_mask(
    padding_mask: torch.Tensor, q_len: int
) -> torch.Tensor:
    kv_len = padding_mask.size(-1)
    kv_pos = torch.arange(kv_len, device=padding_mask.device)
    q_pos = kv_pos[kv_len - q_len :]
    causal = kv_pos[None, :] <= q_pos[:, None]
    return causal[None, None] & padding_mask[:, None, None, :]


def unpadded_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    q_len, kv_len = query.size(2), key.size(2)
    if attention_mask is None:
        return sdpa_attention_forward(
            module, query, key, value, None, dropout, scaling, **kwargs
        )

    attention_mask = attention_mask.bool()
    num_pad = kv_len - attention_mask.sum(dim=-1)
    positions = torch.arange(kv_len, device=attention_mask.device)
    left_padded = torch.equal(attention_mask, positions >= num_pad[:, None])
    if q_len != kv_len or not left_padded:
        mask = _causal_padding_mask(attention_mask, q_len)
        return sdpa_attention_forward(
            module, query, key, value, mask, dropout, scaling, **kwargs
        )

    output = torch.zeros_like(query)
    for b, pad in enumerate(num_pad.tolist()):
        output[b : b + 1, :, pad:] = F.scaled_dot_product_attention(
            query[b : b + 1, :, pad:],
            key[b : b + 1, :, pad:],
            value[b : b + 1, :, pad:],
            dropout_p=dropout,
            scale=scaling,
            is_causal=True,
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous(), None


def install_unpadded_sdpa(model) -> None:
    if getattr(model.config, "sliding_window", None) is not None:
        raise ValueError("unpadded_sdpa does not support sliding windows.")
    AttentionInterface.register(
        ATTN_IMPLEMENTATION, unpadded_sdpa_attention_forward
    )
    AttentionMaskInterface.register(ATTN_IMPLEMENTATION, flash_attention_mask)
    model.set_attn_implementation(ATTN_IMPLEMENTATION)


@contextmanager
def attention_for_batch(model, batch_size):
    """Use unpadded SDPA for batches > 1."""
    previous = model.config._attn_implementation
    try:
        if batch_size > 1:
            install_unpadded_sdpa(model)
        yield
    finally:
        if model.config._attn_implementation != previous:
            model.set_attn_implementation(previous)
