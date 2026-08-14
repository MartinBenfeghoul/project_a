import functools
import types

import torch
import torch.nn.functional as F

from .attention_predictor import (
    _attention_backend_specs,
    _extract_attention_forward_args,
    _get_attention_backend,
)


class SelectiveAttentionHandle:
    def __init__(self, module, forward):
        self.module = module
        self.previous = module.__dict__.get("forward")
        module.forward = types.MethodType(forward, module)

    def remove(self):
        if self.previous is None:
            self.module.__dict__.pop("forward", None)
        else:
            self.module.forward = self.previous


def _apply_selected_mask(scores, attention_mask, positions):
    if attention_mask is None:
        return scores
    batch_size, num_kv_heads, num_groups, selected_len = scores.shape
    if attention_mask.dim() == 2:
        source = attention_mask[:, None, :].expand(batch_size, num_kv_heads, -1)
        valid = source.gather(2, positions).unsqueeze(2)
        return scores.masked_fill(~valid.bool(), torch.finfo(scores.dtype).min)

    source = attention_mask[:, :, -1:, :]
    source = source.expand(
        batch_size, num_kv_heads, num_groups, source.size(-1)
    )
    gather_idx = positions[:, :, None, :].expand(
        batch_size, num_kv_heads, num_groups, selected_len
    )
    selected_mask = source.gather(-1, gather_idx)
    if selected_mask.dtype == torch.bool:
        return scores.masked_fill(~selected_mask, torch.finfo(scores.dtype).min)
    return scores + selected_mask.to(scores.dtype)


@torch.no_grad()
def _selective_forward(module, values, apply_rope):
    hidden_states = values["hidden_states"]
    past_key_values = values["past_key_value"]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = (
        module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = (
        module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )

    cos, sin = values["position_embeddings"]
    query_states, key_states = apply_rope(query_states, key_states, cos, sin)
    past_key_values.append_selective(
        key_states,
        value_states,
        module.layer_idx,
        {
            "sin": sin,
            "cos": cos,
            "cache_position": values.get("cache_position"),
        },
    )
    positions = past_key_values.select_positions(module.layer_idx, query_states)
    selected_keys, selected_values = past_key_values.retrieve_selected(
        module.layer_idx, positions
    )

    batch_size, num_query_heads, _, head_dim = query_states.shape
    num_kv_heads = selected_keys.size(1)
    num_groups = num_query_heads // num_kv_heads
    grouped_query = query_states.reshape(
        batch_size, num_kv_heads, num_groups, head_dim
    )
    scores = (
        torch.einsum("bhgd,bhkd->bhgk", grouped_query, selected_keys)
        * module.scaling
    )
    scores = _apply_selected_mask(
        scores, values.get("attention_mask"), positions
    )
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    output = torch.einsum("bhgk,bhkd->bhgd", weights, selected_values).reshape(
        batch_size, num_query_heads, 1, head_dim
    )
    output = output.transpose(1, 2).reshape(*input_shape, -1)
    return module.o_proj(output), None


def install_selective_attention(model):
    specs = _attention_backend_specs()
    handles = []
    for module in model.modules():
        spec = _get_attention_backend(module, specs)
        if spec is None:
            continue
        original_forward = module.forward

        @functools.wraps(original_forward)
        def forward(
            self, *args, _original=original_forward, _spec=spec, **kwargs
        ):
            values = _extract_attention_forward_args(args, kwargs)
            hidden_states = values.get("hidden_states")
            cache = values.get("past_key_value")
            supports = getattr(cache, "supports_selective_retrieval", None)
            if (
                self.training
                or hidden_states is None
                or hidden_states.size(1) != 1
                or not callable(supports)
                or not supports(self.layer_idx)
            ):
                return _original(*args, **kwargs)
            return _selective_forward(self, values, _spec["apply_rope"])

        handles.append(SelectiveAttentionHandle(module, forward))
    return handles
