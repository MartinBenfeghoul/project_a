import math
import os

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention_backend_specs() -> list[dict]:
    specs = []

    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    )

    specs.append(
        {
            "name": "llama",
            "base_cls": LlamaAttention,
            "flash_cls_name": "LlamaFlashAttention2",
            "apply_rope": llama_apply_rotary_pos_emb,
        }
    )
 
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        apply_rotary_pos_emb as mistral_apply_rotary_pos_emb,
    )

    specs.append(
        {
            "name": "mistral",
            "base_cls": MistralAttention,
            "flash_cls_name": "MistralFlashAttention2",
            "apply_rope": mistral_apply_rotary_pos_emb,
        }
    )

    return specs


def _get_attention_backend(module: nn.Module, specs: list[dict]) -> dict | None:
    for spec in specs:
        if isinstance(module, spec["base_cls"]):
            return spec
    return None

def _repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * num_key_value_groups,
        seq_len,
        head_dim,
    )


class AttentionPredictor(nn.Module):
    """
    Predicts the next pooled attention row from a short
    history of pooled attention rows.

    Input shape: [N, history_step, num_blocks]
    Output shape: [N, num_blocks]
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.conv3(x)
        return x.squeeze(1)


def max_pool_attention(attn: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Max pool attention probabilities over the KV-token axis.
    """
    padding_size = (block_size - attn.shape[-1] % block_size) % block_size
    if padding_size:
        attn = F.pad(attn, (0, padding_size), value=0.0)
    return attn.reshape(*attn.shape[:-1], -1, block_size).amax(dim=-1)


def attention_targets_to_distribution(
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalise pooled attention targets into probability distribution."""
    target = target.float().clamp_min(0)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(eps)


def topk_block_mask(target_dist: torch.Tensor, topk_blocks: int) -> torch.Tensor:
    """Create binary mask for top-k target blocks."""
    k = min(topk_blocks, target_dist.shape[-1])
    indices = target_dist.topk(k, dim=-1).indices
    mask = torch.zeros_like(target_dist)
    mask.scatter_(-1, indices, 1.0)
    return mask


def load_attention_predictor(
    checkpoint_path: str,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> AttentionPredictor:
    """Load attention predictor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = AttentionPredictor().to(device=device)
    model.load_state_dict(state_dict)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    return model


def _extract_attention_forward_args(args, kwargs):
    if len(args) > 1 and isinstance(args[1], tuple):
        names = [
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
            "cache_position",
        ]
    else:
        names = [
            "hidden_states",
            "attention_mask",
            "position_ids",
            "past_key_value",
            "output_attentions",
            "use_cache",
            "cache_position",
            "position_embeddings",
        ]
    values = dict(zip(names, args))
    values.update(kwargs)
    if values.get("past_key_value") is None:
        values["past_key_value"] = values.get("past_key_values")
    return values


def _causal_history_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    history_step: int,
    sliding_window: int | None = None,
) -> torch.Tensor:
    _, _, q_len, head_dim = query_states.shape
    hist_len = min(history_step, q_len)
    query_hist = query_states[:, :, -hist_len:, :]

    scores = torch.matmul(query_hist, key_states.transpose(2, 3))
    scores = scores / math.sqrt(head_dim)

    q_pos = torch.arange(q_len - hist_len, q_len, device=scores.device)
    k_pos = torch.arange(key_states.shape[-2], device=scores.device)
    causal_mask = k_pos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(
        causal_mask[None, None, :, :],
        torch.finfo(scores.dtype).min,
    )
    if sliding_window is not None and sliding_window > 0:
        window_mask = k_pos[None, :] <= (q_pos[:, None] - sliding_window)
        scores = scores.masked_fill(
            window_mask[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            padding_mask = (
                attention_mask[:, None, None, : key_states.shape[-2]] == 0
            )
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        elif attention_mask.dim() == 4:
            scores = scores + attention_mask[
                :, :, -hist_len:, : key_states.shape[-2]
            ]

    return torch.softmax(scores.float(), dim=-1).to(query_states.dtype)


@torch.no_grad()
def _predict_importance_from_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    predictor: AttentionPredictor,
    history_step: int,
    block_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    sliding_window: int | None = None,
) -> torch.Tensor:
    key_states = _repeat_kv(key_states, num_key_value_groups)
    attn_history = _causal_history_attention(
        query_states=query_states,
        key_states=key_states,
        attention_mask=attention_mask,
        history_step=history_step,
        sliding_window=sliding_window,
    )

    pooled = max_pool_attention(attn_history, block_size)
    batch, query_heads, hist_len, num_blocks = pooled.shape

    device = query_states.device
    if next(predictor.parameters()).device != device:
        predictor.to(device=device)
    pred_dtype = next(predictor.parameters()).dtype

    logits = predictor(
        pooled.reshape(batch * query_heads, hist_len, num_blocks).to(
            dtype=pred_dtype
        )
    )
    block_importance = torch.softmax(logits.float(), dim=-1).reshape(
        batch,
        query_heads,
        num_blocks,
    )

    token_importance = block_importance.repeat_interleave(block_size, dim=-1)
    token_importance = token_importance[..., : key_states.shape[-2]]

    if query_heads != num_key_value_heads:
        if query_heads % num_key_value_heads != 0:
            raise ValueError(
                "query_heads must be divisible by num_key_value_heads"
            )
        token_importance = token_importance.reshape(
            batch,
            num_key_value_heads,
            query_heads // num_key_value_heads,
            key_states.shape[-2],
        ).mean(dim=2)

    return token_importance.float()


def install_attention_predictor_hooks(
    model: nn.Module,
    predictor: AttentionPredictor,
    history_step: int = 64,
    block_size: int = 16,
) -> list:
    """
    Install attention pre-hooks that compute small attention-history
    slice for Attention Predictor, then let the original attention module continue.

    The hook stores only value-importance tensors on the active cache via
    cache.set_value_importance(layer_idx, importance).
    """
    attention_specs = _attention_backend_specs()

    handles = []
    found_attention = 0

    def hook(module, args, kwargs):
        values = _extract_attention_forward_args(args, kwargs)
        hidden_states = values.get("hidden_states")
        past_key_value = values.get("past_key_value")
        spec = _get_attention_backend(module, attention_specs)

        if hidden_states is None or hidden_states.shape[1] <= 1:
            return
        if past_key_value is None or not hasattr(
            past_key_value, "set_value_importance"
        ):
            return

        batch, q_len, _ = hidden_states.shape
        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)

        head_dim = module.head_dim
        query_states = query_states.view(batch, q_len, -1, head_dim).transpose(1, 2)
        key_states = key_states.view(batch, q_len, -1, head_dim).transpose(1, 2)
        num_query_heads = query_states.shape[1]
        num_key_value_heads = key_states.shape[1]
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "query heads must be divisible by key-value heads"
            )
        num_key_value_groups = num_query_heads // num_key_value_heads

        position_embeddings = values.get("position_embeddings")
        if position_embeddings is None:
            cos, sin = module.rotary_emb(key_states, values.get("position_ids"))
        else:
            cos, sin = position_embeddings
        query_states, key_states = spec["apply_rope"](
            query_states,
            key_states,
            cos,
            sin,
        )

        importance = _predict_importance_from_qk(
            query_states=query_states,
            key_states=key_states,
            attention_mask=values.get("attention_mask"),
            predictor=predictor,
            history_step=history_step,
            block_size=block_size,
            num_key_value_heads=num_key_value_heads,
            num_key_value_groups=num_key_value_groups,
            sliding_window=getattr(module.config, "sliding_window", None),
        )
        past_key_value.set_value_importance(module.layer_idx, importance)

    for module in model.modules():
        spec = _get_attention_backend(module, attention_specs)
        if spec is None:
            continue
        found_attention += 1
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))


    if not handles:
        raise RuntimeError(
            f"No attention predictor hooks installed. Found {found_attention} "
            "supported attention modules."
        )

    return handles

def get_attn_predictor_hook_handles(args, model):
    attn_predictor = None
    attn_predictor_hook_handles = []
    if args.use_attn_predictor:
        if args.attn_predictor_path is None:
            raise ValueError("--use_attn_predictor requires --attn_predictor_path")
        attn_predictor = load_attention_predictor(
            args.attn_predictor_path,
            device=next(model.parameters()).device,
        )
        attn_predictor_hook_handles = install_attention_predictor_hooks(
            model,
            predictor=attn_predictor,
            history_step=args.attn_predictor_history_step,
            block_size=args.attn_predictor_block_size,
            )
    return attn_predictor_hook_handles


def _load_attn_predictor_config(attn_predictor_path: str) -> dict:
    config_path = os.path.join(os.path.dirname(attn_predictor_path), "config.yaml")
    if os.path.exists(config_path):
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    checkpoint = torch.load(attn_predictor_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        return checkpoint["config"]

    raise ValueError(
        "Could not find attention predictor config. Expected config.yaml next to "
        f"{attn_predictor_path}."
    )


def apply_attn_predictor_config(args):
    if not args.use_attn_predictor:
        args.attn_predictor_history_step = None
        args.attn_predictor_block_size = None
        return args
    if args.attn_predictor_path is None:
        raise ValueError("--use_attn_predictor requires --attn_predictor_path")

    config = _load_attn_predictor_config(args.attn_predictor_path)
    required_keys = ("history_step", "block_size")
    missing = [key for key in required_keys if key not in config or config[key] is None]
    if missing:
        raise ValueError(
            "Attention predictor config is missing required field(s): "
            f"{', '.join(missing)}"
        )

    args.attn_predictor_history_step = int(config["history_step"])
    args.attn_predictor_block_size = int(config["block_size"])

    print(
        "[attn_predictor] Loaded config: "
        f"history_step={args.attn_predictor_history_step}, "
        f"block_size={args.attn_predictor_block_size}"
    )
    return args
