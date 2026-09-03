from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call

from utils.rope import compute_rope_cos_sin, inverse_rope


@dataclass(frozen=True)
class LearnedLayerInit:
    """Learned initialisation for one value-cache MLP layer."""

    weights: dict | None = None

    @property
    def has_weights(self) -> bool:
        return self.weights is not None


class LearnedInit:
    """Checkpoint-backed lookup for per-layer learned initialisation."""

    def __init__(self, layers: dict[int, LearnedLayerInit] | None = None):
        self.layers = layers or {}

    @classmethod
    def empty(cls) -> "LearnedInit":
        return cls()

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
    ) -> "LearnedInit":
        checkpoint = torch.load(path, map_location="cpu")
        layer_weights = {
            int(key.split("_")[1]): value
            for key, value in checkpoint.items()
            if key.startswith("layer_")
        }
        return cls(
            {
                idx: LearnedLayerInit(weights=weights)
                for idx, weights in layer_weights.items()
            }
        )

    @classmethod
    def from_value_mlp_checkpoint(cls, path: str) -> "LearnedInit":
        checkpoint = torch.load(path, map_location="cpu")
        layers = {
            int(k.split("_")[1]): LearnedLayerInit(weights=v)
            for k, v in checkpoint.items()
            if k.startswith("layer_")
        }
        return cls(layers)

    def for_layer(self, layer_idx: int) -> LearnedLayerInit:
        return self.layers.get(layer_idx, LearnedLayerInit())


def trainable_params(mlp) -> list[torch.Tensor]:
    params = list(mlp.weights) + list(mlp.biases)
    if hasattr(mlp, "W_linear"):
        params.append(mlp.W_linear)
    return params


def setup_optimizer(mlps, config):
    params = [param for mlp in mlps for param in trainable_params(mlp)]
    return torch.optim.Adam(
        params,
        lr=float(config.meta_lr),
    )


@dataclass(frozen=True)
class KeyReconstructionConfig:
    """xKV key-cache settings mirrored from evaluation into meta-training."""

    compression_ratio: float = 4.0
    layer_group_size: int = 4
    svd_backend: str = "cholqr"
    quantise_a: bool = False
    quantise_b: bool = False
    compressor_bits: int = 4


def _rope_cache_kwargs(
    keys: torch.Tensor,
    rope_theta: float,
    padding_mask: torch.Tensor | None,
) -> dict:
    """The prefill cache_kwargs an xKV key cache expects from a model."""
    seq_len, head_dim = keys.shape[2], keys.shape[-1]
    cos, sin = compute_rope_cos_sin(
        seq_len,
        head_dim,
        rope_theta,
        keys.device,
        keys.dtype,
    )
    cache_kwargs = {
        "cos": cos[:, 0],
        "sin": sin[:, 0],
        "cache_position": torch.arange(seq_len, device=keys.device),
    }
    if padding_mask is not None:
        cache_kwargs["padding_mask"] = padding_mask.to(
            device=keys.device,
            dtype=torch.bool,
        )
    return cache_kwargs


@torch.no_grad()
def reconstruct_keys_xkv(
    cache,
    config: KeyReconstructionConfig,
    *,
    rope_theta: float,
    padding_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    from cache.backends.xkv import XKVKeysCache

    num_layers = len(cache.layers)
    key_cache = XKVKeysCache(
        layer_group_size=config.layer_group_size,
        num_layers=num_layers,
        xkv_svd_backend=config.svd_backend,
        comp_ratio=config.compression_ratio,
        quantise_a=config.quantise_a,
        quantise_b=config.quantise_b,
        compressor_bits=config.compressor_bits,
    )
    cache_kwargs = _rope_cache_kwargs(
        cache.layers[0].keys,
        rope_theta,
        padding_mask,
    )
    for layer_idx, layer in enumerate(cache.layers):
        key_cache.update(layer.keys.detach(), layer_idx, cache_kwargs)

    keys = []
    for layer_idx in range(num_layers):
        roped = key_cache.get_reconstructed_keys_only(layer_idx)
        if roped is None:
            raise ValueError(
                f"xKV left layer {layer_idx} uncompressed; the whole prefill "
                "must be decomposed before the inner loop runs."
            )
        keys.append(
            key_cache.layers[layer_idx]._undo_rope(roped, prefill=True)
        )
    return keys


def prepare_kvs(
    cache,
    *,
    rope_theta: float,
    dtype: torch.dtype,
    key_config: KeyReconstructionConfig | None = None,
    padding_mask: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Detach, un-RoPE, and cast support KV pairs once per batch."""
    if key_config is not None:
        keys_per_layer = reconstruct_keys_xkv(
            cache,
            key_config,
            rope_theta=rope_theta,
            padding_mask=padding_mask,
        )
    else:
        keys_per_layer = []
        for layer in cache.layers:
            keys = layer.keys.detach()
            cos, sin = compute_rope_cos_sin(
                keys.shape[2],
                keys.shape[-1],
                rope_theta,
                keys.device,
                keys.dtype,
            )
            keys_per_layer.append(inverse_rope(keys, cos, sin))

    return [
        (
            keys.to(dtype=dtype),
            layer.values.detach().to(dtype=dtype),
        )
        for keys, layer in zip(keys_per_layer, cache.layers)
    ]


def _predict(mlps, kvs, params, names_by_mlp) -> list[torch.Tensor]:
    preds = []
    offset = 0
    for mlp, (keys, _), names in zip(mlps, kvs, names_by_mlp):
        count = len(names)
        state = dict(zip(names, params[offset : offset + count]))
        preds.append(functional_call(mlp, state, (keys,)))
        offset += count
    if offset != len(params):
        raise ValueError("Functional parameter count does not match the MLPs")
    return preds


def _mse(preds, kvs) -> torch.Tensor:
    return sum(
        F.mse_loss(pred, values) for pred, (_, values) in zip(preds, kvs)
    )


def _compressed_rows(mlp, values: torch.Tensor, target_cr: float) -> int:
    batch_size, num_heads, num_tokens, head_dim = values.shape
    dtype_size = torch.finfo(values.dtype).bits / 8
    original_bytes = batch_size * num_heads * num_tokens * head_dim * dtype_size
    model_bytes = (
        sum(parameter.numel() for parameter in mlp.parameters()) * dtype_size
    )
    residual_row_bytes = (
        head_dim * dtype_size + torch.iinfo(torch.int32).bits / 8
    )
    residual_rows = int(
        max(0.0, original_bytes / target_cr - model_bytes) / residual_row_bytes
    )
    total_rows = batch_size * num_heads * num_tokens
    residual_rows = max(0, min(residual_rows, total_rows))
    rows_per_batch = num_heads * num_tokens
    return max(
        1,
        int(rows_per_batch * (1.0 - residual_rows / total_rows)),
    )


def _residual_loss(mlps, kvs, preds, target_cr) -> torch.Tensor:
    errors = [
        F.mse_loss(pred, values, reduction="none")
        for pred, (_, values) in zip(preds, kvs)
    ]
    row_errors = [error.mean(dim=-1) for error in errors]
    row_counts = [
        _compressed_rows(mlp, values, target_cr)
        for mlp, (_, values) in zip(mlps, kvs)
    ]
    masks = []
    for error, rows in zip(row_errors, row_counts):
        by_batch = error.flatten(start_dim=1)
        threshold = torch.topk(
            by_batch,
            rows,
            largest=False,
        ).values[:, -1]
        masks.append(error <= threshold[:, None, None])
    return sum(
        (error * mask.unsqueeze(-1)).mean()
        for error, mask in zip(errors, masks)
    )


def _adam_step(params, grads, means, variances, lr, step):
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    next_means = [
        beta1 * mean + (1.0 - beta1) * grad for mean, grad in zip(means, grads)
    ]
    next_variances = [
        beta2 * variance + (1.0 - beta2) * grad.square()
        for variance, grad in zip(variances, grads)
    ]
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    next_params = [
        param
        - lr / correction1 * mean
        / (variance.sqrt() / math.sqrt(correction2) + epsilon)
        for param, mean, variance in zip(params, next_means, next_variances)
    ]
    return next_params, next_means, next_variances


def inner_loop(
    mlps,
    kvs,
    lr,
    steps: int,
    *,
    residual_cr: float,
):
    """Run first-order functional Adam and return the final-only objective."""
    meta_params = [param for mlp in mlps for param in trainable_params(mlp)]
    dtype = kvs[0][0].dtype
    params = [
        param.detach().to(dtype=dtype).clone().requires_grad_(True)
        for param in meta_params
    ]
    names_by_mlp = []
    for mlp in mlps:
        names = [f"weights.{idx}" for idx in range(len(mlp.weights))]
        names += [f"biases.{idx}" for idx in range(len(mlp.biases))]
        if hasattr(mlp, "W_linear"):
            names.append("W_linear")
        names_by_mlp.append(names)
    preds = _predict(mlps, kvs, params, names_by_mlp)
    loss = _mse(preds, kvs)
    initial_loss = loss

    means = [torch.zeros_like(param) for param in params]
    variances = [torch.zeros_like(param) for param in params]
    for step in range(1, steps + 1):
        grads = torch.autograd.grad(loss, params)
        params, means, variances = _adam_step(
            params,
            grads,
            means,
            variances,
            lr,
            step,
        )
        if step < steps:
            preds = _predict(mlps, kvs, params, names_by_mlp)
            loss = _mse(preds, kvs)

    if steps:
        preds = _predict(mlps, kvs, params, names_by_mlp)
        loss = _mse(preds, kvs)
    final_loss = loss
    objective = _residual_loss(
        mlps,
        kvs,
        preds,
        residual_cr,
    )

    grads = torch.autograd.grad(
        objective,
        params,
        allow_unused=True,
    )
    return meta_params, {
        "initial_support_loss": initial_loss.detach().item(),
        "final_support_loss": final_loss.detach().item(),
        "meta_objective": objective.detach().item(),
        "param_grads": grads,
    }


def add_grad(param, grad) -> None:
    if grad is None:
        return
    grad = grad.detach().to(dtype=param.dtype)
    if param.grad is None:
        param.grad = grad.clone()
    else:
        param.grad.add_(grad)
