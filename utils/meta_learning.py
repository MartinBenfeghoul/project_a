from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from .rope import compute_rope_cos_sin, inverse_rope


@dataclass(frozen=True)
class LearnedLayerInit:
    """Learned initialisation for one value-cache MLP layer."""

    weights: dict | None = None
    inner_lrs: list[torch.Tensor] | None = None

    @property
    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def has_inner_lrs(self) -> bool:
        return self.inner_lrs is not None


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
        num_layers_per_mlp: list[int],
        use_residual: bool,
        freeze_W_linear: bool,
    ) -> "LearnedInit":
        checkpoint = torch.load(path, map_location="cpu")
        layer_weights = {
            int(key.split("_")[1]): value
            for key, value in checkpoint.items()
            if key.startswith("layer_")
        }
        layer_lrs = _split_lrs(
            checkpoint,
            layer_weights,
            num_layers_per_mlp,
            use_residual,
            freeze_W_linear,
        )
        layer_indices = set(layer_weights) | set(layer_lrs)
        return cls(
            {
                idx: LearnedLayerInit(
                    weights=layer_weights.get(idx),
                    inner_lrs=layer_lrs.get(idx),
                )
                for idx in layer_indices
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


def adapt_mlp_with_meta_lrs(
    *,
    mlp,
    keys: torch.Tensor,
    values: torch.Tensor,
    loss_func,
    num_epochs: int,
    learned_init: LearnedLayerInit,
    use_residual: bool,
    freeze_W_linear: bool,
    optimizer_cls=torch.optim.SGD,
) -> None:
    n = mlp.num_layers
    saved_lrs = learned_init.inner_lrs
    params = list(mlp.weights) + list(mlp.biases)
    lrs = [lr.to(keys) for lr in saved_lrs[: 2 * n]]
    if use_residual and not freeze_W_linear and len(saved_lrs) > 2 * n:
        params.append(mlp.W_linear)
        lrs.append(saved_lrs[2 * n].to(keys))

    optimizer = optimizer_cls(
        [
            {"params": [param], "lr": float(lr)}
            for param, lr in zip(params, lrs)
        ]
    )

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_func(mlp(keys), values)
        loss.backward()
        optimizer.step()


def _split_lrs(
    state: dict,
    weights: dict[int, dict],
    depths: list[int],
    use_residual: bool,
    freeze_linear: bool,
) -> dict[int, list[torch.Tensor]]:
    if "inner_lr_params" not in state:
        return {}

    flat_lrs = state["inner_lr_params"]
    layer_lrs = {}
    offset = 0
    for layer_idx, depth in enumerate(depths):
        layer_state = weights.get(layer_idx, {})
        has_linear = (
            "W_linear" in layer_state and use_residual and not freeze_linear
        )
        chunk = 2 * depth + int(has_linear)
        layer_lrs[layer_idx] = flat_lrs[offset : offset + chunk]
        offset += chunk
    return layer_lrs


def get_rope_theta(model_config) -> float:
    rope_theta = getattr(model_config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(model_config, "rope_parameters", {}) or {}
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise ValueError("Model config does not define rope_theta")
    return float(rope_theta)



def constrain_lrs(raw_lrs, config) -> list[torch.Tensor] | None:
    if raw_lrs is None:
        return None
    lower, upper = float(config.inner_lr_min), float(config.inner_lr_max)
    return [
        lower + (upper - lower) * torch.sigmoid(param)
        for param in raw_lrs
    ]


def trainable_params(mlp) -> list[torch.Tensor]:
    params = list(mlp.weights) + list(mlp.biases)
    if hasattr(mlp, "W_linear"):
        params.append(mlp.W_linear)
    return params


def expand_lrs(mlps, layer_lrs) -> list[torch.Tensor] | None:
    if layer_lrs is None:
        return None
    return [
        layer_lr
        for mlp, layer_lr in zip(mlps, layer_lrs)
        for _ in trainable_params(mlp)
    ]


def setup_optimizer(mlps, config):
    params = [
        param
        for mlp in mlps
        for param in trainable_params(mlp)
    ]
    raw_lrs = None
    if config.learn_inner_lr:
        lower, upper = float(config.inner_lr_min), float(config.inner_lr_max)
        initial = float(config.inner_lr)
        assert lower < initial < upper
        fraction = (initial - lower) / (upper - lower)
        raw = math.log(fraction / (1.0 - fraction))
        device = params[0].device
        raw_lrs = [
            nn.Parameter(torch.tensor(raw, device=device))
            for _ in mlps
        ]
        params.extend(raw_lrs)

    return raw_lrs, torch.optim.Adam(
        params,
        lr=float(config.meta_lr),
    )


def prepare_kvs(
    cache,
    *,
    rope_theta: float,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Detach, un-RoPE, and cast support KV pairs once per batch."""
    kvs = []
    for layer in cache.layers:
        keys = layer.keys.detach()
        cos, sin = compute_rope_cos_sin(
            keys.shape[2],
            keys.shape[-1],
            rope_theta,
            keys.device,
            keys.dtype,
        )
        kvs.append(
            (
                inverse_rope(keys, cos, sin).to(dtype=dtype),
                layer.values.detach().to(dtype=dtype),
            )
        )
    return kvs


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
        F.mse_loss(pred, values)
        for pred, (_, values) in zip(preds, kvs)
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
        beta1 * mean + (1.0 - beta1) * grad
        for mean, grad in zip(means, grads)
    ]
    next_variances = [
        beta2 * variance + (1.0 - beta2) * grad.square()
        for variance, grad in zip(variances, grads)
    ]
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    next_params = [
        param
        - (lr[idx] if isinstance(lr, list) else lr)
        / correction1
        * mean
        / (variance.sqrt() / math.sqrt(correction2) + epsilon)
        for idx, (param, mean, variance) in enumerate(
            zip(params, next_means, next_variances)
        )
    ]
    return next_params, next_means, next_variances


def inner_loop(
    mlps,
    kvs,
    lr,
    steps: int,
    *,
    outer_lrs: list[nn.Parameter] | None = None,
    residual_cr: float | None = None,
):
    """Run first-order functional Adam and return the final-only objective."""
    meta_params = [
        param
        for mlp in mlps
        for param in trainable_params(mlp)
    ]
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
    objective = (
        _residual_loss(
            mlps,
            kvs,
            preds,
            residual_cr,
        )
        if residual_cr is not None
        else final_loss
    )

    targets = params + (outer_lrs or [])
    grads = torch.autograd.grad(
        objective,
        targets,
        allow_unused=True,
    )
    n_params = len(params)
    initial_value = initial_loss.detach().item()
    final_value = final_loss.detach().item()
    objective_value = (
        objective.detach().item()
        if residual_cr is not None
        else final_value
    )
    return meta_params, {
        "initial_support_loss": initial_value,
        "final_support_loss": final_value,
        "meta_objective": objective_value,
        "param_grads": grads[:n_params],
        "lr_grads": grads[n_params:],
    }


def add_grad(param, grad) -> None:
    if grad is None:
        return
    grad = grad.detach().to(dtype=param.dtype)
    if param.grad is None:
        param.grad = grad.clone()
    else:
        param.grad.add_(grad)
