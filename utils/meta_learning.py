from dataclasses import dataclass

import torch
from torch.func import functional_call


@dataclass(frozen=True)
class MetaLearningLayerInit:
    """Meta-learned initialisation for one value-cache MLP layer."""

    weights: dict | None = None
    inner_lrs: list[torch.Tensor] | None = None

    @property
    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def has_inner_lrs(self) -> bool:
        return self.inner_lrs is not None


class MetaLearningInit:
    """Checkpoint-backed lookup for per-layer meta-learned initialisation."""

    def __init__(self, layers: dict[int, MetaLearningLayerInit] | None = None):
        self.layers = layers or {}

    @classmethod
    def empty(cls) -> "MetaLearningInit":
        return cls()

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        num_layers_per_mlp: list[int],
        use_residual: bool,
        freeze_W_linear: bool,
    ) -> "MetaLearningInit":
        checkpoint = torch.load(path, map_location="cpu")
        layer_weights = _extract_meta_layer_weights(checkpoint)
        layer_inner_lrs = _split_meta_inner_lrs(
            checkpoint=checkpoint,
            layer_weights=layer_weights,
            num_layers_per_mlp=num_layers_per_mlp,
            use_residual=use_residual,
            freeze_W_linear=freeze_W_linear,
        )
        layer_indices = set(layer_weights) | set(layer_inner_lrs)
        return cls(
            {
                idx: MetaLearningLayerInit(
                    weights=layer_weights.get(idx),
                    inner_lrs=layer_inner_lrs.get(idx),
                )
                for idx in layer_indices
            }
        )

    def for_layer(self, layer_idx: int) -> MetaLearningLayerInit:
        return self.layers.get(layer_idx, MetaLearningLayerInit())


def adapt_mlp_with_meta_lrs(
    *,
    mlp,
    keys: torch.Tensor,
    values: torch.Tensor,
    loss_func,
    num_epochs: int,
    meta_init: MetaLearningLayerInit,
    use_residual: bool,
    freeze_W_linear: bool,
    early_stopping_tol: float | None,
) -> None:
    weight_lrs, bias_lrs, w_linear_lr = _meta_lr_tensors(
        mlp=mlp,
        keys=keys,
        meta_init=meta_init,
        use_residual=use_residual,
        freeze_W_linear=freeze_W_linear,
    )
    weights, biases, w_linear = _trainable_meta_params(
        mlp=mlp,
        include_w_linear=w_linear_lr is not None,
    )

    prev_loss = float("inf")
    for _ in range(num_epochs):
        params = _functional_params(weights, biases, w_linear)
        v_hat = functional_call(mlp, params, (keys,))
        loss = loss_func(v_hat, values)
        weights, biases, w_linear = _meta_gradient_step(
            mlp=mlp,
            loss=loss,
            weights=weights,
            biases=biases,
            w_linear=w_linear,
            weight_lrs=weight_lrs,
            bias_lrs=bias_lrs,
            w_linear_lr=w_linear_lr,
        )
        loss_val = loss.item()
        improvement = (prev_loss - loss_val) / (prev_loss + 1e-8)
        if improvement < early_stopping_tol:
            break

    _load_functional_params(mlp, weights, biases, w_linear)


def _extract_meta_layer_weights(checkpoint: dict) -> dict[int, dict]:
    return {
        int(k.split("_")[1]): v
        for k, v in checkpoint.items()
        if k.startswith("layer_")
    }


def _split_meta_inner_lrs(
    checkpoint: dict,
    layer_weights: dict[int, dict],
    num_layers_per_mlp: list[int],
    use_residual: bool,
    freeze_W_linear: bool,
) -> dict[int, list[torch.Tensor]]:
    if "inner_lr_params" not in checkpoint:
        return {}

    flat_lrs = checkpoint["inner_lr_params"]
    layer_inner_lrs = {}
    offset = 0
    for layer_idx, n_mlp in enumerate(num_layers_per_mlp):
        layer_state = layer_weights.get(layer_idx, {})
        has_w_linear = "W_linear" in layer_state and use_residual and not freeze_W_linear
        chunk = 2 * n_mlp + int(has_w_linear)
        layer_inner_lrs[layer_idx] = flat_lrs[offset: offset + chunk]
        offset += chunk
    return layer_inner_lrs


def _meta_lr_tensors(
    *,
    mlp,
    keys: torch.Tensor,
    meta_init: MetaLearningLayerInit,
    use_residual: bool,
    freeze_W_linear: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    n = mlp.num_layers
    inner_lrs = meta_init.inner_lrs
    weight_lrs = [lr.to(device=keys.device, dtype=keys.dtype) for lr in inner_lrs[:n]]
    bias_lrs = [lr.to(device=keys.device, dtype=keys.dtype) for lr in inner_lrs[n: 2 * n]]
    w_linear_lr = None

    has_w_linear_lr = use_residual and not freeze_W_linear and len(inner_lrs) > 2 * n
    if has_w_linear_lr:
        w_linear_lr = inner_lrs[2 * n].to(device=keys.device, dtype=keys.dtype)

    return weight_lrs, bias_lrs, w_linear_lr


def _trainable_meta_params(
    *,
    mlp,
    include_w_linear: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    weights = [p.detach().clone().requires_grad_(True) for p in mlp.weights]
    biases = [p.detach().clone().requires_grad_(True) for p in mlp.biases]
    w_linear = None

    if include_w_linear:
        w_linear = mlp.W_linear.detach().clone().requires_grad_(True)

    return weights, biases, w_linear


def _functional_params(
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    w_linear: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    params = {f"weights.{i}": w for i, w in enumerate(weights)}
    params |= {f"biases.{i}": b for i, b in enumerate(biases)}
    if w_linear is not None:
        params["W_linear"] = w_linear
    return params


def _meta_gradient_step(
    *,
    mlp,
    loss: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    w_linear: torch.Tensor | None,
    weight_lrs: list[torch.Tensor],
    bias_lrs: list[torch.Tensor],
    w_linear_lr: torch.Tensor | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    n = mlp.num_layers
    grad_targets = weights + biases + ([w_linear] if w_linear is not None else [])
    grads = torch.autograd.grad(loss, grad_targets, create_graph=False)
    weight_grads, bias_grads = grads[:n], grads[n: 2 * n]

    weights = [w - lr * g for w, lr, g in zip(weights, weight_lrs, weight_grads)]
    biases = [b - lr * g for b, lr, g in zip(biases, bias_lrs, bias_grads)]
    if w_linear is not None:
        w_linear = w_linear - w_linear_lr * grads[-1]

    weights = [w.detach().requires_grad_(True) for w in weights]
    biases = [b.detach().requires_grad_(True) for b in biases]
    if w_linear is not None:
        w_linear = w_linear.detach().requires_grad_(True)

    return weights, biases, w_linear


def _load_functional_params(
    mlp,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    w_linear: torch.Tensor | None,
) -> None:
    with torch.no_grad():
        for p, w in zip(mlp.weights, weights):
            p.copy_(w)
        for p, b in zip(mlp.biases, biases):
            p.copy_(b)
        if w_linear is not None:
            mlp.W_linear.copy_(w_linear)

