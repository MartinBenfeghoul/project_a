from .base import SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Adam, SGD
from torch.nn.functional import mse_loss
from torch.func import functional_call
import torch
from model.mlp import MLP
from typing import Any

LOSS_FUNC = {"mse": mse_loss}

OPTIMIZER = {"adam": Adam, "sgd": SGD}


class MLPValueLayer(SingleTensorDynamicLayer):
    def __init__(
        self,
        mlp_num_layers: int,
        mlp_hidden_factor: int,
        mlp_num_heads: int,
        per_sequence: bool = False,
        target_perc: float | None = None,
        threshold: float | None = None,
        optimizer_cls: str = "adam",
        num_epochs: int = 5,
        lr: float = 1.0e-3,
        loss_func: str = "mse",
        meta_weights: dict | None = None,
        meta_inner_lrs: list | None = None,
        un_rope: bool = False,
        global_compression: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_factor = mlp_hidden_factor
        self.mlp_num_heads = mlp_num_heads
        self.per_sequence = per_sequence
        self.target_perc = target_perc
        self.threshold = threshold
        self.global_compression = global_compression

        self.loss_func = LOSS_FUNC[loss_func]
        self.optimizer_cls = OPTIMIZER[optimizer_cls]
        self.num_epochs = num_epochs
        self.lr = lr
        self.meta_weights = meta_weights
        self.meta_inner_lrs = meta_inner_lrs

        self.un_rope = un_rope

        self.mlp = None
        self.indices = None
        self.value_residuals = None
        self.is_compressed = False
        self.prefill = True
        self.compressed_len = 0

    def lazy_initialization(self, value_states: torch.Tensor) -> None:

        super().lazy_initialization(value_states)

        _, self.num_heads, _, self.head_dim = value_states.shape

        self.indices = torch.tensor(
            [], dtype=torch.long, device=value_states.device
        )

        self.value_residuals = torch.tensor(
            [], dtype=value_states.dtype, device=value_states.device
        )

        self.mlp = MLP(
            head_dim=self.head_dim,
            num_layers=self.mlp_num_layers,
            hidden_factor=self.mlp_hidden_factor,
            num_heads=self.mlp_num_heads,
            per_sequence=self.per_sequence,
            batch_size=value_states.shape[0] if self.per_sequence else None,
            deterministic_init=self.meta_weights is None,
        ).to(device=value_states.device, dtype=value_states.dtype)

        if self.meta_weights is not None:
            self.mlp.load_state_dict(self.meta_weights)

    def train_mlp(self, keys: torch.Tensor) -> None:
        with torch.enable_grad():
            values = self.tensor.detach()
            keys = keys.detach()

            if self.meta_inner_lrs is not None:
                n = self.mlp.num_layers
                # meta_inner_lrs ordered: weights[0..n-1] then biases[0..n-1]
                weights = [
                    p.detach().clone().requires_grad_(True)
                    for p in self.mlp.weights
                ]
                biases = [
                    p.detach().clone().requires_grad_(True)
                    for p in self.mlp.biases
                ]
                weight_lrs = [
                    lr.to(device=keys.device, dtype=keys.dtype)
                    for lr in self.meta_inner_lrs[:n]
                ]
                bias_lrs = [
                    lr.to(device=keys.device, dtype=keys.dtype)
                    for lr in self.meta_inner_lrs[n:]
                ]

                for _ in range(self.num_epochs):
                    params = {f"weights.{i}": w for i, w in enumerate(weights)}
                    params |= {f"biases.{i}": b for i, b in enumerate(biases)}
                    # keys/values shape: [num_sequences, num_head, num_token, head_dim]
                    v_hat = functional_call(self.mlp, params, (keys,))
                    loss = self.loss_func(v_hat, values)
                    grads = torch.autograd.grad(
                        loss, weights + biases, create_graph=False
                    )
                    weight_grads, bias_grads = grads[:n], grads[n:]
                    weights = [
                        w - lr * g
                        for w, lr, g in zip(weights, weight_lrs, weight_grads)
                    ]
                    biases = [
                        b - lr * g
                        for b, lr, g in zip(biases, bias_lrs, bias_grads)
                    ]
                    weights = [w.detach().requires_grad_(True) for w in weights]
                    biases = [b.detach().requires_grad_(True) for b in biases]

                with torch.no_grad():
                    for p, w in zip(self.mlp.weights, weights):
                        p.copy_(w)
                    for p, b in zip(self.mlp.biases, biases):
                        p.copy_(b)
            else:
                all_params = list(self.mlp.parameters())
                optimizer = self.optimizer_cls(all_params, lr=self.lr)
                for _ in range(self.num_epochs):
                    optimizer.zero_grad()
                    # keys/values shape: [num_sequences, num_head, num_token, head_dim]
                    v_hat = self.mlp(keys)
                    loss = self.loss_func(v_hat, values)
                    loss.backward()
                    optimizer.step()

    def compress(self, keys: torch.Tensor) -> None:
        v_approx = self.mlp(keys)
        errors = self.loss_func(self.tensor, v_approx, reduction="none").mean(
            dim=-1
        )
        self.compressed_len = self.tensor.shape[2]

        if self.global_compression:
            self.errors = errors
            self.value_residuals = (self.tensor - v_approx).detach()
            self.tensor = self.tensor.new_empty(
                (*self.tensor.shape[:2], 0, self.tensor.shape[3])
            )
            return

        if self.threshold is None and self.target_perc is None:
            raise ValueError(
                "MLPValueLayer requires either a threshold or target_perc to compress values"
            )

        if self.target_perc is not None:
            B = errors.shape[0]
            errors_b = errors.view(B, -1)
            k = int(errors_b.shape[1] * (self.target_perc / 100))
            thresh = torch.topk(errors_b, k, largest=False).values[:, -1]
            mask = errors > thresh[:, None, None]
        else:
            mask = errors > self.threshold

        self.indices = mask.nonzero(as_tuple=True)
        b, h, t = self.indices
        self.value_residuals = (
            self.tensor[b, h, t] - v_approx[b, h, t]
        ).detach()
        self.tensor = self.tensor.new_empty(
            (*self.tensor.shape[:2], 0, self.tensor.shape[3])
        )
        self.is_compressed = True

    def decompress(self, keys: torch.Tensor, temp: bool = True) -> torch.Tensor:
        values = self.mlp(keys[:, :, : self.compressed_len, :])
        b, h, t = self.indices
        values[b, h, t] += self.value_residuals
        if not temp:
            self.tensor = values
            self._reset_residuals()
        return values

    def _reset_residuals(self):
        self.is_compressed = False
        self.value_residuals = self.value_residuals.new_empty(0)
        self.indices = (
            self.indices[0][:0],
            self.indices[1][:0],
            self.indices[2][:0],
        )

    def update(
        self,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:

        if cache_kwargs is None or "keys" not in cache_kwargs:
            raise ValueError("MLPValueLayer requires keys in cache_kwargs")

        keys = cache_kwargs["keys"]
        if self.un_rope:
            keys_for_mlp = self._undo_rope(
                keys, cache_kwargs,
                prefill=self.prefill, compressed_len=self.compressed_len,
            )
        else:
            keys_for_mlp = keys
        values = super().update(value_states)

        if self.prefill:
            self.train_mlp(keys_for_mlp)
            self.compress(keys_for_mlp)
            self.prefill = False
            return values
        elif self.is_compressed:
            decomp_values = self.decompress(keys_for_mlp)
            decomp_values = torch.cat([decomp_values, self.tensor], dim=-2)
            return decomp_values
        else:
            raise Exception(
                "Prefill is set to False but the values were not compressed."
            )

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop not implemented")
        logical_len = self.get_seq_length()
        if logical_len <= max_length:
            return

        if self.is_compressed:
            # case 1: crop inside compressed prefix
            if self.compressed_len >= max_length:
                b, h, t = self.indices
                keep = t < max_length
                self.indices = (b[keep], h[keep], t[keep])
                self.value_residuals = self.value_residuals[keep]
                self.compressed_len = max_length
                # suffix is emptied
                self.tensor = self.tensor[..., :0, :]
                return
            # case 2: crop suffix only
            suffix_max = max_length - self.compressed_len

            if suffix_max < 0:
                suffix_max = 0

            if self.tensor.shape[2] > suffix_max:
                self.tensor = self.tensor[..., :suffix_max, :]
            return

        # uncompressed we use parent as everything is store in self.tensor
        super().crop(max_length)


class MLPValueCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        num_layers_per_mlp: list[int],
        hidden_factors_per_mlp: list[int],
        num_heads_per_mlp: list[int],
        target_perc: list[float],
        target_model_num_heads: int = 8,
        per_sequence: bool = False,
        lr: float = 1e-3,
        optimizer: str = "adam",
        loss_func: str = "mse",
        num_epochs: int = 5,
        meta_weights_path: str | None = None,
        un_rope: bool = False,
        rope_theta: float = 500_000.0,
        global_compression: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert (
            len(num_layers_per_mlp)
            == len(hidden_factors_per_mlp)
            == len(num_heads_per_mlp)
            == len(target_perc)
        )

        self.num_layers_per_mlp = num_layers_per_mlp
        self.hidden_factors_per_mlp = hidden_factors_per_mlp
        self.num_heads_per_mlp = num_heads_per_mlp
        self.target_perc = target_perc
        self.per_sequence = per_sequence

        self.target_model_num_heads = target_model_num_heads

        self.lr = lr

        self.optimizer_cls = optimizer
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.un_rope = un_rope
        self.rope_theta = rope_theta
        self.global_compression = global_compression
        self._global_compression_done = False
        self.comp_ratio = 0

        if meta_weights_path is not None:
            checkpoint = torch.load(meta_weights_path, map_location="cpu")
            self._meta_weights: dict[int, dict] = {
                int(k.split("_")[1]): v
                for k, v in checkpoint.items()
                if k.startswith("layer_")
            }
            if "inner_lr_params" in checkpoint:
                flat_lrs = checkpoint["inner_lr_params"]
                self._meta_inner_lrs: dict[int, list] = {}
                offset = 0
                for i, n_mlp in enumerate(num_layers_per_mlp):
                    chunk = 2 * n_mlp
                    self._meta_inner_lrs[i] = flat_lrs[offset : offset + chunk]
                    offset += chunk
            else:
                self._meta_inner_lrs = {}
        else:
            self._meta_weights = {}
            self._meta_inner_lrs = {}

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        return MLPValueLayer(
            mlp_num_layers=self.num_layers_per_mlp[layer_idx],
            mlp_hidden_factor=self.hidden_factors_per_mlp[layer_idx],
            mlp_num_heads=self.num_heads_per_mlp[layer_idx],
            target_perc=(
                None if self.global_compression else self.target_perc[layer_idx]
            ),
            per_sequence=self.per_sequence,
            loss_func=self.loss_func,
            num_epochs=self.num_epochs,
            lr=self.lr,
            optimizer_cls=self.optimizer_cls,
            meta_weights=self._meta_weights.get(layer_idx),
            meta_inner_lrs=self._meta_inner_lrs.get(layer_idx),
            un_rope=self.un_rope,
            rope_theta=self.rope_theta,
            global_compression=self.global_compression,
        )

    def _run_global_compression(self):
        all_errors = torch.cat(
            [layer.errors.reshape(-1) for layer in self.layers]
        )
        global_perc = sum(self.target_perc) / len(self.target_perc)
        k = int(all_errors.numel() * (global_perc / 100))
        thresh = torch.topk(all_errors, k, largest=False).values[-1]

        for layer in self.layers:
            mask = layer.errors > thresh
            layer.indices = mask.nonzero(as_tuple=True)
            b, h, t = layer.indices
            layer.value_residuals = layer.value_residuals[b, h, t]
            layer.is_compressed = True

        self._global_compression_done = True

    def update(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            new_idx = len(self.layers)
            self.layers.append(self._build_layer(new_idx))

        values = self.layers[layer_idx].update(
            value_states=value_states,
            cache_kwargs=cache_kwargs,
        )

        if self.global_compression and not self._global_compression_done:
            if layer_idx == len(self.num_layers_per_mlp) - 1:
                self._run_global_compression()

        return values

    def calc_compression_ratio(self) -> float:

        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            h, d = self.target_model_num_heads, layer.head_dim
            t = (
                layer.compressed_len + layer.tensor.shape[2]
                if layer.tensor.numel()
                else layer.compressed_len
            )

            original = h * t * d

            if not layer.is_compressed:
                original_total += original
                compressed_total += original
                continue

            num_params = sum(p.numel() for p in layer.mlp.parameters())
            num_stored = layer.indices[0].numel() if layer.is_compressed else 0

            compressed = num_params + num_stored * d + num_stored * 3

            original_total += original
            compressed_total += compressed

        assert compressed_total != 0

        return original_total / compressed_total


VALUE_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "mlp": MLPValueCache,
}
