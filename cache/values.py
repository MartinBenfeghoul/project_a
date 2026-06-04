from .base import SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Adam, AdamW, SGD
from torch.nn.functional import mse_loss
import torch
from model.mlp import MLP
from typing import Any
from utils import (
    LearnedInit,
    LearnedLayerInit,
    adapt_mlp_with_meta_lrs,
)

from .turboquant import TurboQuantCache
from utils.turboquant import init_compressor

LOSS_FUNC = {"mse": mse_loss}

OPTIMIZER = {"adam": Adam, "adamw": AdamW, "sgd": SGD}


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
        learned_init: LearnedLayerInit | None = None,
        un_rope: bool = False,
        global_compression: bool = False,
        use_residual: bool = False,
        intermediate_activation: str = "relu",
        W_linear_init: torch.Tensor | None = None,
        linear_only: bool = False,
        freeze_W_linear: bool = True,
        target_cr: float | None = None,
        turboquant_residuals: bool = False,
        compressor_bits: int = 3,
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
        self.learned_init = learned_init or LearnedLayerInit()

        self.un_rope = un_rope
        self.use_residual = use_residual
        self.W_linear_init = W_linear_init
        self.linear_only = linear_only
        self.freeze_W_linear = freeze_W_linear
        self.intermediate_activation = intermediate_activation

        self.mlp = None
        self.indices = None
        self.value_residuals = None
        self.is_compressed = False
        self.prefill = True
        self.compressed_len = 0
        self.rope_cos: torch.Tensor | None = None
        self.rope_sin: torch.Tensor | None = None
        self.key_mean: torch.Tensor | None = None
        self.key_std: torch.Tensor | None = None
        self.target_cr = target_cr
        self._num_params = None
        self.turboquant_residuals = turboquant_residuals
        self.compressor_bits = compressor_bits
        self.compressor = None
        self.mlp_training_history = []

    def lazy_initialization(self, value_states: torch.Tensor) -> None:

        super().lazy_initialization(value_states)

        _, self.num_heads, _, self.head_dim = value_states.shape
        self.compressor = init_compressor(
            self.turboquant_residuals,
            self.compressor_bits,
            value_states.shape[-1],
            value_states.device,
        )

        self.recon_mse = None
        self.indices = torch.tensor(
            [], dtype=torch.long, device=value_states.device
        )

        self.value_residuals = torch.tensor(
            [], dtype=value_states.dtype, device=value_states.device
        )

        if not self.linear_only:
            checkpoint_w_linear = (
                self.learned_init.weights.get("W_linear")
                if self.learned_init.has_weights
                else None
            )
            per_head_residual = (
                (self.W_linear_init is not None and self.W_linear_init.ndim == 3)
                or (checkpoint_w_linear is not None and checkpoint_w_linear.ndim == 3)
            )
            self.mlp = MLP(
                head_dim=self.head_dim,
                num_layers=self.mlp_num_layers,
                hidden_factor=self.mlp_hidden_factor,
                num_heads=self.mlp_num_heads,
                per_sequence=self.per_sequence,
                batch_size=value_states.shape[0] if self.per_sequence else None,
                deterministic_init=not self.learned_init.has_weights,
                use_residual=self.use_residual,
                per_head_residual=per_head_residual,
                intermediate_activation=self.intermediate_activation,
            ).to(device=value_states.device, dtype=value_states.dtype)

            self._num_params = sum(p.numel() for p in self.mlp.parameters())

            if self.learned_init.has_weights:
                self.mlp.load_state_dict(self.learned_init.weights)

            if self.use_residual and self.W_linear_init is not None:
                with torch.no_grad():
                    self.mlp.W_linear.copy_(
                        self.W_linear_init.to(
                            device=value_states.device, dtype=value_states.dtype
                        )
                    )

    def _encode_residuals(self, residuals):
        if self.compressor is None:
            return residuals
        return self.compressor.encode(residuals)

    def _decode_residuals(self):
        if self.compressor is not None:
            return self.compressor.decode(self.value_residuals)
        return self.value_residuals

    def _empty_residuals(self):
        # TODO: if quantisation of residual is enabled, this will break since
        # value_residuals are CompressorParams, not tensors
        if self.compressor is not None:
            return torch.empty(
                0,
                dtype=self.value_residuals.dtype,
                device=self.value_residuals.indices.device,
            )
        return self.value_residuals.new_empty(0)

    def residual_storage_nbytes(self, num_stored, head_dim, dtype):
        if self.compressor is not None:
            return self.compressor.memory_nbytes(self.value_residuals)
        dtype_size = torch.finfo(dtype).bits / 8
        return num_stored * head_dim * dtype_size

    def train_mlp(
        self,
        keys: torch.Tensor,
    ) -> None:
        self.mlp_training_history = []
        if self.linear_only:
            return
        with torch.enable_grad():
            keys = keys.detach()
            values = self.tensor.detach()
            if self.learned_init.has_inner_lrs:
                adapt_mlp_with_meta_lrs(
                    mlp=self.mlp,
                    keys=keys,
                    values=values,
                    loss_func=self.loss_func,
                    num_epochs=self.num_epochs,
                    learned_init=self.learned_init,
                    use_residual=self.use_residual,
                    freeze_W_linear=self.freeze_W_linear,
                )
                with torch.no_grad():
                    self.recon_mse = self.loss_func(
                        self.mlp(keys), values
                    ).item()
                self.mlp_training_history.append(
                    {
                        "epoch": self.num_epochs,
                        "value_recon_mse": self.recon_mse,
                    }
                )
            else:
                all_params = [
                    p
                    for name, p in self.mlp.named_parameters()
                    if not (self.freeze_W_linear and name == "W_linear")
                ]
                use_fused = self.optimizer_cls is Adam and torch.cuda.is_available()
                optimizer = self.optimizer_cls(all_params, lr=self.lr, fused=use_fused)
                for epoch_idx in range(self.num_epochs):
                    optimizer.zero_grad()
                    # keys/values shape: [num_sequences, num_head, num_token, head_dim]
                    v_hat = self.mlp(keys)
                    loss = self.loss_func(v_hat, values)
                    loss.backward()
                    optimizer.step()
                    loss_val = loss.detach()
                    self.mlp_training_history.append(
                        {"epoch": epoch_idx + 1, "value_recon_mse": loss_val}
                    )

                optimizer.zero_grad(set_to_none=True)
                self.recon_mse = loss_val

    def _linear_approx(self, keys: torch.Tensor) -> torch.Tensor:
        w = self.W_linear_init.to(device=keys.device, dtype=keys.dtype)
        if w.ndim == 3:
            return torch.einsum("bhtd,hde->bhte", keys, w)
        return torch.einsum("bhtd,hdqe->bqte", keys, w)

    def compute_new_perc(self, keys):
        b, h, t, d = keys.shape
        num_params = sum(p.numel() for p in self.mlp.parameters())
        delta_dtype = self.tensor.dtype
        value_dtype_size = torch.finfo(self.tensor.dtype).bits / 8
        index_dtype_size = torch.iinfo(torch.int32).bits / 8
        if self.compressor is not None:
            import math

            indices_per_byte = 8 // self.compressor.bits
            per_delta_bytes = (
                math.ceil(d / indices_per_byte)
                + value_dtype_size
                + index_dtype_size
            )  # packed indices + norm + sparse index
        else:
            delta_dtype_size = torch.finfo(delta_dtype).bits / 8
            per_delta_bytes = d * delta_dtype_size + index_dtype_size
        orig = b * h * t * d * value_dtype_size
        target_compressed = orig / self.target_cr
        allowed_delta_storage = target_compressed - (
            num_params * value_dtype_size
        )
        n_deltas = int(allowed_delta_storage / per_delta_bytes)
        n_deltas = max(0, min(n_deltas, b * h * t))
        self.target_perc = 100 * (1 - (n_deltas / (b * h * t)))

    def compress(self, keys: torch.Tensor) -> None:
        self._B, self._H, self._T, _ = keys.shape

        if self.linear_only:
            self.compressed_len = self.tensor.shape[2]
            self.indices = torch.empty(
                0, dtype=torch.int64, device=self.tensor.device
            )
            self.tensor = self.tensor.new_empty(
                (*self.tensor.shape[:2], 0, self.tensor.shape[3])
            )
            self.is_compressed = True
            return

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
            k = max(1, int(errors_b.shape[1] * (self.target_perc / 100)))
            thresh = torch.topk(errors_b, k, largest=False).values[:, -1]
            mask = errors > thresh[:, None, None]
        else:
            mask = errors > self.threshold

        b, h, t = mask.nonzero(as_tuple=True)
        max_index = self._B * self._H * self._T - 1
        idx_dtype = (
            torch.int32
            if max_index < torch.iinfo(torch.int32).max
            else torch.int64
        )
        self.indices = (b * (self._H * self._T) + h * self._T + t).to(idx_dtype)
        self.value_residuals = self._encode_residuals(
            (self.tensor[b, h, t] - v_approx[b, h, t]).detach()
        )
        self.tensor = self.tensor.new_empty(
            (*self.tensor.shape[:2], 0, self.tensor.shape[3])
        )
        self.is_compressed = True

    def decompress(
        self, keys: torch.Tensor, reset: bool = False
    ) -> torch.Tensor:
        prefix_keys = keys[:, :, : self.compressed_len, :]
        values = (
            self._linear_approx(prefix_keys)
            if self.linear_only
            else self.mlp(prefix_keys)
        )
        t = self.indices % self._T
        b = (self.indices // self._T) // self._H
        h = (self.indices // self._T) % self._H
        if b.numel() > 0:
            values[b, h, t] += self._decode_residuals()
        if reset:
            self.tensor = values
            self._reset_residuals()
        return values

    def _reset_residuals(self):
        self.is_compressed = False
        self.value_residuals = self._empty_residuals()
        self.indices = self.indices.new_empty(0)

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
                keys,
                cache_kwargs,
                prefill=self.prefill,
                compressed_len=self.compressed_len,
            )
        else:
            keys_for_mlp = keys
        values = super().update(value_states)

        if self.prefill:
            if self.target_cr and not self.linear_only:
                self.compute_new_perc(keys_for_mlp)
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
        per_sequence: bool = False,
        lr: float = 1e-3,
        optimizer: str = "adam",
        loss_func: str = "mse",
        num_epochs: int = 5,
        meta_weights_path: str | None = None,
        value_mlp_weights_path: str | None = None,
        un_rope: bool = False,
        rope_theta: float = 500_000.0,
        global_compression: bool = False,
        use_residual: bool = False,
        W_linear_per_layer: list[torch.Tensor] | None = None,
        intermediate_activation: str = "relu",
        linear_only: bool = False,
        freeze_W_linear: bool = True,
        target_cr: float | None = None,
        turboquant_residuals: bool = False,
        compressor_bits: int = 3,
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

        self.lr = lr

        self.optimizer_cls = optimizer
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.un_rope = un_rope
        self.rope_theta = rope_theta
        self.global_compression = global_compression
        self.use_residual = use_residual
        self.W_linear_per_layer = W_linear_per_layer
        self.intermediate_activation = intermediate_activation
        self.linear_only = linear_only
        self.freeze_W_linear = freeze_W_linear

        if use_residual and W_linear_per_layer is not None:
            if not un_rope:
                raise ValueError(
                    "use_residual with W_linear_per_layer initialisation requires un_rope=True."
                )
        if linear_only:
            if W_linear_per_layer is None:
                raise ValueError(
                    "linear_only=True requires W_linear_per_layer to be provided."
                )
            if not un_rope:
                raise ValueError("linear_only requires un_rope=True.")
            if global_compression:
                raise ValueError(
                    "linear_only is incompatible with global_compression."
                )
        self._global_compression_done = False
        self.comp_ratio = 0
        self.target_cr = target_cr
        self.turboquant_residuals = turboquant_residuals
        self.compressor_bits = compressor_bits
        self.mlp_log_events = []

        if meta_weights_path is not None and value_mlp_weights_path is not None:
            raise ValueError(
                "Use only one of meta_weights_path or value_mlp_weights_path."
        )

        self.learned_init = (
            LearnedInit.from_value_mlp_checkpoint(value_mlp_weights_path)
            if value_mlp_weights_path is not None
            else LearnedInit.from_checkpoint(
                path=meta_weights_path,
                num_layers_per_mlp=num_layers_per_mlp,
                use_residual=use_residual,
                freeze_W_linear=freeze_W_linear,
            )
            if meta_weights_path is not None
            else LearnedInit.empty()
        )

    def _ensure_layers(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            new_idx = len(self.layers)
            self.layers.append(self._build_layer(new_idx))

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        learned_init = self.learned_init.for_layer(layer_idx)
        checkpoint_has_w_linear = (
            learned_init.has_weights and "W_linear" in learned_init.weights
        )
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
            learned_init=learned_init,
            un_rope=self.un_rope,
            rope_theta=self.rope_theta,
            global_compression=self.global_compression,
            use_residual=self.use_residual,
            W_linear_init=(
                self.W_linear_per_layer[layer_idx]
                if self.W_linear_per_layer is not None and not checkpoint_has_w_linear
                else None
            ),
            intermediate_activation=self.intermediate_activation,
            linear_only=self.linear_only,
            freeze_W_linear=self.freeze_W_linear,
            target_cr=self.target_cr,
            turboquant_residuals=self.turboquant_residuals,
            compressor_bits=self.compressor_bits,
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
            b, h, t = mask.nonzero(as_tuple=True)
            B, H, T = layer.errors.shape
            max_index = B * H * T - 1
            idx_dtype = (
                torch.int32
                if max_index < torch.iinfo(torch.int32).max
                else torch.int64
            )
            layer.indices = (b * (H * T) + h * T + t).to(idx_dtype)
            layer.value_residuals = layer._encode_residuals(
                layer.value_residuals[b, h, t]
            )
            layer.is_compressed = True

        self._global_compression_done = True

    def update(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return self._update_one(value_states, layer_idx, cache_kwargs)

    def _update_one(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self._ensure_layers(layer_idx)
        layer = self.layers[layer_idx]
        values = layer.update(
            value_states=value_states,
            cache_kwargs=cache_kwargs,
        )
        self._collect_layer_training_history(
            layer_idx, value_states, cache_kwargs
        )

        if self.global_compression and not self._global_compression_done:
            if layer_idx == len(self.num_layers_per_mlp) - 1:
                self._run_global_compression()

        return values

    def update_prefill_group(
        self,
        updates: list[tuple[int, torch.Tensor, dict[str, Any] | None]],
    ) -> dict[int, torch.Tensor]:
        return {
            layer_idx: self._update_one(
                value_states,
                layer_idx,
                cache_kwargs,
            )
            for layer_idx, value_states, cache_kwargs in updates
        }

    def _collect_layer_training_history(
        self,
        layer_idx: int,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None,
    ) -> None:
        layer = self.layers[layer_idx]
        if layer.prefill or not layer.mlp_training_history:
            return

        context = cache_kwargs or {}
        for row in layer.mlp_training_history:
            self.mlp_log_events.append(
                {
                    "task_name": context.get("task_name", "unknown"),
                    "request_idx": context.get("request_idx"),
                    "seq_len": int(value_states.shape[-2]),
                    "layer_idx": layer_idx,
                    "epoch": row["epoch"],
                    "value_recon_mse": row["value_recon_mse"],
                    "target_perc": layer.target_perc,
                }
            )
        layer.mlp_training_history = []

    def calc_compression_ratio(self) -> float:

        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            h, d = layer.num_heads, layer.head_dim
            t = layer.compressed_len
            delta_dtype = layer.tensor.dtype
            value_dtype_size = torch.finfo(layer.tensor.dtype).bits / 8
            index_dtype_size = torch.iinfo(layer.indices.dtype).bits / 8

            original = h * t * d * value_dtype_size

            if not layer.is_compressed:
                original_total += original
                compressed_total += original
                continue

            if layer.linear_only:
                compressed = layer.W_linear_init.numel() * value_dtype_size
            else:
                num_params = layer._num_params
                num_stored = layer.indices.numel() if layer.is_compressed else 0
                compressed = (
                    num_params * value_dtype_size
                    + layer.residual_storage_nbytes(num_stored, d, delta_dtype)
                    + num_stored * index_dtype_size
                )

            original_total += original
            compressed_total += compressed

        assert compressed_total != 0

        return original_total / compressed_total

    @property
    def recon_mse(self) -> float | None:
        mses = [l.recon_mse for l in self.layers if l.recon_mse is not None]
        if not mses:
            return None
        return sum(mses) / len(mses)


VALUE_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "mlp": MLPValueCache,
    "turboquant": TurboQuantCache,
}
