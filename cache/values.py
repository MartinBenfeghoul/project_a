import math

from .base import SharedRopeCache, SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Adam
from torch.nn.functional import mse_loss
import torch
from model.mlp import MLP
from typing import Any, Callable
from model.meta_learning import LearnedInit, LearnedLayerInit
from numerics.quantisation import init_compressor


class MLPValueLayer(SingleTensorDynamicLayer):
    def __init__(
        self,
        target_cr: float,
        num_epochs: int = 5,
        learned_init: LearnedLayerInit | None = None,
        use_residual: bool = False,
        W_linear_init: torch.Tensor | None = None,
        rope_cache: SharedRopeCache | None = None,
        turboquant_residuals: bool = False,
        compressor_bits: int = 3,
    ):
        super().__init__(rope_cache=rope_cache)

        self.target_cr = float(target_cr)

        self.num_epochs = num_epochs
        self.learned_init = learned_init or LearnedLayerInit()

        self.use_residual = use_residual
        self.W_linear_init = W_linear_init

        self.mlp = None
        self.indices = None
        self.value_residuals = None
        self.is_compressed = False
        self.prefill = True
        self.compressed_len = 0
        self._num_params = None
        self.turboquant_residuals = turboquant_residuals
        self.compressor_bits = compressor_bits
        self.compressor = None

    def lazy_initialization(self, value_states: torch.Tensor) -> None:

        super().lazy_initialization(value_states)

        _, self.num_heads, _, self.head_dim = value_states.shape
        self.compressor = init_compressor(
            self.turboquant_residuals,
            self.compressor_bits,
            value_states.shape[-1],
            value_states.device,
        )
        self.indices = torch.tensor(
            [], dtype=torch.long, device=value_states.device
        )

        self.value_residuals = torch.tensor(
            [], dtype=value_states.dtype, device=value_states.device
        )

        if self.mlp is None:
            w_linear_init = self.W_linear_init
            if self.learned_init.has_weights:
                w_linear_init = self.learned_init.weights.get(
                    "W_linear",
                    w_linear_init,
                )
            self.mlp = MLP(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                deterministic_init=not self.learned_init.has_weights,
                use_residual=self.use_residual,
            ).to(device=value_states.device, dtype=value_states.dtype)

            self._num_params = sum(p.numel() for p in self.mlp.parameters())

            if self.learned_init.has_weights:
                weights = dict(self.learned_init.weights)
                if not self.use_residual and "W_linear" in weights:
                    weights = {
                        k: v for k, v in weights.items() if k != "W_linear"
                    }
                self.mlp.load_state_dict(weights)

            if self.use_residual and self.W_linear_init is not None:
                with torch.no_grad():
                    linear_init = self.W_linear_init.to(
                        device=value_states.device, dtype=value_states.dtype
                    )
                    self.mlp.W_linear.copy_(linear_init)

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

    def _reconstruction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        errors = mse_loss(predicted, target, reduction="none")
        valid = padding_mask[:, None, :, None]
        return (errors * valid).sum() / (
            valid.sum() * errors.size(1) * errors.size(3)
        )

    def train_mlp(
        self,
        keys: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        with torch.enable_grad():
            keys = keys.detach()
            values = self.tensor.detach()
            all_params = list(self.mlp.parameters())
            use_fused = torch.cuda.is_available()
            optimizer = Adam(all_params, lr=1e-3, fused=use_fused)
            for _ in range(self.num_epochs):
                optimizer.zero_grad()
                # keys/values shape: [num_sequences, num_head, num_token, head_dim]
                v_hat = self.mlp(keys)
                loss = self._reconstruction_loss(
                    v_hat,
                    values,
                    padding_mask,
                )
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

    def compute_residual_budget(
        self,
        padding_mask: torch.Tensor,
    ) -> int:
        """Return the residual rows that fit within the target CR budget."""
        b, h, t, d = self.tensor.shape
        value_dtype_size = torch.finfo(self.tensor.dtype).bits / 8
        max_index = b * h * t - 1
        index_dtype = (
            torch.int32
            if max_index < torch.iinfo(torch.int32).max
            else torch.int64
        )
        index_dtype_size = torch.iinfo(index_dtype).bits / 8
        if self.compressor is not None:
            indices_per_byte = 8 // self.compressor.bits
            residual_row_bytes = (
                math.ceil(d / indices_per_byte)
                + value_dtype_size
                + index_dtype_size
            )  # packed indices + norm + sparse index
        else:
            residual_row_bytes = d * value_dtype_size + index_dtype_size

        valid_rows = padding_mask.sum(dim=1, dtype=torch.long) * h
        total_valid_rows = int(valid_rows.sum().item())
        if total_valid_rows == 0:
            raise ValueError(
                "padding_mask must contain at least one valid token."
            )

        model_bytes = self._num_params * value_dtype_size
        original_bytes = total_valid_rows * d * value_dtype_size
        allowed_residual_storage = original_bytes / self.target_cr - model_bytes
        residual_rows = int(allowed_residual_storage / residual_row_bytes)
        return max(0, min(residual_rows, total_valid_rows))

    def compress(
        self,
        keys: torch.Tensor,
        padding_mask: torch.Tensor,
        residual_budget: int,
    ) -> None:
        self._B, self._H, self._T, _ = keys.shape
        self.original_token_count = int(padding_mask.sum().item())

        v_approx = self.mlp(keys)
        errors = mse_loss(self.tensor, v_approx, reduction="none").mean(dim=-1)
        valid = padding_mask[:, None, :].expand_as(errors)
        self.compressed_len = self.tensor.shape[2]

        mask = torch.zeros_like(valid)
        budget = int(residual_budget)
        valid_flat = valid.flatten()
        valid_indices = valid_flat.nonzero(as_tuple=False).flatten()
        if budget >= valid_indices.numel():
            mask = valid.clone()
        elif budget > 0:
            valid_errors = errors.flatten()[valid_indices]
            selected = torch.topk(
                valid_errors,
                budget,
                largest=True,
                sorted=False,
            ).indices
            mask.view(-1)[valid_indices[selected]] = True

        b, h, t = mask.nonzero(as_tuple=True)
        max_index = self._B * self._H * self._T - 1
        idx_dtype = (
            torch.int32
            if max_index < torch.iinfo(torch.int32).max
            else torch.int64
        )
        self.indices = (b * (self._H * self._T) + h * self._T + t).to(idx_dtype)
        stored_values = self.tensor[b, h, t] - v_approx[b, h, t]
        self.value_residuals = self._encode_residuals(stored_values.detach())
        self.tensor = self.tensor.new_empty(
            (*self.tensor.shape[:2], 0, self.tensor.shape[3])
        )
        self.is_compressed = True

    def decompress(
        self, keys: torch.Tensor, reset: bool = False
    ) -> torch.Tensor:
        prefix_keys = keys[:, :, : self.compressed_len, :]
        values = self.mlp(prefix_keys)
        t = self.indices % self._T
        b = (self.indices // self._T) // self._H
        h = (self.indices // self._T) % self._H
        if b.numel() > 0:
            stored_values = self._decode_residuals().to(values.dtype)
            values[b, h, t] += stored_values
        if reset:
            self.tensor = values
            self._reset_residuals()
        return values

    @staticmethod
    def _gather_tokens(
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return tensor.gather(
            2, positions[..., None].expand(-1, -1, -1, tensor.size(-1))
        )

    def append_decode(self, value_states: torch.Tensor) -> None:
        super().update(value_states)

    def _lookup_stored_rows(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map selected positions to stored value rows and valid matches"""
        batch_size, num_heads = positions.shape[:2]
        head_offsets = torch.arange(
            batch_size * num_heads, device=positions.device
        ).reshape(batch_size, num_heads, 1)
        flat_positions = (
            (head_offsets * self._T + positions)
            .reshape(-1)
            .to(self.indices.dtype)
        )
        stored_rows = torch.searchsorted(self.indices, flat_positions)
        valid = stored_rows < self.indices.numel()
        stored_rows.clamp_(max=self.indices.numel() - 1)
        valid &= self.indices.index_select(0, stored_rows) == flat_positions
        valid &= positions.reshape(-1) < self.compressed_len
        return stored_rows, valid

    def _append_exact_suffix_values(
        self,
        values: torch.Tensor,
        suffix_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Append exact cached values for selected suffix positions."""
        if suffix_positions.size(-1):
            suffix_positions = suffix_positions - self.compressed_len
            suffix_values = self._gather_tokens(self.tensor, suffix_positions)
            values = torch.cat([values, suffix_values], dim=2)
        return values

    @torch.no_grad()
    def retrieve_selected(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        positions = positions.to(device=keys.device, dtype=torch.long)
        prefix_len = positions.size(-1) - self.tensor.size(-2)
        prefix_positions = positions[..., :prefix_len]
        selected_keys = keys[..., :prefix_len, :]
        keys_for_mlp = self._rope_selected(
            selected_keys,
            prefix_positions,
            self.compressed_len,
            inverse=True,
        )
        values = self.mlp(keys_for_mlp)

        if self.indices.numel() > 0:
            stored_rows, valid = self._lookup_stored_rows(prefix_positions)
            stored_values = self.value_residuals.index_select(
                0, stored_rows
            ).to(values.dtype)
            values_flat = values.reshape(-1, values.size(-1))
            values_flat.add_(stored_values * valid[:, None])

        return self._append_exact_suffix_values(
            values,
            positions[..., prefix_len:],
        )

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
        if self.prefill:
            if "padding_mask" not in cache_kwargs:
                raise ValueError(
                    "MLPValueLayer requires padding_mask during prefill."
                )
            padding_mask = cache_kwargs["padding_mask"].to(
                device=keys.device,
                dtype=torch.bool,
            )
            kept_positions = cache_kwargs.get("kept_positions")
            if kept_positions is not None:
                padding_mask = padding_mask.index_select(1, kept_positions)
            if padding_mask.shape != (keys.size(0), keys.size(2)):
                raise ValueError(
                    "padding_mask must have shape [batch, sequence_length]."
                )

        keys_for_mlp = self._undo_rope(
            keys,
            cache_kwargs,
            prefill=self.prefill,
            compressed_len=self.compressed_len,
        )
        values = super().update(value_states)

        if self.prefill:
            residual_budget = self.compute_residual_budget(padding_mask)
            self.train_mlp(keys_for_mlp, padding_mask)
            self.compress(
                keys_for_mlp,
                padding_mask,
                residual_budget,
            )
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
        ddp_cache_data=None,
        *,
        target_cr: float,
        num_epochs: int = 5,
        meta_weights_path: str | None = None,
        value_mlp_weights_path: str | None = None,
        rope_cache: SharedRopeCache | None = None,
        use_residual: bool = False,
        W_linear_per_layer: (
            list[torch.Tensor] | Callable[[], list[torch.Tensor]] | None
        ) = None,
        turboquant_residuals: bool = False,
        compressor_bits: int = 3,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            rope_cache=rope_cache,
        )

        target_cr = float(target_cr)
        if not math.isfinite(target_cr) or target_cr <= 0:
            raise ValueError("target_cr must be finite and positive.")

        self.num_epochs = num_epochs
        self.use_residual = use_residual
        self.comp_ratio = 0
        self.target_cr = target_cr
        self.turboquant_residuals = turboquant_residuals
        self.compressor_bits = compressor_bits

        if meta_weights_path is not None and value_mlp_weights_path is not None:
            raise ValueError(
                "Use only one of meta_weights_path or value_mlp_weights_path."
            )

        self.learned_init = (
            LearnedInit.from_value_mlp_checkpoint(value_mlp_weights_path)
            if value_mlp_weights_path is not None
            else (
                LearnedInit.from_checkpoint(
                    path=meta_weights_path,
                )
                if meta_weights_path is not None
                else LearnedInit.empty()
            )
        )

        self.checkpoint_has_w_linear = "W_linear" in (
            self.learned_init.for_layer(0).weights or {}
        )
        if self.checkpoint_has_w_linear or not use_residual:
            W_linear_per_layer = None
        elif callable(W_linear_per_layer):
            W_linear_per_layer = W_linear_per_layer()
        self.W_linear_per_layer = W_linear_per_layer

    def _ensure_layers(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            new_idx = len(self.layers)
            self.layers.append(self._build_layer(new_idx))

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        learned_init = self.learned_init.for_layer(layer_idx)
        if self.checkpoint_has_w_linear:
            w_linear_init = None
        else:
            w_linear_init = (
                self.W_linear_per_layer[layer_idx]
                if self.W_linear_per_layer is not None
                else None
            )
        return MLPValueLayer(
            target_cr=self.target_cr,
            num_epochs=self.num_epochs,
            learned_init=learned_init,
            rope_cache=self.rope_cache,
            use_residual=self.use_residual,
            W_linear_init=w_linear_init,
            turboquant_residuals=self.turboquant_residuals,
            compressor_bits=self.compressor_bits,
        )

    def update(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return self._update_one(value_states, layer_idx, cache_kwargs)

    def supports_selective_retrieval(self, layer_idx: int) -> bool:
        return (
            layer_idx < len(self.layers)
            and not self.layers[layer_idx].prefill
            and self.layers[layer_idx].is_compressed
        )

    def append_decode(self, value_states: torch.Tensor, layer_idx: int) -> None:
        self.layers[layer_idx].append_decode(value_states)

    def retrieve_selected(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].retrieve_selected(keys, positions)

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

    def calc_compression_ratio(self) -> float:

        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            h, d = layer.num_heads, layer.head_dim
            delta_dtype = layer.tensor.dtype
            value_dtype_size = torch.finfo(layer.tensor.dtype).bits / 8
            index_dtype_size = torch.iinfo(layer.indices.dtype).bits / 8

            original = layer.original_token_count * h * d * value_dtype_size

            if not layer.is_compressed:
                original_total += original
                compressed_total += original
                continue

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
