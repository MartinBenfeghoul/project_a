from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam

from cache.base import SingleTensorCache, SingleTensorDynamicLayer
from cache.cache import CompressedCache


SOURCE_ALIASES = {
    "prev_layer_keys": "prev_keys",
    "prev_layer_values": "prev_values",
    "prev_layer_kv": "prev_kv",
    "current_layer_keys": "current_keys",
    "current_layer_values": "current_values",
}

class SourceTargetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_factor: int,
        num_heads: int,
    ):
        super().__init__()
        curr_dim = input_dim
        hidden_dim = hidden_factor * output_dim
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        torch.manual_seed(1)
        for layer_idx in range(num_layers):
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            weight = nn.Parameter(torch.zeros(1, num_heads, curr_dim, out_dim))
            bias = nn.Parameter(torch.zeros(1, num_heads, 1, out_dim))
            nn.init.kaiming_uniform_(weight, a=5**0.5)
            nn.init.zeros_(bias)
            self.weights.append(weight)
            self.biases.append(bias)
            curr_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, weight) + bias
            if layer_idx < len(self.weights) - 1:
                x = torch.relu(x)
        return x


class MLPPredictionLayer(SingleTensorDynamicLayer):
    def __init__(
        self,
        *,
        source: str,
        mlp_num_layers: int,
        mlp_hidden_factor: int,
        mlp_num_heads: int,
        target_cr: float = 3.0,
        num_epochs: int = 200,
        target_is_key: bool = False,
        rope_theta: float | None = None,
    ):
        super().__init__(rope_theta=rope_theta)
        self.source = SOURCE_ALIASES[source]
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_factor = mlp_hidden_factor
        self.mlp_num_heads = mlp_num_heads
        self.target_cr = target_cr
        self.num_epochs = num_epochs
        self.target_is_key = target_is_key

        self.mlp: SourceTargetMLP | None = None
        self.indices: torch.Tensor | None = None
        self.residuals: torch.Tensor | None = None
        self.source_tensor: torch.Tensor | None = None
        self.prefill = True
        self.is_compressed = False
        self.compressed_len = 0
        self.recon_mse: float | None = None

    def _source_tensor(
        self,
        cache_kwargs: dict[str, Any] | None,
        target_states: torch.Tensor,
    ) -> torch.Tensor | None:
        cache_kwargs = cache_kwargs or {}
        if self.source == "current_keys":
            source = cache_kwargs.get("current_keys")
            if source is None:
                return None
            source = self._undo_rope(source, cache_kwargs, self.prefill, self.compressed_len)
        elif self.source == "current_values":
            source = cache_kwargs.get("current_values")
        elif self.source == "prev_keys":
            source = cache_kwargs.get("prev_layer_keys")
            if source is None:
                return None
            source = self._undo_rope(source, cache_kwargs, self.prefill, self.compressed_len)
        elif self.source == "prev_values":
            source = cache_kwargs.get("prev_layer_values")
            if source is None:
                return None
        elif self.source == "prev_kv":
            prev_keys = cache_kwargs.get("prev_layer_keys")
            prev_values = cache_kwargs.get("prev_layer_values")
            if prev_keys is None or prev_values is None:
                return None
            prev_keys = self._undo_rope(prev_keys, cache_kwargs, self.prefill, self.compressed_len)
            source = torch.cat([prev_keys, prev_values], dim=-1)
        else:
            raise AssertionError(f"Unhandled source {self.source}")

        if source.shape[-2] != target_states.shape[-2] and self.prefill:
            source = source[..., -target_states.shape[-2] :, :]
        return source

    def lazy_initialization(
        self,
        target_states: torch.Tensor,
        source_states: torch.Tensor,
    ) -> None:
        super().lazy_initialization(target_states)
        _, _, _, output_dim = target_states.shape
        input_dim = source_states.shape[-1]
        self.mlp = SourceTargetMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=self.mlp_num_layers,
            hidden_factor=self.mlp_hidden_factor,
            num_heads=self.mlp_num_heads,
        ).to(device=target_states.device, dtype=target_states.dtype)
        self.indices = torch.empty(0, dtype=torch.int64, device=target_states.device)
        self.residuals = torch.empty(
            0,
            output_dim,
            dtype=target_states.dtype,
            device=target_states.device,
        )

    def _train_mlp(
        self,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
    ) -> None:
        params = list(self.mlp.parameters())
        optimizer = Adam(params, lr=1e-3, fused=torch.cuda.is_available())

        with torch.enable_grad():
            source_states = source_states.detach()
            target_states = target_states.detach()
            for _ in range(self.num_epochs):
                optimizer.zero_grad(set_to_none=True)
                pred = self.mlp(source_states)
                loss = mse_loss(pred, target_states)
                loss.backward()
                optimizer.step()
                loss_val = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        self.recon_mse = loss_val

    def _compress(self, source_states: torch.Tensor) -> None:
        pred = self.mlp(source_states)
        errors = mse_loss(self.tensor, pred, reduction="none").mean(dim=-1)
        self.compressed_len = self.tensor.shape[-2]
        if self.target_cr is not None:
            bsz, heads, seq_len, dim = self.tensor.shape
            dtype_size = torch.finfo(self.tensor.dtype).bits / 8
            index_size = torch.iinfo(torch.int32).bits / 8
            original = bsz * heads * seq_len * dim * dtype_size
            target_bytes = original / self.target_cr
            param_bytes = sum(p.numel() for p in self.mlp.parameters()) * dtype_size
            per_residual_bytes = dim * dtype_size + index_size
            n_residuals = int((target_bytes - param_bytes) / per_residual_bytes)
            n_residuals = max(0, min(n_residuals, bsz * heads * seq_len))
            if n_residuals == 0:
                mask = torch.zeros_like(errors, dtype=torch.bool)
            else:
                errors_b = errors.view(errors.shape[0], -1)
                thresh = torch.topk(
                    errors_b,
                    min(n_residuals, errors_b.shape[1]),
                    largest=True,
                ).values[:, -1]
                mask = errors >= thresh[:, None, None]
        else:
            mask = torch.zeros_like(errors, dtype=torch.bool)

        b, h, t = mask.nonzero(as_tuple=True)
        _, heads, seq_len, _ = self.tensor.shape
        max_index = max(0, self.tensor.shape[0] * heads * seq_len - 1)
        idx_dtype = (
            torch.int32 if max_index < torch.iinfo(torch.int32).max else torch.int64
        )
        self.indices = (b * (heads * seq_len) + h * seq_len + t).to(idx_dtype)
        self.residuals = (self.tensor[b, h, t] - pred[b, h, t]).detach()
        self.tensor = self.tensor.new_empty(
            (*self.tensor.shape[:2], 0, self.tensor.shape[3])
        )
        self.is_compressed = True

    def _decompress(self, source_states: torch.Tensor) -> torch.Tensor:
        source_prefix = source_states[:, :, : self.compressed_len, :]
        pred = self.mlp(source_prefix)
        _, heads, seq_len, _ = pred.shape
        t = self.indices % seq_len
        b = (self.indices // seq_len) // heads
        h = (self.indices // seq_len) % heads
        if b.numel() > 0:
            pred[b, h, t] += self.residuals
        return pred

    def _update_source_tensor(self, source_states: torch.Tensor) -> torch.Tensor:
        if self.source_tensor is None or self.prefill:
            self.source_tensor = source_states
        elif source_states.shape[-2] == 1:
            self.source_tensor = torch.cat([self.source_tensor, source_states], dim=-2)
        elif source_states.shape[-2] > self.source_tensor.shape[-2]:
            self.source_tensor = source_states
        return self.source_tensor

    def update(
        self,
        target_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        source_states = self._source_tensor(cache_kwargs, target_states)
        if source_states is None:
            if not self.is_initialized:
                SingleTensorDynamicLayer.lazy_initialization(self, target_states)
            self.tensor = torch.cat([self.tensor, target_states], dim=-2)
            self.seq_len += target_states.shape[-2]
            return self.tensor

        source_states = self._update_source_tensor(source_states)
        target_for_training = (
            self._undo_rope(
                target_states, 
                cache_kwargs,
                self.prefill,
                self.compressed_len
                )
            if self.target_is_key
            else target_states
        )
        if not self.is_initialized:
            self.lazy_initialization(target_for_training, source_states)

        returned_target = super().update(target_for_training)
        if self.prefill:
            self._train_mlp(source_states, target_for_training)
            self._compress(source_states)
            self.prefill = False
            return target_states
        if self.is_compressed:
            pred = self._decompress(source_states)
            pred = torch.cat([pred, self.tensor], dim=-2)
            if self.target_is_key:
                pred = self._apply_rope(pred, compressed_len=self.compressed_len)
            return pred
        return returned_target


class MLPPredictionCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        source: str,
        num_layers_per_mlp: list[int],
        hidden_factors_per_mlp: list[int],
        num_heads_per_mlp: list[int],
        target_cr: float = 3.0,
        num_epochs: int = 200,
        target_is_key: bool = False,
        rope_theta: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, rope_theta=rope_theta, **kwargs)
        self.source = source
        self.num_layers_per_mlp = num_layers_per_mlp
        self.hidden_factors_per_mlp = hidden_factors_per_mlp
        self.num_heads_per_mlp = num_heads_per_mlp
        self.target_cr = target_cr
        self.num_epochs = num_epochs
        self.target_is_key = target_is_key

    def _build_layer(self, layer_idx: int) -> MLPPredictionLayer:
        return MLPPredictionLayer(
            source=self.source,
            mlp_num_layers=self.num_layers_per_mlp[layer_idx],
            mlp_hidden_factor=self.hidden_factors_per_mlp[layer_idx],
            mlp_num_heads=self.num_heads_per_mlp[layer_idx],
            target_cr=self.target_cr,
            num_epochs=self.num_epochs,
            target_is_key=self.target_is_key,
            rope_theta=self.rope_theta,
        )

    def update(
        self,
        tensor_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            self.layers.append(self._build_layer(len(self.layers)))
        return self.layers[layer_idx].update(tensor_states, cache_kwargs)

    @property
    def recon_mse(self) -> float | None:
        mses = [layer.recon_mse for layer in self.layers if layer.recon_mse is not None]
        if not mses:
            return None
        return sum(mses) / len(mses)

    def calc_compression_ratio(self) -> float:
        original_total = 0
        compressed_total = 0
        for layer in self.layers:
            if not layer.is_initialized or layer.tensor is None:
                continue
            heads = layer.mlp_num_heads
            dim = layer.tensor.shape[-1]
            seq_len = layer.compressed_len
            dtype_size = torch.finfo(layer.dtype).bits / 8
            original = heads * seq_len * dim * dtype_size
            if layer.mlp is None or not layer.is_compressed or layer.indices is None:
                compressed = original
            else:
                index_dtype_size = torch.iinfo(layer.indices.dtype).bits / 8
                num_params = sum(p.numel() for p in layer.mlp.parameters())
                compressed = (
                    num_params * dtype_size
                    + layer.indices.numel() * dim * dtype_size
                    + layer.indices.numel() * index_dtype_size
                )
            original_total += original
            compressed_total += compressed
        if compressed_total == 0:
            return 1.0
        return original_total / compressed_total


class AnalysisCompressedCache(CompressedCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_layer_key_states = {}
        self._previous_layer_value_states = {}

    def _attach_prediction_source_kwargs(
        self,
        cache_kwargs: dict[str, Any],
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        cache_kwargs["current_keys"] = key_states
        cache_kwargs["current_values"] = value_states
        if layer_idx > 0:
            cache_kwargs["prev_layer_keys"] = self._previous_layer_key_states.get(
                layer_idx - 1
            )
            cache_kwargs["prev_layer_values"] = self._previous_layer_value_states.get(
                layer_idx - 1
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = {} if cache_kwargs is None else dict(cache_kwargs)
        self._attach_prediction_source_kwargs(
            cache_kwargs, key_states, value_states, layer_idx
        )
        keys = self.key_cache.update(key_states, layer_idx, cache_kwargs)
        if self._cache_context:
            cache_kwargs = {**self._cache_context, **cache_kwargs}
        values = self.value_cache.update(value_states, layer_idx, cache_kwargs)
        self._previous_layer_key_states[layer_idx] = keys.detach()
        self._previous_layer_value_states[layer_idx] = values.detach()
        return keys, values

    @property
    def key_recon_mse(self) -> float | None:
        if hasattr(self.key_cache, "recon_mse"):
            return self.key_cache.recon_mse
        return None

    @property
    def key_prediction_mse(self) -> float | None:
        if hasattr(self.key_cache, "recon_mse"):
            return self.key_cache.recon_mse
        return None
