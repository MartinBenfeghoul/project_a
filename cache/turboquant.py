from typing import Any, Iterable

import torch

from .base import SingleTensorCache, SingleTensorDynamicLayer
from utils.turboquant import CompressorParams, MSECompressor


class TurboQuantLayer(SingleTensorDynamicLayer):
    def __init__(self, bits: int = 4, rope_theta: float | None = None):
        super().__init__(rope_theta=rope_theta)
        self.bits = bits
        self.compressor: MSECompressor | None = None
        self.compressed_params: CompressorParams | None = None
        self.prefill = True
        self.compressed_len = 0

    def lazy_initialization(self, tensor_states: torch.Tensor) -> None:
        super().lazy_initialization(tensor_states)
        self.compressor = MSECompressor(
            dim=tensor_states.shape[-1],
            bits=self.bits,
            device=tensor_states.device,
        )

    def _apply_compressed_params_batch_op(self, op) -> None:
        if self.compressed_params is None:
            return

        params = self.compressed_params
        indices = op(params.indices)
        norms = op(params.norms)
        self.compressed_params = CompressorParams(
            indices=indices,
            norms=norms,
            shape=tuple(indices.shape[:-1]) + (params.shape[-1],),
            dtype=params.dtype,
            bits=params.bits,
            idx_pad=params.idx_pad,
        )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        def op(tensor):
            return tensor.index_select(0, beam_idx.to(tensor.device))

        if self.is_initialized and self.tensor is not None:
            self.tensor = op(self.tensor)
        self._apply_compressed_params_batch_op(op)

    def batch_repeat_interleave(self, repeats: int) -> None:
        def op(tensor):
            return tensor.repeat_interleave(repeats, dim=0)

        if self.is_initialized and self.tensor is not None:
            self.tensor = op(self.tensor)
        self._apply_compressed_params_batch_op(op)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        def op(tensor):
            return tensor[indices.to(tensor.device)]

        if self.is_initialized and self.tensor is not None:
            self.tensor = op(self.tensor)
        self._apply_compressed_params_batch_op(op)

    def update(
        self,
        tensor_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        tensor = super().update(tensor_states)

        if self.prefill:
            self.compressed_len = tensor.shape[-2]
            self.compressed_params = self.compressor.encode(tensor)
            self._evict(end_idx=self.compressed_len)
            self.prefill = False
            return tensor

        if self.compressed_params is None:
            raise RuntimeError("Prefill is False but no compressed tensor found.")

        prefix = self.compressor.decode(self.compressed_params)
        suffix = self.tensor
        if suffix.shape[-2] > 0:
            return torch.cat([prefix, suffix], dim=-2)
        return prefix


class TurboQuantCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        compressor_bits: int = 4,
        ddp_cache_data: Iterable[torch.Tensor] | None = None,
        **kwargs,
    ):
        super().__init__(*args, ddp_cache_data=None, **kwargs)
        self.compressor_bits = compressor_bits

        if ddp_cache_data is not None:
            for layer_idx, tensor_states in enumerate(ddp_cache_data):
                self.update(tensor_states, layer_idx)

    def update_events(self, *args, **kwargs) -> None:
        pass

    @property
    def comp_ratio(self) -> float:
        return self.calc_compression_ratio()

    def update(
        self,
        tensor_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                TurboQuantLayer(
                    bits=self.compressor_bits,
                    rope_theta=self.rope_theta,
                )
            )
        return self.layers[layer_idx].update(tensor_states, cache_kwargs)

    def calc_compression_ratio(self) -> float:
        original_total = 0.0
        compressed_total = 0.0
        for layer in self.layers:
            params = layer.compressed_params
            if params is None:
                continue

            dtype_size = torch.empty((), dtype=params.dtype).element_size()
            b, h, t_compressed, d = params.shape
            t_suffix = layer.tensor.shape[-2] if layer.tensor.dim() == 4 else 0
            t = t_compressed + t_suffix
            original_total += b * h * t * d * dtype_size
            compressed_total += (
                layer.compressor.memory_nbytes(params)
                + b * h * t_suffix * d * dtype_size
            )

        if compressed_total == 0:
            return 1.0
        return original_total / compressed_total
