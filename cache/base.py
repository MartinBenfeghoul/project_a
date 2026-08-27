import warnings

import torch

from typing import Any, Iterable

from utils.rope import apply_packed_rope, inverse_packed_rope


class SharedRopeCache:
    """One packed RoPE table shared by every key/value cache layer."""

    def __init__(self):
        self.packed: torch.Tensor | None = None

    def capture(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        expected_shape = (cos.size(0), 1, cos.size(2), cos.size(3))
        if self.packed is not None and self.packed.shape == expected_shape:
            return

        half_dim = cos.size(-1) // 2
        self.packed = torch.cat(
            (cos[..., :half_dim], sin[..., :half_dim]),
            dim=-1,
        ).detach().contiguous()

    def _table(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.packed is None:
            raise ValueError(
                "RoPE state is unavailable. The model must provide 'cos' and "
                "'sin' in cache_kwargs during prefill."
            )
        return self.packed.to(device=device, dtype=dtype)

    def prefix(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        if positions is None:
            if length > table.size(2):
                raise ValueError(
                    f"Requested RoPE length {length}, but only "
                    f"{table.size(2)} positions were captured."
                )
            return table[:, :, :length]
        positions = positions[:length].to(device=table.device, dtype=torch.long)
        return table.index_select(2, positions)

    def selected(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        batch_size, num_heads = positions.shape[:2]
        if table.size(0) == 1 and batch_size > 1:
            table = table.expand(batch_size, -1, -1, -1)
        if table.size(0) != batch_size:
            raise ValueError(
                f"RoPE batch size {table.size(0)} does not match "
                f"requested batch size {batch_size}."
            )
        table = table.expand(-1, num_heads, -1, -1)
        gather_idx = positions[..., None].expand(
            -1, -1, -1, table.size(-1)
        )
        return table.gather(2, gather_idx)

    def fused(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        if length > table.size(2):
            raise ValueError(
                f"Requested RoPE length {length}, but only "
                f"{table.size(2)} positions were captured."
            )
        return table[0, 0, :length].contiguous()

    @property
    def nbytes(self) -> int:
        if self.packed is None:
            return 0
        return self.packed.numel() * self.packed.element_size()

    @property
    def supports_fused(self) -> bool:
        return self.packed is not None and self.packed.size(0) == 1


class RopeLayerMixin:
    """Use shared packed RoPE state to un-rope/re-rope cached tensors."""

    def _init_rope(self, rope_cache: SharedRopeCache | None = None):
        self.rope_cache = rope_cache or SharedRopeCache()
        self.rope_positions: torch.Tensor | None = None

    def _resolve_packed_rope(
        self,
        prefix_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_kwargs: dict | None = None,
    ) -> torch.Tensor:
        if (
            cache_kwargs is not None
            and "cos" in cache_kwargs
            and "sin" in cache_kwargs
        ):
            self.rope_cache.capture(cache_kwargs["cos"], cache_kwargs["sin"])
            kept_positions = cache_kwargs.get("kept_positions")
            if kept_positions is not None:
                self.rope_positions = kept_positions.detach()

        return self.rope_cache.prefix(
            prefix_len,
            device,
            dtype,
            positions=self.rope_positions,
        )

    def _undo_rope(
        self,
        keys: torch.Tensor,
        cache_kwargs: dict | None = None,
        prefill: bool = True,
        compressed_len: int = 0,
    ) -> torch.Tensor:
        T = keys.shape[2]
        prefix_len = T if prefill else min(compressed_len, T)
        if prefix_len == 0:
            return keys

        packed = self._resolve_packed_rope(
            prefix_len,
            keys.device,
            keys.dtype,
            cache_kwargs=cache_kwargs if prefill else None,
        )

        prefix = inverse_packed_rope(keys[:, :, :prefix_len], packed)
        if prefix_len < T:
            return torch.cat([prefix, keys[:, :, prefix_len:]], dim=2)
        return prefix

    def _rope_selected(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        compressed_len: int,
        inverse: bool,
    ) -> torch.Tensor:
        positions = positions.to(device=keys.device, dtype=torch.long)
        if compressed_len <= 0:
            return keys
        valid = positions < compressed_len
        rope_positions = positions.clamp(min=0, max=compressed_len - 1)
        if self.rope_positions is not None:
            mapping = self.rope_positions.to(
                device=keys.device,
                dtype=torch.long,
            )
            rope_positions = mapping[rope_positions]
        packed = self.rope_cache.selected(
            rope_positions,
            keys.device,
            keys.dtype,
        )
        transformed = (
            inverse_packed_rope(keys, packed)
            if inverse
            else apply_packed_rope(keys, packed)
        )
        return torch.where(valid[..., None], transformed, keys)

    def _apply_rope(
        self,
        keys: torch.Tensor,
        compressed_len: int = 0,
    ) -> torch.Tensor:
        T = keys.shape[2]
        prefix_len = min(compressed_len, T)
        if prefix_len == 0:
            return keys

        packed = self._resolve_packed_rope(
            prefix_len,
            keys.device,
            keys.dtype,
        )

        prefix = keys[:, :, :prefix_len]
        prefix_roped = apply_packed_rope(prefix, packed)
        if prefix_len < T:
            return torch.cat([prefix_roped, keys[:, :, prefix_len:]], dim=2)
        return prefix_roped

    def _fused_rope(self, length: int, tensor: torch.Tensor) -> torch.Tensor:
        return self.rope_cache.fused(length, tensor.device, tensor.dtype)

    @property
    def supports_fused_rope(self) -> bool:
        return self.rope_positions is None and self.rope_cache.supports_fused


class SingleTensorDynamicLayer(RopeLayerMixin):
    """
    Dynamic cache layer for a single tensor of shape [B, H, T, D].
    Modeled after the DynamicLayer in transformers' DynamicCache,
    but simplified for the single-tensor use case.
    """

    # TODO: add all method from CacheLayerMixin as well (DynamicLayerr's Base)
    def __init__(self, rope_cache: SharedRopeCache | None = None):
        self.tensor: torch.Tensor | None = None
        self.is_initialized = False
        self.device = None
        self.dtype = None
        self._init_rope(rope_cache)

    def lazy_initialization(self, tensor_states: torch.Tensor) -> None:
        self.dtype, self.device = tensor_states.dtype, tensor_states.device
        self.tensor = torch.tensor([], dtype=self.dtype, device=self.device)
        self.seq_len = 0  # custom attribute
        self.is_initialized = True

    def update(self, tensor_states: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(tensor_states)
        self.tensor = torch.cat([self.tensor, tensor_states], dim=-2)
        self.seq_len += tensor_states.shape[-2]
        return self.tensor

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self.seq_len  # custom changes

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """
        Return the length and offset of the cache, used to generate the mask
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def reset(self) -> None:
        if self.is_initialized:
            self.tensor.zero_()  # TODO: why do they keep the tensor...?

    def _evict(
        self,
        end_idx: int | None = None,
        reset_seq_len: bool = False,
    ) -> None:
        # custom API: Created to avoid overwriting the reset method
        # for full compliance with HF's DynamicLayer API
        if self.is_initialized:
            if end_idx is None:
                self.tensor = torch.tensor(
                    [], dtype=self.dtype, device=self.device
                )
                if reset_seq_len:
                    self.seq_len = 0
            else:
                self.tensor = self.tensor[..., end_idx:, :].clone()
                if reset_seq_len:
                    self.seq_len = self.tensor.shape[-2]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() > 0:
            self.tensor = self.tensor.index_select(
                0, beam_idx.to(self.tensor.device)
            )

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop not implemented")
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.tensor = self.tensor[..., :max_length, :]
        self.seq_len = max_length

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            self.tensor = self.tensor.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            self.tensor = self.tensor[indices, ...]


class SingleTensorCache:
    """
    Dynamic cache for one tensor stream (keys OR values).
    Modeled after the DynamicCache in transformers,
    but simplified for the single-tensor use case.
    """

    # TODO: add all method from Cache as well (DynamicCache's Base)
    def __init__(
        self,
        ddp_cache_data: Iterable[torch.Tensor] | None = None,
        rope_cache: SharedRopeCache | None = None,
        **kwargs,
    ):
        self.layers: list[SingleTensorDynamicLayer] = []
        self.rope_cache = rope_cache or SharedRopeCache()

        if ddp_cache_data is not None:
            for tensor_states in ddp_cache_data:
                layer = SingleTensorDynamicLayer(rope_cache=self.rope_cache)
                layer.update(tensor_states)
                self.layers.append(layer)

    def update(
        self,
        tensor_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                SingleTensorDynamicLayer(rope_cache=self.rope_cache)
            )
        return self.layers[layer_idx].update(tensor_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def _evict(self, layer_idx: int | None = None, end_idx: int | None = None):
        # custom API
        if layer_idx is not None:
            if layer_idx < len(self.layers):
                self.layers[layer_idx]._evict(end_idx=end_idx)
        else:
            warnings.warn("Clearing all layers in cache.")
            for layer in self.layers:
                layer._evict(end_idx=end_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int):
        raise NotImplementedError("crop not implemented")
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.tensor
