import warnings
import torch

from typing import Any, Iterable

from utils.rope import inverse_rope, apply_rope, compute_rope_cos_sin


class RopeLayerMixin:
    """Per-layer RoPE state for caches that un-rope/re-rope keys or values."""

    def _init_rope(self, rope_theta=None):
        self.rope_theta = rope_theta
        self.rope_cos: torch.Tensor | None = None
        self.rope_sin: torch.Tensor | None = None

    def _resolve_rope_cos_sin(
        self,
        prefix_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_kwargs: dict | None = None,
        save: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve cos/sin from cache_kwargs, saved state, or recomputation."""
        if (
            cache_kwargs is not None
            and "cos" in cache_kwargs
            and "sin" in cache_kwargs
        ):
            cos, sin = cache_kwargs["cos"], cache_kwargs["sin"]
            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            cos = cos[:, :, :prefix_len].to(device=device, dtype=dtype)
            sin = sin[:, :, :prefix_len].to(device=device, dtype=dtype)
        elif self.rope_cos is not None and self.rope_sin is not None:
            cos = self.rope_cos[:, :, :prefix_len].to(
                device=device, dtype=dtype
            )
            sin = self.rope_sin[:, :, :prefix_len].to(
                device=device, dtype=dtype
            )
        else:
            if self.rope_theta is None:
                raise ValueError(
                    "_resolve_rope_cos_sin: rope_theta is None. "
                    "Pass rope_theta at construction time, or provide "
                    "'cos'/'sin' in cache_kwargs."
                )
            warnings.warn(
                "_resolve_rope_cos_sin: cos/sin not found in cache_kwargs"
                " or saved state, recomputing from rope_theta"
            )
            cos, sin = compute_rope_cos_sin(
                prefix_len,
                head_dim,
                self.rope_theta,
                device,
                dtype,
            )

        if save or self.rope_cos is None or self.rope_sin is None:
            self.rope_cos = cos
            self.rope_sin = sin

        return cos, sin

    def _cache_rope_state(
        self,
        keys: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> None:
        """Pre-save rope cos/sin from cache_kwargs for later use."""
        self._resolve_rope_cos_sin(
            keys.shape[2],
            keys.shape[-1],
            keys.device,
            keys.dtype,
            cache_kwargs=cache_kwargs,
            save=True,
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

        # Only use cache_kwargs cos/sin during prefill
        # (decode cos/sin only covers new tokens)
        cos, sin = self._resolve_rope_cos_sin(
            prefix_len,
            keys.shape[-1],
            keys.device,
            keys.dtype,
            cache_kwargs=cache_kwargs if prefill else None,
            save=prefill,
        )

        prefix = inverse_rope(keys[:, :, :prefix_len], cos, sin)
        if prefix_len < T:
            return torch.cat([prefix, keys[:, :, prefix_len:]], dim=2)
        return prefix

    def _apply_rope(
        self,
        keys: torch.Tensor,
        compressed_len: int = 0,
    ) -> torch.Tensor:
        T = keys.shape[2]
        prefix_len = min(compressed_len, T)
        if prefix_len == 0:
            return keys

        cos, sin = self._resolve_rope_cos_sin(
            prefix_len,
            keys.shape[-1],
            keys.device,
            keys.dtype,
        )

        prefix = keys[:, :, :prefix_len]
        prefix_roped = apply_rope(prefix, cos, sin)
        if prefix_len < T:
            return torch.cat([prefix_roped, keys[:, :, prefix_len:]], dim=2)
        return prefix_roped


class SingleTensorDynamicLayer(RopeLayerMixin):
    """
    Dynamic cache layer for a single tensor of shape [B, H, T, D].
    Modeled after the DynamicLayer in transformers' DynamicCache,
    but simplified for the single-tensor use case.
    """

    # TODO: add all method from CacheLayerMixin as well (DynamicLayerr's Base)
    def __init__(self, rope_theta=None):
        self.tensor: torch.Tensor | None = None
        self.is_initialized = False
        self.device = None
        self.dtype = None
        self._init_rope(rope_theta)

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
        if not self.is_initialized or self.tensor.numel() == 0:
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
                self.tensor = self.tensor[..., end_idx:, :]
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
        rope_theta: float | None = None,
        **kwargs,
    ):
        self.layers: list[SingleTensorDynamicLayer] = []
        self.rope_theta = rope_theta

        if ddp_cache_data is not None:
            for tensor_states in ddp_cache_data:
                layer = SingleTensorDynamicLayer(rope_theta=rope_theta)
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
                SingleTensorDynamicLayer(rope_theta=self.rope_theta)
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
