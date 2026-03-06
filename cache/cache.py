import warnings
import torch

from transformers.cache_utils import (
    Any,
    Iterable,
    PreTrainedConfig,
)


def get_cache(cache_type, CACHE_CLASSES, verbose=True):
    if cache_type not in CACHE_CLASSES:
        raise ValueError(f"Invalid cache type: {cache_type}")
    if verbose:
        print(f"Loading cache type {cache_type}")
    return CACHE_CLASSES[cache_type]


class SingleTensorDynamicLayer:
    """
    Dynamic cache layer for a single tensor of shape [B, H, T, D].
    Modeled after the DynamicLayer in transformers' DynamicCache,
    but simplified for the single-tensor use case.
    """

    # TODO: add all method from CacheLayerMixin as well (DynamicLayerr's Base)
    def __init__(self):
        self.tensor: torch.Tensor | None = None
        self.is_initialized = False
        self.device = None
        self.dtype = None

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
        # custom API: Created to avoid overwriting the reset method for full compliance with HF's DynamicLayer API
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
        **kwargs,
    ):
        self.layers: list[SingleTensorDynamicLayer] = []

        if ddp_cache_data is not None:
            for tensor_states in ddp_cache_data:
                layer = SingleTensorDynamicLayer()
                layer.update(tensor_states)
                self.layers.append(layer)

    def update(
        self,
        tensor_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            self.layers.append(SingleTensorDynamicLayer())
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


class CompressedCache:
    """
    Dynamic cache that applies low-rank decomposition to keys
        and learns values in MLPs.
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[torch.Tensor] | None = None,
        config: PreTrainedConfig | None = None,
        key_cache_kwargs: dict | None = None,
        value_cache_kwargs: dict | None = None,
        **kwargs,
    ):
        # super().__init__(ddp_cache_data=ddp_cache_data, config=config)

        if key_cache_kwargs is None:
            key_cache_kwargs = {"cache_type": "baseline"}
        else:
            key_cache_kwargs = dict(key_cache_kwargs)

        if value_cache_kwargs is None:
            value_cache_kwargs = {"cache_type": "baseline"}
        else:
            value_cache_kwargs = dict(value_cache_kwargs)

        # local import to avoid circular import at module load
        from .key_cache import KEY_CACHE_CLASSES

        key_cache_type = key_cache_kwargs.pop("cache_type")
        self.key_cache = get_cache(
            key_cache_type, KEY_CACHE_CLASSES, kwargs["verbose"]
        )(
            ddp_cache_data=ddp_cache_data,
            **key_cache_kwargs,
        )

        from .value_cache import VALUE_CACHE_CLASSES

        value_cache_type = value_cache_kwargs.pop("cache_type")
        self.value_cache = get_cache(
            value_cache_type, VALUE_CACHE_CLASSES, kwargs["verbose"]
        )(
            ddp_cache_data=ddp_cache_data,
            **value_cache_kwargs,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_cache.update(key_states, layer_idx, cache_kwargs)
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_kwargs["keys"] = keys
        values = self.value_cache.update(value_states, layer_idx, cache_kwargs)
        return keys, values

    @property
    def comp_ratio(self) -> float | None:
        """
        Calculate compression ratios from both caches.
        """
        if hasattr(self.key_cache, "comp_ratio"):
            key_cr = self.key_cache.comp_ratio
        else:
            key_cr = None
        if hasattr(self.value_cache, "comp_ratio"):
            value_cr = self.value_cache.calc_compression_ratio()
        else:
            value_cr = None

        if key_cr is not None and value_cr is not None:
            return 2 / ((1 / key_cr) + (1 / value_cr))
        elif key_cr is not None:
            return key_cr
        elif value_cr is not None:
            return value_cr
        else:
            return None

    def update_events(self, *args, **kwargs) -> None:
        """
        Forward event updates to caches.
        """
        if hasattr(self.key_cache, "update_events"):
            self.key_cache.update_events(*args, **kwargs)
        if hasattr(self.value_cache, "update_events"):
            self.value_cache.update_events(*args, **kwargs)

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop not implemented")
        for k_layer, v_layer in zip(
            self.key_cache.layers, self.value_cache.layers
        ):
            k_layer.crop(max_length)
            v_layer.crop(max_length)

    def __getattr__(self, name):
        return getattr(self.key_cache, name)

    def __iter__(self):
        for k_layer, v_layer in zip(
            self.key_cache.layers, self.value_cache.layers
        ):
            yield k_layer.tensor, v_layer.tensor
