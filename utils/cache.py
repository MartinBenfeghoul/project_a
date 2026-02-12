import torch

from transformers.cache_utils import (
    Cache,
    DynamicCache as DC,
    Any,
    Iterable,
    PreTrainedConfig,
)

# NOTE: avoid top-level import to prevent circular import with key_cache.py
# from .key_cache import KEY_CACHE_CLASSES


class DynamicCache(DC):
    """This class simply intercepts kwargs for a more flexible base class."""

    def __init__(
        self,
        *args,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        **kwargs,
    ):
        super().__init__(
            ddp_cache_data,
            config,
            offloading,
            offload_only_non_sliding,
        )


class SingleTensorDynamicLayer:
    """
    Dynamic cache layer for a single tensor of shape [B, H, T, D].
    Modeled after the DynamicLayer in transformers' DynamicCache,
    but simplified for the single-tensor use case.
    """

    def __init__(self):
        self.tensor: torch.Tensor | None = None
        self.is_initialized = False
        self.device = None
        self.dtype = None

    def lazy_initialization(self, tensor_states: torch.Tensor) -> None:
        self.dtype, self.device = tensor_states.dtype, tensor_states.device
        self.tensor = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, tensor_states: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(tensor_states)
        self.tensor = torch.cat([self.tensor, tensor_states], dim=-2)
        return self.tensor

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.tensor.numel() == 0:
            return 0
        return self.tensor.shape[-2]

    def reset(self) -> None:
        if self.is_initialized:
            self.tensor.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() > 0:
            self.tensor = self.tensor.index_select(
                0, beam_idx.to(self.tensor.device)
            )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.tensor = self.tensor[..., :max_length, :]

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

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int):
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


class CompressedCache(Cache):
    """
    Dynamic cache that applies low-rank decomposition to keys 
        and learns values in MLPs.
    """
    # TODO: make it fully HF-compatible by implementing the same API as DynamicCache and adding a config class

    def __init__(
        self,
        ddp_cache_data: Iterable[torch.Tensor] | None = None,
        key_cache_kwargs: dict | None = None,
        value_cache_kwargs: dict | None = None,
    ):
        super().__init__()

        key_cache_kwargs = (
            {} if key_cache_kwargs is None else dict(key_cache_kwargs)
        )
        value_cache_kwargs = (
            {} if value_cache_kwargs is None else dict(value_cache_kwargs)
        )

        # local import to avoid circular import at module load
        from .key_cache import KEY_CACHE_CLASSES

        key_cache_type = key_cache_kwargs.pop("cache_type")
        self.key_cache = KEY_CACHE_CLASSES[key_cache_type](**key_cache_kwargs)

        # TODO: replace value cache with the cache in the value branch
        self.value_cache = SingleTensorCache(
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
        
        values = self.value_cache.update(value_states, layer_idx, cache_kwargs)
        return keys, values
