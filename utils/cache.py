import torch

from transformers.cache_utils import (
    Cache,
    DynamicCache as DC,
    Any,
)

from .key_cache import KEY_CACHE_CLASSES


class CompressedCache(Cache):

    def __init__(
        self,
        key_cache_kwargs: dict = {},
        value_cache_kwargs: dict = {},
    ):
        super().__init__()

        key_cache_type = key_cache_kwargs.pop("cache_type")
        self.key_cache = KEY_CACHE_CLASSES[key_cache_type](**key_cache_kwargs)

        self.value_cache = DC(**value_cache_kwargs)  # TODO: replace this with actual value cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        keys = self.key_cache.update(key_states, layer_idx, cache_kwargs)
        _, values = self.value_cache.update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        return keys, values
