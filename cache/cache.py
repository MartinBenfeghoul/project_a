import copy
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
        from .keys import KEY_CACHE_CLASSES

        key_cache_type = key_cache_kwargs.pop("cache_type")
        self.key_cache = get_cache(
            key_cache_type, KEY_CACHE_CLASSES, kwargs.get("verbose", True)
        )(
            ddp_cache_data=ddp_cache_data,
            **key_cache_kwargs,
        )

        from .values import VALUE_CACHE_CLASSES

        value_cache_type = value_cache_kwargs.pop("cache_type")
        if value_cache_type in VALUE_CACHE_CLASSES:
            self.value_cache = get_cache(
                value_cache_type,
                VALUE_CACHE_CLASSES,
                kwargs.get("verbose", True),
            )(
                ddp_cache_data=ddp_cache_data,
                **value_cache_kwargs,
            )
        elif value_cache_type in KEY_CACHE_CLASSES:
            value_cache_kwargs = copy.deepcopy(key_cache_kwargs)
            value_cache_kwargs["unrope_keys"] = False
            self.value_cache = get_cache(
                value_cache_type,
                KEY_CACHE_CLASSES,
                kwargs.get("verbose", True),
            )(
                ddp_cache_data=ddp_cache_data,
                **value_cache_kwargs,
            )
        else:
            raise ValueError(f"Invalid cache type: {value_cache_type}")
        self._temp_value_importance = {}

    def set_value_importance(
        self,
        layer_idx: int,
        value_importance: torch.Tensor,
    ) -> None:
        self._temp_value_importance[layer_idx] = value_importance

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
        if layer_idx in self._temp_value_importance:
            cache_kwargs["value_importance"] = self._temp_value_importance.pop(
                layer_idx
            )
        fn = getattr(self.key_cache, "get_reconstructed_keys_only", None)
        if callable(fn) and getattr(self.key_cache, "prefill", False):
            recon_keys = self.key_cache.get_reconstructed_keys_only(layer_idx)
            cache_kwargs["keys"] = keys if recon_keys is None else recon_keys
        else:
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

    def update_events(self, *args, **kwargs):
        """
        Forward event updates to caches.
        """
        n_events = None
        if hasattr(self.key_cache, "update_events"):
            n_events = self.key_cache.update_events(*args, **kwargs)
        if hasattr(self.value_cache, "update_events"):
            self.value_cache.update_events(*args, **kwargs)
        return n_events

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop not implemented")
        for k_layer, v_layer in zip(
            self.key_cache.layers, self.value_cache.layers
        ):
            k_layer.crop(max_length)
            v_layer.crop(max_length)

    def __getattr__(self, name):
        # Allow direct access to key_cache attributes where not defined in CompressedCache
        # TODO: check this makes sense for all attributes we want to expose
        return getattr(self.key_cache, name)

    def __iter__(self):
        for k_layer, v_layer in zip(
            self.key_cache.layers, self.value_cache.layers
        ):
            yield k_layer.tensor, v_layer.tensor
