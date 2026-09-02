"""The compressed cache object."""

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from . import accounting
from .config import (
    CompressedCacheConfig,
    build_key_cache,
    build_value_cache,
)
from .eviction import EvictionPolicy
from .rope import SharedRopeCache
from .selective import SelectiveReconstruction


@dataclass
class DeferredValueUpdate:
    """A value update held back until its xKV layer group is decomposed."""

    value_states: torch.Tensor
    cache_kwargs: dict[str, Any]


class CompressedCache:
    """
    Dynamic cache that applies low-rank decomposition to keys
        and learns values in MLPs.
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[torch.Tensor] | None = None,
        config: CompressedCacheConfig | None = None,
        cache_context: dict[str, Any] | None = None,
        verbose: bool = True,
    ):
        self.config = config or CompressedCacheConfig()
        self.rope_cache = SharedRopeCache()

        self.key_cache = build_key_cache(
            self.config.key,
            ddp_cache_data=ddp_cache_data,
            rope_cache=self.rope_cache,
            verbose=verbose,
        )
        self.value_cache, self.pass_keys_to_value_cache = build_value_cache(
            self.config.value,
            ddp_cache_data=ddp_cache_data,
            rope_cache=self.rope_cache,
            verbose=verbose,
        )

        self.selective = SelectiveReconstruction(
            self.config.selective,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
        )
        self.eviction = EvictionPolicy(
            self.config.eviction_keep_ratio,
            key_cache=self.key_cache,
        )

        self._cache_context = dict(cache_context or {})
        self._deferred_value_updates: dict[int, DeferredValueUpdate] = {}
        self.prefill = True

    # --- selective reconstruction -----------------------------------------

    @property
    def selective_reconstruction(self) -> bool:
        return self.selective.enabled

    @property
    def selective_layers(self):
        return self.selective.layers

    def select_positions(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.selective.select_positions(layer_idx, query_states)

    def supports_selective_retrieval(self, layer_idx: int) -> bool:
        return self.selective.supports_retrieval(layer_idx)

    def retrieve_selected(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.selective.retrieve(layer_idx, positions)

    def append_selective(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any],
    ) -> None:
        # Selective decode bypasses update(), so end the prefill phase here.
        self._end_prefill_on_first_decode(key_states)
        append_keys = getattr(self.key_cache, "append_decode", None)
        if callable(append_keys):
            append_keys(key_states, layer_idx, cache_kwargs)
        else:
            self.key_cache.update(key_states, layer_idx, cache_kwargs)
        self.value_cache.append_decode(value_states, layer_idx)

    # --- eviction ---------------------------------------------------------

    @property
    def eviction_keep_ratio(self) -> float:
        return self.eviction.keep_ratio

    @property
    def kept_positions(self):
        return self.eviction.kept_positions

    def set_value_importance(
        self,
        layer_idx: int,
        value_importance: torch.Tensor,
    ) -> None:
        self.eviction.set_value_importance(layer_idx, value_importance)

    # --- the update path --------------------------------------------------

    def _end_prefill_on_first_decode(self, key_states: torch.Tensor) -> None:
        if key_states.size(-2) == 1 and self.prefill:
            self.update_events()

    def _true_seq_len(self, full_key_states: torch.Tensor) -> float:
        """Original tokens per sequence."""
        mask = self._cache_context.get("padding_mask")
        if mask is None:
            return float(full_key_states.size(-2))
        return mask.to(torch.bool).sum().item() / mask.size(0)

    def _aligned_padding_mask(
        self,
        key_states: torch.Tensor,
        keep_positions: torch.Tensor | None,
    ) -> torch.Tensor | None:
        mask = self._cache_context.get("padding_mask")
        if mask is None or key_states.size(-2) == 1:
            return None
        mask = mask.to(device=key_states.device, dtype=torch.bool)
        if keep_positions is not None:
            mask = mask.index_select(1, keep_positions)
        if mask.shape != (key_states.size(0), key_states.size(-2)):
            raise ValueError(
                "padding_mask must have shape [batch, sequence_length]."
            )
        return mask

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._end_prefill_on_first_decode(key_states)

        full_key_states, full_value_states = key_states, value_states
        key_states, value_states, keep_positions = self.eviction.apply(
            key_states,
            value_states,
            layer_idx,
            self._aligned_padding_mask(key_states, None),
        )
        padding_mask = self._aligned_padding_mask(key_states, keep_positions)
        if self.selective.enabled and key_states.size(-2) > 1:
            self.selective.store_landmarks(
                layer_idx,
                key_states,
                value_states,
                true_seq_len=self._true_seq_len(full_key_states),
                padding_mask=padding_mask,
            )
        cache_kwargs = self.eviction.annotate_cache_kwargs(
            cache_kwargs, keep_positions
        )
        if padding_mask is not None:
            cache_kwargs = {
                **(cache_kwargs or {}),
                "padding_mask": padding_mask,
            }

        keys = self.key_cache.update(key_states, layer_idx, cache_kwargs)
        cache_kwargs = self._value_cache_kwargs(cache_kwargs)
        self.eviction.pop_value_importance(layer_idx)
        values = self._update_value_cache(
            value_states, layer_idx, cache_kwargs, keys
        )

        if keep_positions is not None:
            return full_key_states, full_value_states
        return keys, values

    def _value_cache_kwargs(
        self,
        cache_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if cache_kwargs is None:
            cache_kwargs = {}
        if self._cache_context:
            cache_kwargs = {**self._cache_context, **cache_kwargs}
        return cache_kwargs

    def _update_value_cache(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any],
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Update the value backend, deferring across an xKV layer group.

        A grouped key cache only reconstructs its keys once the last layer
        of the group arrives. Value layers that need those keys are parked
        until then and flushed together.
        """
        bounds = self._value_update_group_bounds(layer_idx)
        if bounds is not None and layer_idx != bounds[1]:
            self._deferred_value_updates[layer_idx] = DeferredValueUpdate(
                value_states=value_states,
                cache_kwargs=dict(cache_kwargs),
            )
            return value_states
        if bounds is not None and self._has_deferred_value_updates(bounds):
            return self._flush_deferred_value_updates(
                layer_idx,
                value_states,
                cache_kwargs,
                bounds,
            )
        if self.pass_keys_to_value_cache:
            self._attach_keys_to_value_cache_kwargs(
                cache_kwargs, layer_idx, keys
            )
        return self.value_cache.update(value_states, layer_idx, cache_kwargs)

    def _value_update_group_bounds(
        self,
        layer_idx: int,
    ) -> tuple[int, int] | None:
        if not self.pass_keys_to_value_cache:
            return None
        if not callable(
            getattr(self.value_cache, "update_prefill_group", None)
        ):
            return None
        if not self.prefill:
            return None

        get_group_bounds = getattr(self.key_cache, "_get_group_bounds", None)
        if not callable(get_group_bounds):
            return None
        return get_group_bounds(layer_idx)

    def _has_deferred_value_updates(self, bounds: tuple[int, int]) -> bool:
        group_start, group_last_layer = bounds
        return any(
            idx in self._deferred_value_updates
            for idx in range(group_start, group_last_layer)
        )

    def _attach_keys_to_value_cache_kwargs(
        self,
        cache_kwargs: dict[str, Any],
        layer_idx: int,
        keys: torch.Tensor,
    ) -> None:
        fn = getattr(self.key_cache, "get_reconstructed_keys_only", None)
        if callable(fn) and self.prefill:
            cache_kwargs["keys"] = fn(layer_idx)
        else:
            cache_kwargs["keys"] = keys

    def _flush_deferred_value_updates(
        self,
        layer_idx: int,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any],
        bounds: tuple[int, int],
    ) -> torch.Tensor:
        group_start, group_last_layer = bounds
        updates = []
        for group_layer_idx in range(group_start, group_last_layer + 1):
            if group_layer_idx == layer_idx:
                continue
            deferred = self._deferred_value_updates.pop(group_layer_idx)
            deferred.cache_kwargs["keys"] = (
                self.key_cache.get_reconstructed_keys_only(group_layer_idx)
            )
            updates.append(
                (
                    group_layer_idx,
                    deferred.value_states,
                    deferred.cache_kwargs,
                )
            )

        current_recon_keys = self.key_cache.get_reconstructed_keys_only(
            layer_idx
        )
        current_kwargs = dict(cache_kwargs)
        current_kwargs["keys"] = current_recon_keys
        updates.append((layer_idx, value_states, current_kwargs))

        results = self.value_cache.update_prefill_group(updates)
        return results[layer_idx]

    # --- accounting and transformers plumbing -----------------------------

    @property
    def comp_ratio(self) -> float | None:
        """
        Calculate compression ratios from both caches.
        """
        return accounting.compression_ratio(
            self.key_cache,
            self.value_cache,
            self.selective,
            self.rope_cache,
        )

    def update_events(self, *args, **kwargs):
        """
        Forward event updates to caches.
        """
        self.prefill = False
        n_events = None
        if hasattr(self.key_cache, "update_events"):
            n_events = self.key_cache.update_events(*args, **kwargs)
        if hasattr(self.value_cache, "update_events"):
            self.value_cache.update_events(*args, **kwargs)
        return n_events

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop not implemented")

    def __getattr__(self, name):
        # Allow direct access to key_cache attributes where not defined in CompressedCache
        # TODO: check this makes sense for all attributes we want to expose
        return getattr(self.key_cache, name)

    def __iter__(self):
        for k_layer, v_layer in zip(
            self.key_cache.layers, self.value_cache.layers
        ):
            yield k_layer.tensor, v_layer.tensor
