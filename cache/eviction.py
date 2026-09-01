"""Physical eviction of prefill KV tokens."""

import math

import torch


class EvictionPolicy:
    """Chooses which prefill positions to keep, and applies that choice."""

    def __init__(self, keep_ratio: float, key_cache):
        self.keep_ratio = keep_ratio
        self.key_cache = key_cache
        self.kept_positions: dict[int, torch.Tensor] = {}
        self._value_importance: dict[int, torch.Tensor] = {}
        self._group_keep_positions: dict[int, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return self.keep_ratio < 1.0

    def set_value_importance(
        self,
        layer_idx: int,
        value_importance: torch.Tensor,
    ) -> None:
        self._value_importance[layer_idx] = value_importance

    def pop_value_importance(self, layer_idx: int) -> None:
        self._value_importance.pop(layer_idx, None)

    def _build_keep_positions(
        self,
        value_importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        score = value_importance[..., :seq_len].float().mean(dim=(0, 1))
        keep_count = max(1, math.ceil(seq_len * self.keep_ratio))

        force = torch.zeros(seq_len, dtype=torch.bool, device=score.device)

        # Attention sinks and local window
        force[:8] = True
        force[-min(seq_len, 64) :] = True

        remaining = max(0, keep_count - int(force.sum().item()))
        keep = force.clone()
        if remaining > 0:
            selectable = score.masked_fill(force, float("-inf"))
            topk = selectable.topk(
                min(remaining, seq_len), largest=True
            ).indices
            keep[topk] = True

        return keep.nonzero(as_tuple=False).flatten().sort().values

    def _group_key(self, layer_idx: int) -> int:
        """Identify the layers that have to evict the same positions.

        The key cache may fit one shared factor across a group of layers,
        which only means anything if row t is the same token in every layer
        of that group. Evicting per layer would misalign those rows and
        destroy the cross-layer structure the shared factor exists to
        capture, so the whole group reuses the decision made by its first
        layer. Caches without layer groups keep their per-layer behaviour.
        """
        get_group_bounds = getattr(self.key_cache, "_get_group_bounds", None)
        if not callable(get_group_bounds):
            return layer_idx
        return get_group_bounds(layer_idx)[0]

    def apply(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Drop evicted rows, returning the kept positions when it applied."""
        if not self.enabled:
            return key_states, value_states, None
        if key_states.shape[-2] == 1:  # decode
            return key_states, value_states, None

        group_key = self._group_key(layer_idx)
        keep_positions = self._group_keep_positions.get(group_key)
        if keep_positions is None:
            keep_positions = self._build_keep_positions(
                self._value_importance.get(layer_idx),
                key_states.shape[-2],
            )
            self._group_keep_positions[group_key] = keep_positions
        keep_positions = keep_positions.to(device=key_states.device)
        self.kept_positions[layer_idx] = keep_positions.detach().cpu()
        return (
            key_states.index_select(-2, keep_positions),
            value_states.index_select(-2, keep_positions),
            keep_positions,
        )

    def annotate_cache_kwargs(
        self,
        cache_kwargs: dict | None,
        keep_positions: torch.Tensor | None,
    ) -> dict | None:
        """Tell the backends which positions survived, if any were dropped."""
        if keep_positions is not None:
            cache_kwargs = {} if cache_kwargs is None else dict(cache_kwargs)
            cache_kwargs["kept_positions"] = keep_positions
            cache_kwargs["allow_sparse_kv"] = True
        elif self.enabled and cache_kwargs is not None:  # decode
            cache_kwargs = dict(cache_kwargs)
            cache_kwargs["allow_sparse_kv"] = True
        return cache_kwargs
