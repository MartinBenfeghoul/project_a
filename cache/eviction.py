"""Physical eviction of prefill KV tokens."""

import math

import torch


class EvictionPolicy:
    """Chooses which prefill positions to keep, and applies that choice."""

    sink_tokens = 8
    local_tokens = 64

    def __init__(self, keep_ratio: float, key_cache):
        self.keep_ratio = keep_ratio
        self.key_cache = key_cache
        self.kept_positions: dict[int, torch.Tensor] = {}
        self._value_importance: dict[int, torch.Tensor] = {}
        self._group_keep_positions: dict[int, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return self.keep_ratio < 1.0

    @property
    def compression_ratio(self) -> float:
        return 1 / self.keep_ratio

    def set_value_importance(
        self,
        layer_idx: int,
        value_importance: torch.Tensor,
    ) -> None:
        self._value_importance[layer_idx] = value_importance

    def pop_value_importance(self, layer_idx: int) -> None:
        self._value_importance.pop(layer_idx, None)

    def _importance_score(
        self,
        value_importance: torch.Tensor,
        seq_len: int,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Average importance per position across the batch and heads."""
        score = value_importance[..., :seq_len].float()
        if padding_mask is None:
            return score.mean(dim=(0, 1))
        weight = padding_mask[:, None, :].to(score.device, score.dtype)
        counts = weight.sum(dim=(0, 1)).mul(score.size(1)).clamp_min(1.0)
        return (score * weight).sum(dim=(0, 1)) / counts

    def _forced_positions(
        self,
        seq_len: int,
        padding_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Attention sinks and the local window, which are never evicted."""
        force = torch.zeros(seq_len, dtype=torch.bool, device=device)
        force[-min(seq_len, self.local_tokens) :] = True
        if padding_mask is None:
            force[: self.sink_tokens] = True
            return force
        starts = (~padding_mask).sum(dim=1).to(device)
        offsets = torch.arange(self.sink_tokens, device=device)
        sinks = (starts[:, None] + offsets).clamp_(max=seq_len - 1)
        force[sinks.flatten()] = True
        return force

    def _build_keep_positions(
        self,
        value_importance: torch.Tensor,
        seq_len: int,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score = self._importance_score(value_importance, seq_len, padding_mask)
        usable = (
            torch.ones(seq_len, dtype=torch.bool, device=score.device)
            if padding_mask is None
            else padding_mask.any(dim=0).to(score.device)
        )
        keep_count = max(1, math.ceil(int(usable.sum()) * self.keep_ratio))

        force = self._forced_positions(seq_len, padding_mask, score.device)
        force &= usable

        remaining = max(0, keep_count - int(force.sum().item()))
        keep = force.clone()
        if remaining > 0:
            selectable = score.masked_fill(force | ~usable, float("-inf"))
            available = int((usable & ~force).sum())
            topk = selectable.topk(
                min(remaining, available), largest=True
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
        padding_mask: torch.Tensor | None = None,
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
                padding_mask,
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
