import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from efficiency import FusedLandmarkScorer
from .base import SharedRopeCache
from .config import (
    CompressedCacheConfig,
    build_key_cache,
    build_value_cache,
)


@dataclass
class DeferredValueUpdate:
    """A value update held back until its xKV layer group is decomposed."""

    value_states: torch.Tensor
    cache_kwargs: dict[str, Any]


@dataclass
class SelectiveLayerState:
    landmarks: torch.Tensor
    landmark_indices: torch.Tensor
    landmark_count: int
    prompt_len: int
    true_prompt_len: int
    outliers: torch.Tensor
    exact_positions: torch.Tensor
    exact_keys: torch.Tensor
    exact_values: torch.Tensor

    @property
    def key_overhead_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.landmarks,
                self.landmark_indices,
                self.exact_positions,
                self.outliers,
                self.exact_keys,
            )
        )

    @property
    def exact_value_nbytes(self) -> int:
        return self.exact_values.numel() * self.exact_values.element_size()

    @property
    def original_key_nbytes(self) -> int:
        return (
            self.exact_keys.size(0)
            * self.exact_keys.size(1)
            * self.true_prompt_len
            * self.exact_keys.size(3)
            * self.exact_keys.element_size()
        )


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
        self.selective_config = self.config.selective
        self.selective_reconstruction = self.selective_config.enabled
        self.selective_layers: dict[int, SelectiveLayerState] = {}
        self._selective_scorer = FusedLandmarkScorer()

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
        self._cache_context = dict(cache_context or {})
        self._temp_value_importance = {}
        self.eviction_keep_ratio = self.config.eviction_keep_ratio
        self.kept_positions = {}
        self._group_keep_positions: dict[int, torch.Tensor] = {}
        self._deferred_value_updates: dict[int, DeferredValueUpdate] = {}

    def _build_chunk_landmarks(
        self,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mean-key landmarks and validity masks for fixed-size chunks"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        chunk_size = self.selective_config.chunk_size
        num_chunks = math.ceil(seq_len / chunk_size)
        padded_len = num_chunks * chunk_size
        padding = padded_len - seq_len
        padded_keys = (
            torch.nn.functional.pad(key_states, (0, 0, 0, padding))
            if padding
            else key_states
        ).reshape(
            batch_size, num_heads, num_chunks, chunk_size, head_dim
        )
        valid = (
            torch.arange(padded_len, device=key_states.device).reshape(
                1, 1, num_chunks, chunk_size
            )
            < seq_len
        )
        counts = valid.sum(dim=3, keepdim=True).clamp_min(1)
        landmarks = padded_keys.sum(dim=3) / counts
        return padded_keys, landmarks, valid

    def _num_local_chunks(self, seq_len: int, num_chunks: int) -> int:
        """Count chunks overlapping the exact local-token window."""
        local_start = max(0, seq_len - self.selective_config.local_tokens)
        return num_chunks - local_start // self.selective_config.chunk_size

    def _select_outlier_chunks(
        self,
        padded_keys: torch.Tensor,
        landmarks: torch.Tensor,
        valid: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select nonlocal chunks whose tokens least match their landmark."""
        num_chunks = padded_keys.size(2)
        local_chunks = self._num_local_chunks(seq_len, num_chunks)
        similarity = torch.nn.functional.cosine_similarity(
            padded_keys,
            landmarks.unsqueeze(3),
            dim=-1,
        ).masked_fill(~valid, 1.0)
        similarity[..., -local_chunks:, :] = torch.inf
        outlier_count = min(
            self.selective_config.outlier_chunks,
            num_chunks - local_chunks,
        )
        return (
            similarity.amin(dim=3)
            .topk(outlier_count, dim=-1, largest=False)
            .indices
        )

    def _expand_chunk_positions(
        self,
        chunks: torch.Tensor,
    ) -> torch.Tensor:
        """Expand chunk indices into token positions."""
        offsets = torch.arange(
            self.selective_config.chunk_size,
            device=chunks.device,
        )
        return (
            chunks[..., None] * self.selective_config.chunk_size + offsets
        ).flatten(2)

    def _record_selective_overhead(self, layer_idx: int) -> None:
        """Cache persistent selective-key bytes and pass them to rank selection"""
        nbytes = self.selective_layers[layer_idx].key_overhead_nbytes
        record = getattr(self.key_cache, "set_selective_overhead", None)
        if callable(record):
            record(layer_idx, nbytes)

    def _store_selective_landmarks(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        true_seq_len: int | None = None,
    ) -> None:
        _, _, seq_len, head_dim = key_states.shape
        padded_keys, landmarks, valid = self._build_chunk_landmarks(key_states)
        outliers = self._select_outlier_chunks(
            padded_keys,
            landmarks,
            valid,
            seq_len,
        )
        local_chunks = self._num_local_chunks(seq_len, landmarks.size(2))
        eligible = torch.ones(
            landmarks.shape[:-1], device=landmarks.device, dtype=torch.bool
        )
        eligible.scatter_(2, outliers, False)
        eligible[..., -local_chunks:] = False
        landmark_count = int(eligible[0, 0].sum().item())
        landmark_indices = (
            torch.arange(landmarks.size(2), device=landmarks.device)
            .reshape(1, 1, -1)
            .expand_as(eligible)[eligible]
            .reshape(landmarks.size(0), landmarks.size(1), landmark_count)
        )
        landmarks = landmarks.gather(
            2,
            landmark_indices[..., None].expand(-1, -1, -1, head_dim),
        )
        landmark_padding = (-landmark_count) % 8
        if landmark_padding:
            landmarks = torch.nn.functional.pad(
                landmarks, (0, 0, 0, landmark_padding)
            )
            landmark_indices = torch.nn.functional.pad(
                landmark_indices, (0, landmark_padding)
            )
        outlier_positions = self._expand_chunk_positions(outliers)
        local_positions = (
            torch.arange(
                max(0, seq_len - self.selective_config.local_tokens),
                seq_len,
                device=key_states.device,
            )
            .reshape(1, 1, -1)
            .expand(key_states.size(0), key_states.size(1), -1)
        )
        positions = torch.cat([outlier_positions, local_positions], dim=-1)

        gather_idx = positions[..., None].expand(-1, -1, -1, head_dim)
        self.selective_layers[layer_idx] = SelectiveLayerState(
            landmarks=landmarks,
            landmark_indices=landmark_indices,
            landmark_count=landmark_count,
            prompt_len=seq_len,
            true_prompt_len=(
                seq_len if true_seq_len is None else true_seq_len
            ),
            outliers=outliers,
            exact_positions=positions,
            exact_keys=key_states.gather(2, gather_idx),
            exact_values=value_states.gather(2, gather_idx),
        )
        self._record_selective_overhead(layer_idx)

    def select_positions(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        state = self.selective_layers[layer_idx]
        landmarks = state.landmarks
        batch_size, num_query_heads, _, head_dim = query_states.shape
        num_kv_heads = landmarks.size(1)
        num_groups = num_query_heads // num_kv_heads
        grouped_query = query_states.reshape(
            batch_size, num_kv_heads, num_groups, head_dim
        )
        scores = self._selective_scorer.score(grouped_query, landmarks)
        scores = scores[..., : state.landmark_count]
        selected_chunks = math.ceil(
            self.selective_config.token_budget
            / self.selective_config.chunk_size
        )
        selected_indices = scores.topk(
            min(selected_chunks, state.landmark_count),
            dim=-1,
            sorted=False,
        ).indices
        chunks = state.landmark_indices.gather(2, selected_indices)
        prepare_chunks = getattr(
            self.key_cache, "prepare_selected_chunks", None
        )
        if callable(prepare_chunks):
            reordered = prepare_chunks(
                layer_idx,
                chunks,
                self.selective_config.chunk_size,
            )
            if reordered is not None:
                chunks = reordered
        offsets = torch.arange(
            self.selective_config.chunk_size,
            device=scores.device,
        )
        positions = (
            chunks[..., None] * self.selective_config.chunk_size + offsets
        ).reshape(batch_size, num_kv_heads, -1)
        positions.clamp_(max=state.prompt_len - 1)

        current_len = self.key_cache.get_seq_length(layer_idx)
        if current_len > state.prompt_len:
            suffix = (
                torch.arange(
                    state.prompt_len,
                    current_len,
                    device=scores.device,
                )
                .reshape(1, 1, -1)
                .expand(batch_size, num_kv_heads, -1)
            )
            positions = torch.cat([positions, suffix], dim=-1)
        return torch.cat([positions, state.exact_positions], dim=-1)

    def supports_selective_retrieval(self, layer_idx: int) -> bool:
        value_supports = getattr(
            self.value_cache, "supports_selective_retrieval", None
        )
        key_supports = getattr(
            self.key_cache, "supports_selective_retrieval", None
        )
        return (
            layer_idx in self.selective_layers
            and callable(value_supports)
            and value_supports(layer_idx)
            and (not callable(key_supports) or key_supports(layer_idx))
        )

    def append_selective(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any],
    ) -> None:
        # Selective decode bypasses update(), so end the prefill phase here.
        if key_states.size(-2) == 1 and getattr(
            self.key_cache, "prefill", False
        ):
            self.update_events()
        append_keys = getattr(self.key_cache, "append_decode", None)
        if callable(append_keys):
            append_keys(key_states, layer_idx, cache_kwargs)
        else:
            self.key_cache.update(key_states, layer_idx, cache_kwargs)
        self.value_cache.append_decode(value_states, layer_idx)

    def retrieve_selected(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.selective_layers[layer_idx]
        positions = positions[..., : -state.exact_positions.size(-1)]
        retrieve_keys = getattr(self.key_cache, "retrieve_selected", None)
        if callable(retrieve_keys):
            keys = retrieve_keys(layer_idx, positions)
        else:
            full_keys = self.key_cache.layers[layer_idx].tensor
            keys = full_keys.gather(
                2,
                positions[..., None].expand(-1, -1, -1, full_keys.size(-1)),
            )
        values = self.value_cache.retrieve_selected(keys, positions, layer_idx)
        keys = torch.cat([keys, state.exact_keys], dim=2)
        values = torch.cat([values, state.exact_values], dim=2)
        return keys, values

    def set_value_importance(
        self,
        layer_idx: int,
        value_importance: torch.Tensor,
    ) -> None:
        self._temp_value_importance[layer_idx] = value_importance

    def _build_keep_positions(
        self,
        value_importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        score = value_importance[..., :seq_len].float().mean(dim=(0, 1))
        keep_count = max(1, math.ceil(seq_len * self.eviction_keep_ratio))

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

    def _eviction_group_key(self, layer_idx: int) -> int:
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

    def _maybe_apply_eviction(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.eviction_keep_ratio == 1.0:
            return key_states, value_states, None
        if key_states.shape[-2] == 1:
            return key_states, value_states, None

        group_key = self._eviction_group_key(layer_idx)
        keep_positions = self._group_keep_positions.get(group_key)
        if keep_positions is None:
            keep_positions = self._build_keep_positions(
                self._temp_value_importance.get(layer_idx),
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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.size(-2) == 1 and getattr(
            self.key_cache, "prefill", False
        ):
            self.update_events()
        full_key_states, full_value_states = key_states, value_states
        key_states, value_states, keep_positions = self._maybe_apply_eviction(
            key_states, value_states, layer_idx
        )
        if self.selective_reconstruction and key_states.size(-2) > 1:
            self._store_selective_landmarks(
                layer_idx,
                key_states,
                value_states,
                true_seq_len=full_key_states.size(-2),
            )
        if keep_positions is not None:
            cache_kwargs = {} if cache_kwargs is None else dict(cache_kwargs)
            cache_kwargs["kept_positions"] = keep_positions
            cache_kwargs["allow_sparse_kv"] = True
        elif (
            self.eviction_keep_ratio < 1.0 and cache_kwargs is not None
        ):  # decode
            cache_kwargs = dict(cache_kwargs)
            cache_kwargs["allow_sparse_kv"] = True

        keys = self.key_cache.update(key_states, layer_idx, cache_kwargs)
        if cache_kwargs is None:
            cache_kwargs = {}
        if self._cache_context:
            cache_kwargs = {**self._cache_context, **cache_kwargs}
        if layer_idx in self._temp_value_importance:
            self._temp_value_importance.pop(layer_idx)
        value_group_bounds = self._value_update_group_bounds(layer_idx)
        if (
            value_group_bounds is not None
            and layer_idx != value_group_bounds[1]
        ):
            self._deferred_value_updates[layer_idx] = DeferredValueUpdate(
                value_states=value_states,
                cache_kwargs=dict(cache_kwargs),
            )
            values = value_states
        elif (
            value_group_bounds is not None
            and self._should_flush_deferred_value_updates(value_group_bounds)
        ):
            values = self._flush_deferred_value_updates(
                layer_idx,
                value_states,
                cache_kwargs,
                value_group_bounds,
            )
        else:
            if self.pass_keys_to_value_cache:
                self._attach_keys_to_value_cache_kwargs(
                    cache_kwargs,
                    layer_idx,
                    keys,
                )
            values = self.value_cache.update(
                value_states, layer_idx, cache_kwargs
            )
        if keep_positions is not None:
            return full_key_states, full_value_states
        return keys, values

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
        if not getattr(self.key_cache, "prefill", False):
            return None

        get_group_bounds = getattr(self.key_cache, "_get_group_bounds", None)
        if not callable(get_group_bounds):
            return None
        return get_group_bounds(layer_idx)

    def _should_flush_deferred_value_updates(
        self,
        bounds: tuple[int, int],
    ) -> bool:
        group_start, group_last_layer = bounds
        expected_layers = range(group_start, group_last_layer)
        deferred_layers = [
            idx
            for idx in expected_layers
            if idx in self._deferred_value_updates
        ]
        if not deferred_layers:
            return False
        return True

    def _attach_keys_to_value_cache_kwargs(
        self,
        cache_kwargs: dict[str, Any],
        layer_idx: int,
        keys: torch.Tensor,
    ) -> None:
        fn = getattr(self.key_cache, "get_reconstructed_keys_only", None)
        if callable(fn) and getattr(self.key_cache, "prefill", False):
            recon_keys = self.key_cache.get_reconstructed_keys_only(layer_idx)
            cache_kwargs["keys"] = recon_keys
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
            comp_ratio = 2 / ((1 / key_cr) + (1 / value_cr))
            if self.selective_layers:
                original_key_bytes = sum(
                    state.original_key_nbytes
                    for state in self.selective_layers.values()
                )
                selective_bytes = sum(
                    state.key_overhead_nbytes + state.exact_value_nbytes
                    for state in self.selective_layers.values()
                )
                selective_bytes += self._selective_scorer.nbytes
                selective_bytes += self.rope_cache.nbytes
                selective_bytes += getattr(
                    self.key_cache, "selective_reconstruction_nbytes", 0
                )
                return (2 * original_key_bytes) / (
                    (2 * original_key_bytes / comp_ratio) + selective_bytes
                )
            return comp_ratio
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
