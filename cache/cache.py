import copy
import math
import torch

from efficiency import FusedLandmarkScorer

from transformers.cache_utils import (
    Any,
    Iterable,
    PreTrainedConfig,
)

SELECTIVE_TOKEN_BUDGET = 2048
SELECTIVE_CHUNK_SIZE = 8
SELECTIVE_LOCAL_TOKENS = 32
SELECTIVE_OUTLIER_CHUNKS = 48


def get_cache(cache_type, CACHE_CLASSES, verbose=True):
    if cache_type not in CACHE_CLASSES:
        raise ValueError(f"Invalid cache type: {cache_type}")
    if verbose:
        print(f"Loading cache type {cache_type}")
    return CACHE_CLASSES[cache_type]


def adjust_lr_comp_ratios(key_cache_kwargs, value_cache_kwargs):
    if "comp_ratio" not in key_cache_kwargs:
        return

    comp_ratio = key_cache_kwargs["comp_ratio"]
    key_comp_ratio = 1.25 * comp_ratio
    value_comp_ratio = (5 / 6) * comp_ratio
    if value_comp_ratio <= 1.0:
        raise ValueError(
            f"Adjusted value comp ratio {value_comp_ratio} is <= 1.0, "
            f"consider starting with a higher comp ratio than {comp_ratio}."
        )

    key_cache_kwargs["comp_ratio"] = key_comp_ratio
    value_cache_kwargs["comp_ratio"] = value_comp_ratio


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
        log_key_recon_mse: bool = True,  # TODO: add full plumbing for this flag
        adjust_key_value_comp_ratio: bool = False,
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

        key_cache_type = key_cache_kwargs.pop("cache_type")
        value_cache_type = value_cache_kwargs.pop("cache_type")
        self.selective_reconstruction = key_cache_kwargs.pop(
            "selective_reconstruction", False
        )
        self.selective_landmarks = {}
        self.selective_landmark_indices = {}
        self.selective_landmark_counts = {}
        self.selective_prompt_lens = {}
        self.selective_outliers = {}
        self.selective_exact_positions = {}
        self.selective_exact_keys = {}
        self.selective_exact_values = {}
        self._selective_bytes = {}
        self._selective_scorer = FusedLandmarkScorer()

        # local import to avoid circular import at module load
        from .keys import KEY_CACHE_CLASSES
        from .values import VALUE_CACHE_CLASSES

        value_key_cache_kwargs = copy.deepcopy(key_cache_kwargs)
        if (
            adjust_key_value_comp_ratio
            and value_cache_type in KEY_CACHE_CLASSES
            and "baseline" not in (key_cache_type, value_cache_type)
        ):
            adjust_lr_comp_ratios(key_cache_kwargs, value_key_cache_kwargs)

        self.key_cache = get_cache(
            key_cache_type, KEY_CACHE_CLASSES, kwargs.get("verbose", True)
        )(
            ddp_cache_data=ddp_cache_data,
            **key_cache_kwargs,
        )

        if value_cache_type in VALUE_CACHE_CLASSES:
            self.value_cache = get_cache(
                value_cache_type,
                VALUE_CACHE_CLASSES,
                kwargs.get("verbose", True),
            )(
                ddp_cache_data=ddp_cache_data,
                **value_cache_kwargs,
            )
            self.pass_keys_to_value_cache = True
        elif value_cache_type in KEY_CACHE_CLASSES:
            value_cache_kwargs = value_key_cache_kwargs
            value_cache_kwargs["unrope_keys"] = False
            self.value_cache = get_cache(
                value_cache_type,
                KEY_CACHE_CLASSES,
                kwargs.get("verbose", True),
            )(
                ddp_cache_data=ddp_cache_data,
                **value_cache_kwargs,
            )
            self.pass_keys_to_value_cache = False
        else:
            raise ValueError(f"Invalid cache type: {value_cache_type}")
        self._cache_context = dict(kwargs.get("cache_context") or {})
        self._temp_value_importance = {}
        self.log_key_recon_mse = log_key_recon_mse
        self._key_recon_mses = []
        self.eviction_keep_ratio = kwargs.get("eviction_keep_ratio", 1.0)
        self.kept_positions = {}
        self._deferred_value_updates = {}

    @staticmethod
    def _build_chunk_landmarks(
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mean-key landmarks and validity masks for fixed-size chunks"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        num_chunks = math.ceil(seq_len / SELECTIVE_CHUNK_SIZE)
        padded_len = num_chunks * SELECTIVE_CHUNK_SIZE
        padding = padded_len - seq_len
        padded_keys = (
            torch.nn.functional.pad(key_states, (0, 0, 0, padding))
            if padding
            else key_states
        ).reshape(batch_size, num_heads, num_chunks, SELECTIVE_CHUNK_SIZE, head_dim)
        valid = (
            torch.arange(padded_len, device=key_states.device).reshape(
                1, 1, num_chunks, SELECTIVE_CHUNK_SIZE
            )
            < seq_len
        )
        counts = valid.sum(dim=3, keepdim=True).clamp_min(1)
        landmarks = padded_keys.sum(dim=3) / counts
        return padded_keys, landmarks, valid

    @staticmethod
    def _num_local_chunks(seq_len: int, num_chunks: int) -> int:
        """Count chunks overlapping the exact local-token window."""
        local_start = max(0, seq_len - SELECTIVE_LOCAL_TOKENS)
        return num_chunks - local_start // SELECTIVE_CHUNK_SIZE

    @classmethod
    def _select_outlier_chunks(
        cls,
        padded_keys: torch.Tensor,
        landmarks: torch.Tensor,
        valid: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select nonlocal chunks whose tokens least match their landmark."""
        num_chunks = padded_keys.size(2)
        local_chunks = cls._num_local_chunks(seq_len, num_chunks)
        similarity = torch.nn.functional.cosine_similarity(
            padded_keys,
            landmarks.unsqueeze(3),
            dim=-1,
        ).masked_fill(~valid, 1.0)
        similarity[..., -local_chunks:, :] = torch.inf
        outlier_count = min(SELECTIVE_OUTLIER_CHUNKS, num_chunks - local_chunks)
        return (
            similarity.amin(dim=3)
            .topk(outlier_count, dim=-1, largest=False)
            .indices
        )

    @staticmethod
    def _expand_chunk_positions(
        chunks: torch.Tensor,
    ) -> torch.Tensor:
        """Expand chunk indices into token positions."""
        offsets = torch.arange(
            SELECTIVE_CHUNK_SIZE,
            device=chunks.device,
        )
        return (
            chunks[..., None] * SELECTIVE_CHUNK_SIZE + offsets
        ).flatten(2)

    def _record_selective_overhead(self, layer_idx: int) -> None:
        """Cache persistent selective-key bytes and pass them to rank selection"""
        tensors = (
            self.selective_landmarks[layer_idx],
            self.selective_landmark_indices[layer_idx],
            self.selective_exact_positions[layer_idx],
            self.selective_outliers[layer_idx],
            self.selective_exact_keys[layer_idx],
        )
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        self._selective_bytes[layer_idx] = nbytes
        record = getattr(self.key_cache, "set_selective_overhead", None)
        if callable(record):
            record(layer_idx, nbytes)

    def _store_selective_landmarks(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
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
                max(0, seq_len - SELECTIVE_LOCAL_TOKENS),
                seq_len,
                device=key_states.device,
            )
            .reshape(1, 1, -1)
            .expand(key_states.size(0), key_states.size(1), -1)
        )
        positions = torch.cat([outlier_positions, local_positions], dim=-1)

        self.selective_landmarks[layer_idx] = landmarks
        self.selective_landmark_indices[layer_idx] = landmark_indices
        self.selective_landmark_counts[layer_idx] = landmark_count
        self.selective_prompt_lens[layer_idx] = seq_len
        self.selective_outliers[layer_idx] = outliers
        gather_idx = positions[..., None].expand(-1, -1, -1, head_dim)
        self.selective_exact_positions[layer_idx] = positions
        self.selective_exact_keys[layer_idx] = key_states.gather(2, gather_idx)
        self.selective_exact_values[layer_idx] = value_states.gather(
            2, gather_idx
        )
        self._record_selective_overhead(layer_idx)

    def select_positions(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        landmarks = self.selective_landmarks[layer_idx]
        batch_size, num_query_heads, _, head_dim = query_states.shape
        num_kv_heads = landmarks.size(1)
        num_groups = num_query_heads // num_kv_heads
        grouped_query = query_states.reshape(
            batch_size, num_kv_heads, num_groups, head_dim
        )
        landmark_count = self.selective_landmark_counts[layer_idx]
        scores = self._selective_scorer.score(grouped_query, landmarks)
        scores = scores[..., :landmark_count]
        prompt_len = self.selective_prompt_lens[layer_idx]
        selected_chunks = math.ceil(
            SELECTIVE_TOKEN_BUDGET / SELECTIVE_CHUNK_SIZE
        )
        selected_indices = scores.topk(
            min(selected_chunks, landmark_count),
            dim=-1,
            sorted=False,
        ).indices
        chunks = self.selective_landmark_indices[layer_idx].gather(
            2, selected_indices
        )
        prepare_chunks = getattr(
            self.key_cache, "prepare_selected_chunks", None
        )
        if callable(prepare_chunks):
            reordered = prepare_chunks(layer_idx, chunks, SELECTIVE_CHUNK_SIZE)
            if reordered is not None:
                chunks = reordered
        offsets = torch.arange(SELECTIVE_CHUNK_SIZE, device=scores.device)
        positions = (
            chunks[..., None] * SELECTIVE_CHUNK_SIZE + offsets
        ).reshape(batch_size, num_kv_heads, -1)
        positions.clamp_(max=prompt_len - 1)

        current_len = self.key_cache.get_seq_length(layer_idx)
        if current_len > prompt_len:
            suffix = (
                torch.arange(prompt_len, current_len, device=scores.device)
                .reshape(1, 1, -1)
                .expand(batch_size, num_kv_heads, -1)
            )
            positions = torch.cat([positions, suffix], dim=-1)
        return torch.cat(
            [positions, self.selective_exact_positions[layer_idx]], dim=-1
        )

    def supports_selective_retrieval(self, layer_idx: int) -> bool:
        value_supports = getattr(
            self.value_cache, "supports_selective_retrieval", None
        )
        key_supports = getattr(
            self.key_cache, "supports_selective_retrieval", None
        )
        return (
            layer_idx in self.selective_landmarks
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
        exact_positions = self.selective_exact_positions[layer_idx]
        positions = positions[..., : -exact_positions.size(-1)]
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
        keys = torch.cat([keys, self.selective_exact_keys[layer_idx]], dim=2)
        values = torch.cat(
            [values, self.selective_exact_values[layer_idx]], dim=2
        )
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
            topk = selectable.topk(min(remaining, seq_len), largest=True).indices
            keep[topk] = True

        return keep.nonzero(as_tuple=False).flatten().sort().values

    def _maybe_apply_eviction(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.eviction_keep_ratio == 1.0:
            return key_states, value_states, None
        value_importance = self._temp_value_importance.get(layer_idx)
        if key_states.shape[-2] == 1:
            return key_states, value_states, None

        keep_positions = self._build_keep_positions(value_importance, key_states.shape[-2]).to(
            device=key_states.device
        )
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
        full_key_states, full_value_states = key_states, value_states
        if self.selective_reconstruction and key_states.size(-2) > 1:
            self._store_selective_landmarks(layer_idx, key_states, value_states)
        key_states, value_states, keep_positions = self._maybe_apply_eviction(
            key_states, value_states, layer_idx
        )
        if keep_positions is not None:
            cache_kwargs = {} if cache_kwargs is None else dict(cache_kwargs)
            cache_kwargs["kept_positions"] = keep_positions
            cache_kwargs["allow_sparse_kv"] = True
        elif self.eviction_keep_ratio < 1.0 and cache_kwargs is not None: # decode
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
            self._deferred_value_updates[layer_idx] = (
                value_states,
                dict(cache_kwargs),
                key_states,
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
                key_states,
                value_group_bounds,
            )
        else:
            if self.pass_keys_to_value_cache or self.log_key_recon_mse:
                self._attach_keys_to_value_cache_kwargs(
                    cache_kwargs,
                    layer_idx,
                    key_states,
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
        key_states: torch.Tensor,
        keys: torch.Tensor,
    ) -> None:
        fn = getattr(self.key_cache, "get_reconstructed_keys_only", None)
        if callable(fn) and getattr(self.key_cache, "prefill", False):
            recon_keys = self.key_cache.get_reconstructed_keys_only(layer_idx)
            if self.log_key_recon_mse and recon_keys is not None:
                self._key_recon_mses.append(
                    torch.nn.functional.mse_loss(recon_keys, key_states).item()
                )
            cache_kwargs["keys"] = recon_keys
        else:
            cache_kwargs["keys"] = keys

    def _flush_deferred_value_updates(
        self,
        layer_idx: int,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any],
        key_states: torch.Tensor,
        bounds: tuple[int, int],
    ) -> torch.Tensor:
        group_start, group_last_layer = bounds
        updates = []
        for group_layer_idx in range(group_start, group_last_layer + 1):
            if group_layer_idx == layer_idx:
                continue
            deferred_values, deferred_kwargs, original_keys = (
                self._deferred_value_updates.pop(group_layer_idx)
            )
            recon_keys = self.key_cache.get_reconstructed_keys_only(
                group_layer_idx
            )
            if self.log_key_recon_mse and recon_keys is not None:
                self._key_recon_mses.append(
                    torch.nn.functional.mse_loss(
                        recon_keys, original_keys
                    ).item()
                )
            deferred_kwargs["keys"] = recon_keys
            updates.append((group_layer_idx, deferred_values, deferred_kwargs))

        current_recon_keys = self.key_cache.get_reconstructed_keys_only(
            layer_idx
        )
        if self.log_key_recon_mse and current_recon_keys is not None:
            self._key_recon_mses.append(
                torch.nn.functional.mse_loss(
                    current_recon_keys, key_states
                ).item()
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
            if self.selective_exact_keys:
                original_key_bytes = sum(
                    tensor.size(0)
                    * tensor.size(1)
                    * self.selective_prompt_lens[layer_idx]
                    * tensor.size(3)
                    * tensor.element_size()
                    for layer_idx, tensor in self.selective_exact_keys.items()
                )
                selective_bytes = sum(
                    self._selective_bytes.get(layer_idx, 0)
                    for layer_idx in self.selective_exact_keys
                )
                selective_bytes += sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in self.selective_exact_values.values()
                )
                selective_bytes += self._selective_scorer.nbytes
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

    @property
    def key_recon_mse(self) -> float | None:
        if not self._key_recon_mses:
            return None
        return sum(self._key_recon_mses) / len(self._key_recon_mses)

    @property
    def eta_mean(self):
        return getattr(self.key_cache, "eta_mean", None)

    @property
    def eta_std(self):
        return getattr(self.key_cache, "eta_std", None)

    @property
    def value_recon_mse(self) -> float | None:
        if hasattr(self.value_cache, "recon_mse"):
            return self.value_cache.recon_mse
        return None

    @property
    def value_mlp_log_events(self) -> list[dict[str, Any]]:
        return getattr(self.value_cache, "mlp_log_events", [])

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
