import time
import warnings
from typing import Any

import torch
import torch.nn.functional as F

from utils.matrix_decomposition import (
    DECOMP_METHODS,
    calc_segment_store_compression_ratio,
    decompose_to_segment_store,
    reconstruct_segments,
)
from utils.segmentation import (
    build_cluster_segment_ranges,
    find_thresholds,
    group_keys_by_cluster,
    group_sequences_by_cluster,
    kmeans_cluster_sequences,
    restore_grouped_keys_order,
    restore_grouped_sequences_order,
)
from utils.logging import (
    init_timing_stats,
    update_timing_stats,
    sync_cuda,
)
from .base import SingleTensorCache
from .turboquant import TurboQuantCache


def get_expected_seq_len(cache_kwargs):
    if cache_kwargs is not None:
        cache_position = cache_kwargs.get("cache_position", None)
        if cache_position is not None:
            return cache_position[..., -1] + 1
    return None


def check_recon_length(recon_keys, cache_kwargs):
    exp_seq_len = get_expected_seq_len(cache_kwargs)
    if exp_seq_len is not None and recon_keys.size(-2) != exp_seq_len:
        raise ValueError(
            f"Reconstructed keys have seq_len {recon_keys.size(-2)} "
            f"but cache_position expects {exp_seq_len}."
        )
    elif exp_seq_len is None:
        raise ValueError(
            "Expected sequence length could not be determined from cache_kwargs."
        )


class DecomposedKeysCache(SingleTensorCache):
    timing_stats = None

    def __init__(
        self,
        *args,
        decomposition_method: str = None,
        local_window: int = 0,
        log_timing_stats: bool = False,
        rank_selection: str = "comp_ratio",
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        decomp_n_iter: int = 3,
        lr: float = 1e-2,
        unrope_keys: bool = False,
        quantise_a: bool = False,
        quantise_b: bool = False,
        compressor_bits: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if decomposition_method not in DECOMP_METHODS:
            raise ValueError(
                f"Decomposition method {decomposition_method} not found in "
                f"available methods: {DECOMP_METHODS.keys()}"
            )

        self.decomposition_method = decomposition_method
        self.decompose = DECOMP_METHODS[decomposition_method]
        self.local_window = local_window
        self.rank_selection = rank_selection
        self.r = comp_ratio
        self.e = energy_threshold
        self.decomp_n = decomp_n_iter
        self.lr = lr
        self.unrope_keys = unrope_keys
        self.quantise_a = quantise_a
        self.quantise_b = quantise_b
        self.compressor_bits = compressor_bits

        self.prefill = True
        self.lr_keys = {}
        self.comp_ratio = None
        self.compressed_len = 0
        self.log_timing_stats = log_timing_stats
        type(self).timing_stats = (
            init_timing_stats() if self.log_timing_stats else None
        )

    def calc_compression_ratio(self):
        default_ratio = (
            self.r
            if self.rank_selection == "comp_ratio"
            and not self.quantise_a
            and not self.quantise_b
            else None
        )
        return calc_segment_store_compression_ratio(
            self.lr_keys, default_ratio=default_ratio
        )

    def update_events(self, *args, **kwargs):
        self.prefill = False

    def _decomposition_kwargs(self):
        return {
            "rank_selection": self.rank_selection,
            "cr": self.r,
            "energy_threshold": self.e,
            "n_iter": self.decomp_n,
            "lr": self.lr,
            "quantise_a": self.quantise_a,
            "quantise_b": self.quantise_b,
            "compressor_bits": self.compressor_bits,
        }

    def _store_layer_segments(self, layer_idx, layer_segments):
        self.lr_keys[layer_idx] = layer_segments
        self.comp_ratio = self.calc_compression_ratio()

    def _apply_lr_keys_batch_op(self, op):
        for layer_idx, layer_segments in self.lr_keys.items():
            self.lr_keys[layer_idx] = op(layer_segments)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        beam_idx_list = beam_idx.tolist()
        self._apply_lr_keys_batch_op(
            lambda layer_segments: [
                layer_segments[idx] for idx in beam_idx_list
            ]
        )

    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats)
        self._apply_lr_keys_batch_op(
            lambda layer_segments: [
                batch_segments
                for batch_segments in layer_segments
                for _ in range(repeats)
            ]
        )

    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices)
        indices_list = indices.tolist()
        self._apply_lr_keys_batch_op(
            lambda layer_segments: [layer_segments[idx] for idx in indices_list]
        )

    def _get_suffix_start(self, keys, segment_ranges=None):
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative.")

        if segment_ranges is None:
            prefix_end = keys.size(-2)
        else:
            prefix_ends = [
                ranges[-1][1] if ranges else 0 for ranges in segment_ranges
            ]
            prefix_end = prefix_ends[0] if prefix_ends else 0
            if any(end != prefix_end for end in prefix_ends[1:]):
                raise ValueError(
                    "All batches must share the same compressed prefix length."
                )

        return max(0, prefix_end - self.local_window)

    def _trim_segment_ranges(self, segment_ranges, suffix_start):
        if segment_ranges is None:
            return None

        trimmed_ranges = []
        for batch_ranges in segment_ranges:
            trimmed_batch_ranges = []
            for start_idx, end_idx in batch_ranges:
                end_idx = min(end_idx, suffix_start)
                if start_idx < end_idx:
                    trimmed_batch_ranges.append((start_idx, end_idx))
            trimmed_ranges.append(trimmed_batch_ranges)
        return trimmed_ranges

    def _decompose_keys(
        self, keys, layer_idx, segment_ranges=None, cache_kwargs=None
    ):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        suffix_start = self._get_suffix_start(keys, segment_ranges)
        self.compressed_len = suffix_start
        decomp_keys = keys[..., :suffix_start, :]
        if self.unrope_keys:
            decomp_keys = self.layers[layer_idx]._undo_rope(
                decomp_keys,
                cache_kwargs,
                prefill=self.prefill,
                compressed_len=suffix_start,
            )
        trimmed_segment_ranges = self._trim_segment_ranges(
            segment_ranges, suffix_start
        )

        if suffix_start == 0:
            layer_segments = [[] for _ in range(keys.size(0))]
        else:
            if trimmed_segment_ranges is None or self.unrope_keys:
                keys_to_decompose = decomp_keys
            else:
                keys_to_decompose = keys

            layer_segments = decompose_to_segment_store(
                keys_to_decompose,
                self.decompose,
                segment_ranges=trimmed_segment_ranges,
                **self._decomposition_kwargs(),
            )

        if self.log_timing_stats:
            sync_cuda(keys)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )

        self._store_layer_segments(layer_idx, layer_segments)
        return suffix_start

    def _reconstruct_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        recon_keys = reconstruct_segments(self.lr_keys[layer_idx], keys)
        if self.unrope_keys:
            recon_keys = self.layers[layer_idx]._apply_rope(
                recon_keys,
                compressed_len=self.compressed_len,
            )
        if self.log_timing_stats:
            sync_cuda(recon_keys)
            update_timing_stats(
                type(self), "reconstruct", time.perf_counter() - start_time
            )
        return recon_keys

    def get_reconstructed_keys_only(self, layer_idx):
        """Utility method to get reconstructed keys without adding suffix keys.
        If local_window > 0, the returned reconstructed keys will still be concatenated
        with the local window of suffix keys for consistency of output sequence length.
        Furthermore, it returns None if no compressed keys are found for the layer.
        """
        if not self.lr_keys.get(layer_idx):
            warnings.warn(
                f"No compressed keys found for layer {layer_idx}. "
                "get_reconstructed_keys_only will return None."
            )
            return None
        suffix_keys = self.layers[layer_idx].tensor
        if self.local_window > 0:
            assert (
                suffix_keys.size(-2) > 0
            ), f"Suffix keys are required for local_window > 0 but got shape {suffix_keys.shape}."
            return self._reconstruct_keys(
                keys=suffix_keys[..., : self.local_window, :],
                layer_idx=layer_idx,
            )
        else:
            return self._reconstruct_keys(
                keys=suffix_keys[..., :0, :],
                layer_idx=layer_idx,
            )


class LowRankKeysCache(DecomposedKeysCache):
    def update(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        keys = super().update(key_states, layer_idx, cache_kwargs)

        if self.prefill:
            suffix_start = self._decompose_keys(
                keys, layer_idx, cache_kwargs=cache_kwargs
            )
            self._evict(layer_idx=layer_idx, end_idx=suffix_start)
            return keys
        elif self.lr_keys.get(layer_idx, False):
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys
        else:
            raise Exception(
                "Prefill is set to False and no low_rank keys were found."
            )


class XKVKeysCache(DecomposedKeysCache):
    """
    Cross-layer SVD key cache following xKV's grouped-layer formulation.

    The cache groups adjacent layers into contiguous blocks of size
    `layer_group_size`, flattens heads within each layer, performs one SVD
    over the horizontally concatenated key tensors of the full group, stores
    the shared folded left factor `A = U S` on the last layer of the group,
    and stores each layer's own right factor on that layer.
    """

    def __init__(
        self,
        *args,
        layer_group_size: int = 2,
        num_layers: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive.")
        if num_layers is not None and num_layers <= 0:
            raise ValueError("num_layers must be positive when provided.")
        if self.decomposition_method != "svd":
            raise NotImplementedError(
                "XKVKeysCache currently supports decomposition_method='svd' "
                "only so the shared folded left factor can be stored "
                "explicitly."
            )
        if self.quantise_a or self.quantise_b:
            raise NotImplementedError(
                "XKVKeysCache does not yet support quantised shared/grouped "
                "factors."
            )

        self.layer_group_size = layer_group_size
        self.num_layers = num_layers
        self.shared_a = {}
        self.group_metadata = {}

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError(
            "XKVKeysCache batch ops are intentionally left unimplemented."
        )

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError(
            "XKVKeysCache batch ops are intentionally left unimplemented."
        )

    def batch_select_indices(self, indices: torch.Tensor):
        raise NotImplementedError(
            "XKVKeysCache batch ops are intentionally left unimplemented."
        )

    def _get_group_bounds(self, layer_idx):
        group_start = (layer_idx // self.layer_group_size) * self.layer_group_size
        group_last = group_start + self.layer_group_size - 1
        if self.num_layers is not None:
            group_last = min(group_last, self.num_layers - 1)
        return group_start, group_last

    def calc_compression_ratio(self):
        default_ratio = (
            self.r
            if self.rank_selection == "comp_ratio"
            and not self.quantise_a
            and not self.quantise_b
            else None
        )
        if default_ratio is not None:
            return default_ratio

        layers_by_group: dict[int, list[int]] = {}
        for layer_idx, metadata in self.group_metadata.items():
            layers_by_group.setdefault(
                metadata["group_last_layer"], []
            ).append(layer_idx)

        crs = 0.0
        num_segments = 0
        for group_last, group_layers in layers_by_group.items():
            for batch_idx, batch_shared in enumerate(self.shared_a[group_last]):
                for seg_idx, shared_segment in enumerate(batch_shared):
                    A = shared_segment["factor"]
                    m, k = A.size(-2), A.size(-1)
                    n = sum(
                        self.lr_keys[l][batch_idx][seg_idx]["factor"].size(-1)
                        for l in group_layers
                    )
                    crs += (m * n) / (k * (m + n))
                    num_segments += 1
        return crs / num_segments if num_segments > 0 else 0.0

    def _decompose_keys(self, layer_idx, cache_kwargs=None):
        if self.log_timing_stats:
            sync_cuda(self.layers[layer_idx].tensor)
            start_time = time.perf_counter()

        group_start, group_last_layer = self._get_group_bounds(layer_idx)
        if group_last_layer != layer_idx:
            raise ValueError(
                "xKV decomposition should only be triggered from the last "
                f"layer of a group, got layer {layer_idx} for group ending "
                f"at layer {group_last_layer}."
            )

        group_layers = tuple(range(group_start, group_last_layer + 1))
        group_tensors = [self.layers[i].tensor for i in group_layers]
        batch_size = group_tensors[-1].size(0)
        assert all(
            t.size(-2) == group_tensors[-1].size(-2) for t in group_tensors
        ), "All layers in an xKV group must share the same cached length."

        suffix_start = self._get_suffix_start(group_tensors[-1])
        self.compressed_len = suffix_start
        if suffix_start == 0:
            shared_segments = [[] for _ in range(batch_size)]
            per_layer_segments = [
                [[] for _ in range(batch_size)] for _ in group_layers
            ]
        else:
            prefix_tensors = []
            split_sizes = []
            for group_layer_idx, tensor in zip(group_layers, group_tensors):
                prefix_tensor = tensor[..., :suffix_start, :]
                if self.unrope_keys:
                    prefix_tensor = self.layers[group_layer_idx]._undo_rope(
                        prefix_tensor,
                        cache_kwargs,
                        prefill=self.prefill,
                        compressed_len=suffix_start,
                    )
                prefix_flat = prefix_tensor.transpose(1, 2).reshape(
                    prefix_tensor.size(0),
                    prefix_tensor.size(-2),
                    -1,
                )
                prefix_tensors.append(prefix_flat)
                split_sizes.append(prefix_flat.size(-1))

            group_prefix = torch.cat(prefix_tensors, dim=-1).unsqueeze(1)
            segment_ranges = [[(0, suffix_start)] for _ in range(batch_size)]
            grouped_segments = decompose_to_segment_store(
                group_prefix,
                self.decompose,
                segment_ranges=segment_ranges,
                **self._decomposition_kwargs(),
            )

            shared_segments = [[] for _ in range(batch_size)]
            per_layer_segments = [
                [[] for _ in range(batch_size)] for _ in group_layers
            ]
            for batch_idx, batch_segments in enumerate(grouped_segments):
                for segment in batch_segments:
                    shared_factor, grouped_right = segment["factors"]
                    shared_segments[batch_idx].append(
                        {
                            "range": segment["range"],
                            "factor": shared_factor,
                        }
                    )
                    right_factors = torch.split(
                        grouped_right, split_sizes, dim=-1
                    )
                    for layer_segments, right_factor in zip(
                        per_layer_segments, right_factors
                    ):
                        layer_segments[batch_idx].append(
                            {
                                "range": segment["range"],
                                "factor": right_factor,
                            }
                        )

        self.shared_a[group_last_layer] = shared_segments
        for group_layer_idx, layer_segments in zip(
            group_layers, per_layer_segments
        ):
            self.lr_keys[group_layer_idx] = layer_segments
            self.group_metadata[group_layer_idx] = {
                "group_last_layer": group_last_layer,
                "compressed_len": suffix_start,
            }
        self.comp_ratio = self.calc_compression_ratio()

        if self.log_timing_stats:
            sync_cuda(self.layers[layer_idx].tensor)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )
        return suffix_start

    def _reconstruct_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        metadata = self.group_metadata.get(layer_idx)
        if metadata is None:
            raise ValueError(
                f"No xKV metadata found for layer {layer_idx}."
            )

        shared_segments = self.shared_a[metadata["group_last_layer"]]
        layer_segments = self.lr_keys[layer_idx]

        paired_segments = [
            [
                {
                    "range": s["range"],
                    "factors": (s["factor"], l["factor"]),
                }
                for s, l in zip(batch_shared, batch_layer)
            ]
            for batch_shared, batch_layer in zip(
                shared_segments, layer_segments
            )
        ]

        flat_dim = keys.size(1) * keys.size(-1)
        empty_suffix = keys.new_empty(keys.size(0), 1, 0, flat_dim)
        prefix_flat = reconstruct_segments(
            paired_segments, empty_suffix
        ).squeeze(1)

        batch_size, num_heads, _, head_dim = keys.shape
        prefix_keys = prefix_flat.reshape(
            batch_size,
            prefix_flat.size(1),
            num_heads,
            head_dim,
        ).transpose(1, 2)

        if self.unrope_keys:
            prefix_keys = self.layers[layer_idx]._apply_rope(
                prefix_keys,
                compressed_len=metadata["compressed_len"],
            )
        recon_keys = torch.cat([prefix_keys, keys], dim=-2)

        if self.log_timing_stats:
            sync_cuda(recon_keys)
            update_timing_stats(
                type(self), "reconstruct", time.perf_counter() - start_time
            )
        return recon_keys

    def update(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        keys = super().update(key_states, layer_idx, cache_kwargs)

        if self.prefill:
            group_start, group_last_layer = self._get_group_bounds(layer_idx)
            if layer_idx != group_last_layer:
                return keys

            suffix_start = self._decompose_keys(layer_idx, cache_kwargs=cache_kwargs)
            for group_layer_idx in range(group_start, group_last_layer + 1):
                self._evict(layer_idx=group_layer_idx, end_idx=suffix_start)
            return keys

        if self.lr_keys.get(layer_idx, False):
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys

        raise Exception(
            "Prefill is set to False and no xKV factors were found for "
            f"layer {layer_idx}. If this layer is part of a terminal partial "
            "group, pass num_layers through later plumbing so the group "
            "boundary is known."
        )


class SurpriseLRKCache(DecomposedKeysCache):
    def __init__(
        self,
        *args,
        gamma: float = 3.0,
        min_size: int = 8,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.gamma = gamma
        self.min_size = min_size

    def update_events(self, logits, labels):
        B, T, V = logits.shape
        surprise = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            reduction="none",
        ).reshape(B, T)

        self.events = []
        n_events = 0
        for b in range(B):
            events = find_thresholds(
                surprise[b],
                threshold_param=self.gamma,
                min_size=self.min_size,
            )[0]
            # The final key has no surprise score, so extend the last segment
            # boundary to include the last token from prefill.
            events[-1] += 1
            self.events.append(events)
            n_events += len(events) - 1
        self.prefill = False
        return n_events / B

    def _build_segment_ranges(self):
        segment_ranges = []
        for events in self.events:
            batch_ranges = []
            for i, start_idx in enumerate(events[:-1]):
                batch_ranges.append((start_idx, events[i + 1]))
            segment_ranges.append(batch_ranges)
        return segment_ranges

    def update(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        keys = super().update(key_states, layer_idx, cache_kwargs)

        if self.prefill:
            if self.unrope_keys:
                self.layers[layer_idx]._cache_rope_state(keys, cache_kwargs)
            return keys

        if layer_idx not in self.lr_keys:
            suffix_start = self._decompose_keys(
                keys,
                layer_idx,
                segment_ranges=self._build_segment_ranges(),
                cache_kwargs=cache_kwargs,
            )
            keys = keys[..., suffix_start:, :]
            self._evict(layer_idx=layer_idx, end_idx=suffix_start)

        recon_keys = self._reconstruct_keys(keys, layer_idx)
        check_recon_length(recon_keys, cache_kwargs)
        return recon_keys


class KMeansMixin:
    """Shared kmeans configuration helpers for low-rank key caches."""

    def _init_kmeans(
        self,
        n_clusters: int = 8,
        kmeans_cluster_size: float | None = None,
        kmeans_n_iter: int = 8,
        kmeans_init: str = "infllm",
        kmeans_dtype: torch.dtype | str = torch.float32,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if kmeans_cluster_size is not None and kmeans_cluster_size <= 0:
            raise ValueError("kmeans_cluster_size must be positive.")
        if isinstance(kmeans_dtype, str):
            try:
                kmeans_dtype = getattr(torch, kmeans_dtype)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown kmeans dtype: {kmeans_dtype}"
                ) from exc
        if not isinstance(kmeans_dtype, torch.dtype):
            raise TypeError(
                "kmeans_dtype must be a torch.dtype or dtype name."
            )
        self.n_clusters = n_clusters
        self.kmeans_cluster_size = kmeans_cluster_size
        self.kmeans_n = kmeans_n_iter
        self.kmeans_init = kmeans_init
        self.kmeans_dtype = kmeans_dtype

    def _get_cluster_count(self, seq_len: int) -> int:
        if self.kmeans_cluster_size is not None:
            cluster_count = int(round(seq_len / self.kmeans_cluster_size))
        else:
            cluster_count = self.n_clusters
        return max(1, min(cluster_count, seq_len))


class KMeansLRKCache(KMeansMixin, DecomposedKeysCache):
    def __init__(
        self,
        *args,
        n_clusters: int = 8,
        kmeans_cluster_size: float | None = None,
        kmeans_n_iter: int = 8,
        kmeans_init: str = "infllm",
        kmeans_dtype: torch.dtype | str = torch.float32,
        kmeans_avg_heads: bool = False,
        kmeans_per_head: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_kmeans(
            n_clusters=n_clusters,
            kmeans_cluster_size=kmeans_cluster_size,
            kmeans_n_iter=kmeans_n_iter,
            kmeans_init=kmeans_init,
            kmeans_dtype=kmeans_dtype,
        )
        self.kmeans_avg_heads = kmeans_avg_heads
        self.kmeans_per_head = kmeans_per_head
        if self.kmeans_avg_heads and self.kmeans_per_head:
            raise ValueError(
                "kmeans_avg_heads and kmeans_per_head are mutually exclusive."
            )
        if self.kmeans_per_head:
            self.kmeans_mode = "per_head"
        elif self.kmeans_avg_heads:
            self.kmeans_mode = "avg_heads"
        else:
            self.kmeans_mode = "concat_heads"
        self.cluster_metadata = {}

    def _cluster_prefix(self, prefix_keys):
        batch_size, _, seq_len, _ = prefix_keys.shape
        if seq_len == 0:
            empty_long = torch.empty(
                batch_size, 0, dtype=torch.long, device=prefix_keys.device
            )
            return (
                prefix_keys,
                empty_long,
                empty_long,
                [[] for _ in range(batch_size)],
            )

        if self.kmeans_mode == "per_head":
            _, num_heads, _, head_dim = prefix_keys.shape
            flat_prefix = prefix_keys.reshape(
                batch_size * num_heads, seq_len, head_dim
            )
            cluster_count = self._get_cluster_count(seq_len)
            assignments = kmeans_cluster_sequences(
                flat_prefix,
                n_clusters=cluster_count,
                n_iter=max(1, self.kmeans_n),
                kmeans_init=self.kmeans_init,
                dtype=self.kmeans_dtype,
            )
            grouped_prefix, _, inverse_permutation = group_sequences_by_cluster(
                flat_prefix, assignments
            )
            segment_ranges = build_cluster_segment_ranges(
                assignments,
                n_clusters=cluster_count,
            )
            return (
                grouped_prefix,
                assignments.reshape(batch_size, num_heads, seq_len),
                inverse_permutation.reshape(batch_size, num_heads, seq_len),
                segment_ranges,
            )

        if self.kmeans_mode == "avg_heads":
            token_features = prefix_keys.mean(dim=1)
        else:
            token_features = prefix_keys.transpose(1, 2).reshape(
                batch_size, seq_len, -1
            )
        cluster_count = self._get_cluster_count(seq_len)
        assignments = kmeans_cluster_sequences(
            token_features,
            n_clusters=cluster_count,
            n_iter=max(1, self.kmeans_n),
            kmeans_init=self.kmeans_init,
            dtype=self.kmeans_dtype,
        )
        grouped_prefix, _, inverse_permutation = group_keys_by_cluster(
            prefix_keys, assignments
        )
        segment_ranges = build_cluster_segment_ranges(
            assignments,
            n_clusters=cluster_count,
        )
        return grouped_prefix, assignments, inverse_permutation, segment_ranges

    def _decompose_keys(
        self,
        keys,
        prefix_end,
        layer_idx,
        cache_kwargs=None,
    ):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        if prefix_end < 0:
            raise ValueError("prefix_end must be non-negative.")
        if prefix_end > keys.size(-2):
            raise ValueError(
                "prefix_end cannot exceed the cache sequence length."
            )
        suffix_start = max(0, prefix_end - self.local_window)
        self.compressed_len = suffix_start

        if suffix_start == 0:
            batch_size, num_heads = keys.shape[:2]
            if self.kmeans_mode == "per_head":
                layer_segments = [[] for _ in range(batch_size * num_heads)]
                assignments = torch.empty(
                    batch_size,
                    num_heads,
                    0,
                    dtype=torch.long,
                    device=keys.device,
                )
            else:
                layer_segments = [[] for _ in range(batch_size)]
                assignments = torch.empty(
                    batch_size, 0, dtype=torch.long, device=keys.device
                )
            inverse_permutation = assignments
        else:
            prefix_keys = keys[..., :suffix_start, :]
            if self.unrope_keys:
                prefix_keys = self.layers[layer_idx]._undo_rope(
                    prefix_keys,
                    cache_kwargs,
                    prefill=self.prefill,
                    compressed_len=suffix_start,
                )
            (
                grouped_prefix,
                assignments,
                inverse_permutation,
                segment_ranges,
            ) = self._cluster_prefix(prefix_keys)
            layer_segments = decompose_to_segment_store(
                grouped_prefix,
                self.decompose,
                segment_ranges=segment_ranges,
                **self._decomposition_kwargs(),
            )

        if self.log_timing_stats:
            sync_cuda(keys)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )

        self.cluster_metadata[layer_idx] = {
            "mode": self.kmeans_mode,
            "assignments": assignments,
            "inverse_permutation": inverse_permutation,
        }
        self._store_layer_segments(layer_idx, layer_segments)
        return suffix_start

    def _reconstruct_keys(self, keys, layer_idx, cache_kwargs=None):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        metadata = self.cluster_metadata.get(layer_idx)
        if metadata is None:
            raise ValueError(
                f"No cluster metadata found for layer {layer_idx}."
            )

        empty_suffix = keys[..., :0, :]
        if metadata["mode"] == "per_head":
            batch_size, num_heads, seq_len = metadata[
                "inverse_permutation"
            ].shape
            empty_suffix = empty_suffix.reshape(
                batch_size * num_heads, 0, empty_suffix.size(-1)
            )
            grouped_prefix = reconstruct_segments(
                self.lr_keys[layer_idx], empty_suffix
            )
            prefix_keys = restore_grouped_sequences_order(
                grouped_prefix,
                metadata["inverse_permutation"].reshape(
                    batch_size * num_heads, seq_len
                ),
            ).reshape(batch_size, num_heads, seq_len, -1)
        else:
            grouped_prefix = reconstruct_segments(
                self.lr_keys[layer_idx], empty_suffix
            )
            prefix_keys = restore_grouped_keys_order(
                grouped_prefix, metadata["inverse_permutation"]
            )
        if self.unrope_keys:
            prefix_keys = self.layers[layer_idx]._apply_rope(
                prefix_keys, compressed_len=self.compressed_len
            )
        recon_keys = torch.cat([prefix_keys, keys], dim=-2)

        if self.log_timing_stats:
            sync_cuda(recon_keys)
            update_timing_stats(
                type(self), "reconstruct", time.perf_counter() - start_time
            )
        return recon_keys

    def update_events(self, *args, **kwargs):
        self.prefill = False
        if not self.cluster_metadata:
            return None

        layer_counts = []
        for metadata in self.cluster_metadata.values():
            assignments = metadata["assignments"]
            if assignments.numel() == 0:
                layer_counts.append(0.0)
                continue

            # In per-head mode, assignments has shape [B, H, T], so reshaping
            # to [-1, T] treats each (batch, head) sequence independently.
            counts = []
            flat_assignments = assignments.reshape(
                -1, assignments.size(-1)
            )
            for sequence_assignments in flat_assignments:
                counts.append(torch.unique(sequence_assignments).numel())
            layer_counts.append(sum(counts) / len(counts))
        return sum(layer_counts) / len(layer_counts)

    def _apply_cluster_metadata_batch_op(self, op):
        for metadata in self.cluster_metadata.values():
            metadata["assignments"] = op(metadata["assignments"])
            metadata["inverse_permutation"] = op(
                metadata["inverse_permutation"]
            )

    def _apply_per_head_lr_keys_batch_op(self, op):
        for layer_idx, layer_segments in self.lr_keys.items():
            metadata = self.cluster_metadata.get(layer_idx)
            if metadata is None or metadata["mode"] != "per_head":
                continue

            _, num_heads, _ = metadata["inverse_permutation"].shape
            batched_segments = [
                layer_segments[i : i + num_heads]
                for i in range(0, len(layer_segments), num_heads)
            ]
            updated_batches = op(batched_segments)
            self.lr_keys[layer_idx] = [
                head_segments
                for batch_segments in updated_batches
                for head_segments in batch_segments
            ]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        if self.kmeans_mode == "per_head":
            SingleTensorCache.reorder_cache(self, beam_idx)
            beam_idx_list = beam_idx.tolist()
            self._apply_per_head_lr_keys_batch_op(
                lambda batched_segments: [
                    batched_segments[idx] for idx in beam_idx_list
                ]
            )
        else:
            super().reorder_cache(beam_idx)

        def op(tensor):
            return tensor.index_select(0, beam_idx.to(tensor.device))

        self._apply_cluster_metadata_batch_op(op)

    def batch_repeat_interleave(self, repeats: int):
        if self.kmeans_mode == "per_head":
            SingleTensorCache.batch_repeat_interleave(self, repeats)
            self._apply_per_head_lr_keys_batch_op(
                lambda batched_segments: [
                    batch_segments
                    for batch_segments in batched_segments
                    for _ in range(repeats)
                ]
            )
        else:
            super().batch_repeat_interleave(repeats)
        self._apply_cluster_metadata_batch_op(
            lambda tensor: tensor.repeat_interleave(repeats, dim=0)
        )

    def batch_select_indices(self, indices: torch.Tensor):
        if self.kmeans_mode == "per_head":
            SingleTensorCache.batch_select_indices(self, indices)
            indices_list = indices.tolist()
            self._apply_per_head_lr_keys_batch_op(
                lambda batched_segments: [
                    batched_segments[idx] for idx in indices_list
                ]
            )
        else:
            super().batch_select_indices(indices)

        def op(tensor):
            return tensor[indices.to(tensor.device)]

        self._apply_cluster_metadata_batch_op(op)

    def update(
        self, key_states: torch.Tensor, layer_idx: int, cache_kwargs=None
    ) -> torch.Tensor:
        keys = super().update(key_states, layer_idx, cache_kwargs)
        if self.prefill:
            suffix_start = self._decompose_keys(
                keys,
                keys.size(-2),
                layer_idx,
                cache_kwargs=cache_kwargs,
            )
            self._evict(layer_idx=layer_idx, end_idx=suffix_start)
            return keys

        if self.lr_keys.get(layer_idx, False):
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys
        else:
            raise Exception(
                "Prefill is set to False and no clustered low-rank keys were "
                "found."
            )


class KMeansXKVKeysCache(KMeansMixin, XKVKeysCache):
    """
    xKV-style grouped key cache with KMeans row clustering before SVD.

    Each group of adjacent layers is flattened across heads, concatenated
    across layers, reordered by token cluster, and decomposed segment-wise.
    The grouped left factor is stored once on the last layer in the group,
    while each layer stores only its own segment right factors.
    """

    def __init__(
        self,
        *args,
        n_clusters: int = 8,
        kmeans_cluster_size: float | None = None,
        kmeans_n_iter: int = 8,
        kmeans_init: str = "infllm",
        kmeans_dtype: torch.dtype | str = torch.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_kmeans(
            n_clusters=n_clusters,
            kmeans_cluster_size=kmeans_cluster_size,
            kmeans_n_iter=kmeans_n_iter,
            kmeans_init=kmeans_init,
            kmeans_dtype=kmeans_dtype,
        )

    def _cluster_group_prefix(self, group_prefix):
        batch_size, seq_len, _ = group_prefix.shape
        if seq_len == 0:
            empty_long = torch.empty(
                batch_size, 0, dtype=torch.long, device=group_prefix.device
            )
            return group_prefix, empty_long, [[] for _ in range(batch_size)]

        cluster_count = self._get_cluster_count(seq_len)
        assignments = kmeans_cluster_sequences(
            group_prefix,
            n_clusters=cluster_count,
            n_iter=max(1, self.kmeans_n),
            kmeans_init=self.kmeans_init,
            dtype=self.kmeans_dtype,
        )
        grouped_prefix, _, inverse_permutation = group_sequences_by_cluster(
            group_prefix, assignments
        )
        segment_ranges = build_cluster_segment_ranges(
            assignments,
            n_clusters=cluster_count,
        )
        return grouped_prefix, inverse_permutation, segment_ranges

    def _decompose_keys(self, layer_idx, cache_kwargs=None):
        if self.log_timing_stats:
            sync_cuda(self.layers[layer_idx].tensor)
            start_time = time.perf_counter()

        group_start, group_last_layer = self._get_group_bounds(layer_idx)
        if group_last_layer != layer_idx:
            raise ValueError(
                "KMeans xKV decomposition should only be triggered from the "
                f"last layer of a group, got layer {layer_idx} for group "
                f"ending at layer {group_last_layer}."
            )

        group_layers = tuple(range(group_start, group_last_layer + 1))
        group_tensors = [self.layers[i].tensor for i in group_layers]
        batch_size = group_tensors[-1].size(0)
        suffix_start = self._get_suffix_start(group_tensors[-1])
        self.compressed_len = suffix_start

        if suffix_start == 0:
            shared_segments = [[] for _ in range(batch_size)]
            per_layer_segments = [
                [[] for _ in range(batch_size)] for _ in group_layers
            ]
            inverse_permutation = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=group_tensors[-1].device,
            )
        else:
            prefix_tensors = []
            split_sizes = []
            for group_layer_idx, tensor in zip(group_layers, group_tensors):
                prefix_tensor = tensor[..., :suffix_start, :]
                if self.unrope_keys:
                    prefix_tensor = self.layers[group_layer_idx]._undo_rope(
                        prefix_tensor,
                        cache_kwargs,
                        prefill=self.prefill,
                        compressed_len=suffix_start,
                    )
                prefix_flat = prefix_tensor.transpose(1, 2).reshape(
                    prefix_tensor.size(0),
                    prefix_tensor.size(-2),
                    -1,
                )
                prefix_tensors.append(prefix_flat)
                split_sizes.append(prefix_flat.size(-1))

            group_prefix = torch.cat(prefix_tensors, dim=-1)
            grouped_prefix, inverse_permutation, segment_ranges = (
                self._cluster_group_prefix(group_prefix)
            )
            grouped_segments = decompose_to_segment_store(
                grouped_prefix.unsqueeze(1),
                self.decompose,
                segment_ranges=segment_ranges,
                **self._decomposition_kwargs(),
            )

            shared_segments = [[] for _ in range(batch_size)]
            per_layer_segments = [
                [[] for _ in range(batch_size)] for _ in group_layers
            ]
            for batch_idx, batch_segments in enumerate(grouped_segments):
                for segment in batch_segments:
                    shared_factor, grouped_right = segment["factors"]
                    shared_segments[batch_idx].append(
                        {
                            "range": segment["range"],
                            "factor": shared_factor,
                        }
                    )
                    right_factors = torch.split(
                        grouped_right,
                        split_sizes,
                        dim=-1,
                    )
                    for layer_segments, right_factor in zip(
                        per_layer_segments, right_factors
                    ):
                        layer_segments[batch_idx].append(
                            {
                                "range": segment["range"],
                                "factor": right_factor,
                            }
                        )

        self.shared_a[group_last_layer] = shared_segments
        for group_layer_idx, layer_segments in zip(
            group_layers, per_layer_segments
        ):
            self.lr_keys[group_layer_idx] = layer_segments
            self.group_metadata[group_layer_idx] = {
                "group_last_layer": group_last_layer,
                "compressed_len": suffix_start,
                "inverse_permutation": inverse_permutation,
            }
        self.comp_ratio = self.calc_compression_ratio()

        if self.log_timing_stats:
            sync_cuda(self.layers[layer_idx].tensor)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )
        return suffix_start

    def _reconstruct_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        metadata = self.group_metadata.get(layer_idx)
        if metadata is None:
            raise ValueError(
                f"No clustered xKV metadata found for layer {layer_idx}."
            )

        shared_segments = self.shared_a[metadata["group_last_layer"]]
        layer_segments = self.lr_keys[layer_idx]
        if len(shared_segments) != len(layer_segments):
            raise ValueError(
                "Shared and per-layer clustered segment stores must have the "
                "same batch size."
            )

        paired_segments = []
        for batch_shared, batch_layer in zip(shared_segments, layer_segments):
            if len(batch_shared) != len(batch_layer):
                raise ValueError(
                    "Shared and per-layer clustered segment stores must "
                    "contain the same number of segments per batch."
                )
            batch_segments = []
            for shared_segment, layer_segment in zip(batch_shared, batch_layer):
                if shared_segment["range"] != layer_segment["range"]:
                    raise ValueError(
                        "Shared and per-layer clustered segments must share "
                        "the same token ranges."
                    )
                batch_segments.append(
                    {
                        "range": shared_segment["range"],
                        "factors": (
                            shared_segment["factor"],
                            layer_segment["factor"],
                        ),
                    }
                )
            paired_segments.append(batch_segments)

        flat_dim = keys.size(1) * keys.size(-1)
        empty_suffix = keys.new_empty(keys.size(0), 1, 0, flat_dim)
        prefix_flat = reconstruct_segments(
            paired_segments,
            empty_suffix,
        ).squeeze(1)
        prefix_flat = restore_grouped_sequences_order(
            prefix_flat,
            metadata["inverse_permutation"],
        )

        batch_size, num_heads, _, head_dim = keys.shape
        prefix_keys = prefix_flat.reshape(
            batch_size,
            prefix_flat.size(1),
            num_heads,
            head_dim,
        ).transpose(1, 2)

        if self.unrope_keys:
            prefix_keys = self.layers[layer_idx]._apply_rope(
                prefix_keys,
                compressed_len=metadata["compressed_len"],
            )
        recon_keys = torch.cat([prefix_keys, keys], dim=-2)

        if self.log_timing_stats:
            sync_cuda(recon_keys)
            update_timing_stats(
                type(self), "reconstruct", time.perf_counter() - start_time
            )
        return recon_keys
    

KEY_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "low_rank": LowRankKeysCache,
    "xkv": XKVKeysCache,
    "surprise_lr": SurpriseLRKCache,
    "kmeans_lr": KMeansLRKCache,
    "kmeans_xkv": KMeansXKVKeysCache,
    "turboquant": TurboQuantCache,
}
