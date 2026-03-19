import time

import torch
import torch.nn.functional as F

from transformers.cache_utils import Any

from utils.matrix_decomposition import (
    DECOMP_METHODS,
    calc_segment_store_compression_ratio,
    decompose_to_segment_store,
    reconstruct_segments,
)
from utils.segmentation import find_thresholds
from .cache import SingleTensorCache


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


def init_timing_stats():
    return {
        "decompose_time": 0.0,
        "reconstruct_time": 0.0,
        "decompose_relative": 0.0,
        "reconstruct_relative": 0.0,
        "decompose_calls": 0,
        "reconstruct_calls": 0,
        "most_time_consuming": None,
    }


def sync_cuda(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(device=tensor.device)


def update_timing_stats(cache_cls, operation, elapsed_time):
    cache_cls.timing_stats[f"{operation}_time"] += elapsed_time
    cache_cls.timing_stats[f"{operation}_calls"] += 1
    total_time = (
        cache_cls.timing_stats["decompose_time"]
        + cache_cls.timing_stats["reconstruct_time"]
    )
    if total_time > 0:
        cache_cls.timing_stats["decompose_relative"] = (
            cache_cls.timing_stats["decompose_time"] / total_time
        )
        cache_cls.timing_stats["reconstruct_relative"] = (
            cache_cls.timing_stats["reconstruct_time"] / total_time
        )
        cache_cls.timing_stats["most_time_consuming"] = max(
            ("decompose", "reconstruct"),
            key=lambda op: cache_cls.timing_stats[f"{op}_time"],
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
        n_iter: int = 3,
        lr: float = 1e-2,
        unrope_keys: bool = False,
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
        self.n = n_iter
        self.lr = lr
        self.unrope_keys = unrope_keys

        self.prefill = True
        self.lr_keys = {}
        self.comp_ratio = None
        self.compressed_len = 0
        self.log_timing_stats = log_timing_stats
        type(self).timing_stats = (
            init_timing_stats() if self.log_timing_stats else None
        )

    def calc_compression_ratio(self):
        default_ratio = self.r if self.rank_selection == "comp_ratio" else None
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
            "n_iter": self.n,
            "lr": self.lr,
        }

    def _store_layer_segments(self, layer_idx, layer_segments):
        self.lr_keys[layer_idx] = layer_segments
        self.comp_ratio = self.calc_compression_ratio()

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
                decomp_keys, cache_kwargs,
                prefill=self.prefill, compressed_len=suffix_start,
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

    def _reconstruct_keys(self, keys, layer_idx, cache_kwargs=None):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()

        recon_keys = reconstruct_segments(self.lr_keys[layer_idx], keys)
        if self.unrope_keys:
            recon_keys = self.layers[layer_idx]._apply_rope(
                recon_keys, compressed_len=self.compressed_len,
            )
        if self.log_timing_stats:
            sync_cuda(recon_keys)
            update_timing_stats(
                type(self), "reconstruct", time.perf_counter() - start_time
            )
        return recon_keys


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
            recon_keys = self._reconstruct_keys(
                keys, layer_idx, cache_kwargs=cache_kwargs
            )
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys
        else:
            raise Exception(
                "Prefill is set to False and no low_rank keys were found."
            )


class SurpriseLRKCache(DecomposedKeysCache):
    def __init__(
        self,
        *args,
        decomposition_method: str,
        log_timing_stats: bool = False,
        rank_selection: str = "comp_ratio",
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        n_iter: int = 3,
        lr: float = 1e-2,
        gamma: float = 3.0,
        min_size: int = 8,
        **kwargs,
    ):
        super().__init__(
            *args,
            decomposition_method=decomposition_method,
            log_timing_stats=log_timing_stats,
            rank_selection=rank_selection,
            comp_ratio=comp_ratio,
            energy_threshold=energy_threshold,
            n_iter=n_iter,
            lr=lr,
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

        recon_keys = self._reconstruct_keys(
            keys, layer_idx, cache_kwargs=cache_kwargs
        )
        check_recon_length(recon_keys, cache_kwargs)
        return recon_keys


KEY_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "low_rank": LowRankKeysCache,
    "surprise_lr": SurpriseLRKCache,
}
