import time

import torch
import torch.nn.functional as F

from transformers.cache_utils import Any

# from transformers.models.llama.modeling_llama import LlamaAttention

from utils.matrix_decomposition import DECOMP_METHODS
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
    if exp_seq_len is not None:
        assert (
            recon_keys.size(-2) == exp_seq_len
        ), f"Reconstructed keys have seq_len {recon_keys.size(-2)} but cache_position expects {exp_seq_len}."


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


class LowRankKeysCache(SingleTensorCache):
    timing_stats = None

    def __init__(
        self,
        *args,
        decomposition_method: str = None,
        log_timing_stats: bool = False,
        # decomposition method-agnostic args
        rank_selection: str = "comp_ratio",  # comp_ratio, energy
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        n_iter: int = 3,
        # LoRA-specific args
        lr: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert (
            decomposition_method in DECOMP_METHODS.keys()
        ), f"Decomposition method {decomposition_method} not found in available methods: {DECOMP_METHODS.keys()}"
        self.decomposition_method = decomposition_method
        self.decompose = DECOMP_METHODS[decomposition_method]

        self.rank_selection = rank_selection
        self.r = comp_ratio
        self.e = energy_threshold
        self.n = n_iter

        self.lr = lr

        self.prefill = True
        self.lr_keys = {}
        self.log_timing_stats = log_timing_stats
        type(self).timing_stats = (
            init_timing_stats() if self.log_timing_stats else None
        )

    def calc_compression_ratio(self):
        if self.rank_selection == "comp_ratio":
            return self.r
        else:
            crs = 0
            for A, B in self.lr_keys.values():
                m, k = A.shape[-2:]
                n = B.size(-1)
                crs += (m * n) / (k * (m + n))
            return crs / len(self.lr_keys)

    def update_events(self, *args, **kwargs):
        self.prefill = False

    def _decompose_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()
        A, B = self.decompose(
            keys,
            self.rank_selection,
            cr=self.r,
            energy_threshold=self.e,
            n_iter=self.n,
            lr=self.lr,
        )
        if self.log_timing_stats:
            sync_cuda(keys)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )
        self.lr_keys[layer_idx] = (A, B)
        self.comp_ratio = self.calc_compression_ratio()

    def _reconstruct_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()
        A, B = self.lr_keys[layer_idx]
        recon_keys = A @ B
        if keys.size(-2) > 0:
            recon_keys = torch.cat([recon_keys, keys], dim=-2)
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
        keys = super().update(
            key_states,
            layer_idx,
            cache_kwargs,
        )
        if self.prefill:
            self._decompose_keys(keys, layer_idx)
            self._evict(layer_idx=layer_idx)
            return keys
        elif self.lr_keys.get(layer_idx, False):
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(
                recon_keys, cache_kwargs
            )  # TODO: remove later - keep during development for safety
            return recon_keys
        else:
            raise Exception(
                "Prefill is set to False and no low_rank keys were found."
            )


class SurpriseLRKCache(SingleTensorCache):
    timing_stats = None

    def __init__(
        self,
        *args,
        decomposition_method: str,
        log_timing_stats: bool = False,
        # decomposition method-agnostic args
        rank_selection: str = "comp_ratio",  # comp_ratio, energy
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        n_iter: int = 3,
        # LoRA-specific args
        lr: float = 1e-2,
        # segmentation args
        gamma: float = 3.0,
        min_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert (
            decomposition_method in DECOMP_METHODS.keys()
        ), f"Decomposition method {decomposition_method} not found in available methods: {DECOMP_METHODS.keys()}"
        self.decomposition_method = decomposition_method
        self.decompose = DECOMP_METHODS[decomposition_method]

        self.rank_selection = rank_selection
        self.r = comp_ratio
        self.e = energy_threshold
        self.n = n_iter

        self.lr = lr

        self.gamma = gamma
        self.min_size = min_size

        self.prefill = True
        self.lr_keys = {}
        self.log_timing_stats = log_timing_stats
        type(self).timing_stats = (
            init_timing_stats() if self.log_timing_stats else None
        )

    def calc_compression_ratio(self):
        if self.rank_selection == "comp_ratio":
            return self.r
        else:
            crs = 0
            num_events = 0
            for lr_keys in self.lr_keys.values():
                for b in range(len(lr_keys)):
                    for A, B in lr_keys[b]:
                        m, k = A.shape[-2:]
                        n = B.size(-1)
                        crs += (m * n) / (k * (m + n))
                        num_events += 1
            return crs / num_events

    def update_events(self, logits, labels):
        B, T, V = logits.shape
        surprise = (
            F.cross_entropy(  # to avoid materialising (B, T, V) probs tensor
                logits.reshape(B * T, V),
                labels.reshape(B * T),
                reduction="none",
            ).reshape(B, T)
        )

        self.events = []
        for b in range(B):
            events = find_thresholds(
                surprise[b],
                threshold_param=self.gamma,
                min_size=self.min_size,
            )[0]
            # overwrite the final position to include the last token in prefill
            # as there is no suprise for this position it won't be included
            events[-1] += 1
            self.events.append(events)
        self.prefill = False

    def _decompose_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()
        lr_keys = []
        for b in range(len(self.events)):
            batch_svd_keys = []
            for i, st in enumerate(self.events[b][:-1]):
                ed = self.events[b][i + 1]
                keys_subset = keys[b, ..., st:ed, :]
                A, B = self.decompose(
                    keys_subset,
                    self.rank_selection,
                    cr=self.r,
                    energy_threshold=self.e,
                    n_iter=self.n,
                    lr=self.lr,
                )
                batch_svd_keys.append((A, B))
            lr_keys.append(batch_svd_keys)
        if self.log_timing_stats:
            sync_cuda(keys)
            update_timing_stats(
                type(self), "decompose", time.perf_counter() - start_time
            )
        self.lr_keys[layer_idx] = lr_keys
        self.comp_ratio = self.calc_compression_ratio()
        return keys[..., ed:, :]

    def _reconstruct_keys(self, keys, layer_idx):
        if self.log_timing_stats:
            sync_cuda(keys)
            start_time = time.perf_counter()
        lr_keys = self.lr_keys[layer_idx]
        recon_keys = []
        # TODO: Teresa suggests reconstructing events of the same size in batches
        # for more efficiency - look into this w.r.t frequency of events with
        # the same size + efficiency gains when considering re-indexing too
        for b in range(len(lr_keys)):
            batch_recon_keys = []
            for A, B in lr_keys[b]:
                batch_recon_keys.append(A @ B)
            if keys.size(-2) > 0:
                batch_recon_keys.append(keys[b])
            recon_keys.append(torch.cat(batch_recon_keys, dim=-2))
        recon_keys = torch.stack(recon_keys, dim=0)
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
        keys = super().update(
            key_states,
            layer_idx,
            cache_kwargs,
        )
        if self.prefill:
            return keys
        elif not self.lr_keys.get(layer_idx, False):
            keys = self._decompose_keys(keys, layer_idx)
            self._evict(layer_idx=layer_idx, end_idx=self.events[0][-1])
        recon_keys = self._reconstruct_keys(keys, layer_idx)
        check_recon_length(
            recon_keys, cache_kwargs
        )  # TODO: remove later - keep during development for safety
        return recon_keys


KEY_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "low_rank": LowRankKeysCache,
    "surprise_lr": SurpriseLRKCache,
}
