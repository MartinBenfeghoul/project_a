import torch

from transformers.cache_utils import Any

# from transformers.models.llama.modeling_llama import LlamaAttention

from .matrix_decomposition import DECOMP_METHODS
from .segmentation import find_thresholds
from .cache import SingleTensorCache


class LowRankKeysCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        decomposition_method: str = None,
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
        A, B = self.decompose(
            keys,
            self.rank_selection,
            cr=self.r,
            energy_threshold=self.e,
            n_iter=self.n,
            lr=self.lr,
        )
        self.lr_keys[layer_idx] = (A, B)
        self.comp_ratio = self.calc_compression_ratio()

    def _reconstruct_keys(self, keys, layer_idx):
        A, B = self.lr_keys[layer_idx]
        recon_keys = A @ B
        if A.size(-2) < keys.size(-2):  # TODO: update this logic to handle evicted tokens - ie. keys is now only the new tokens since prefill
            recon_keys = torch.cat(
                [recon_keys, keys[..., A.size(-2) :, :]],
                dim=-2,
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
            self.clear(layer_idx=layer_idx) 
            return keys
        elif self.lr_keys.get(layer_idx, False):
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            cache_position = cache_kwargs.get("cache_position", None) 
        if cache_position is not None:
            assert recon_keys.size(-2) == cache_position.size(-1), f"Reconstructed keys have seq_len {recon_keys.size(-2)} but cache_position has size {cache_position.size(-1)}"
            return recon_keys
        else:
            raise Exception(
                "Prefill is set to False and no low_rank keys were found."
            )


class SurpriseLRKCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        decomposition_method: str = None,
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
        prob = torch.softmax(logits, dim=-1)
        surprise = -torch.log(
            torch.gather(prob, dim=-1, index=labels.unsqueeze(-1))
        ).squeeze(-1)

        self.events = []
        for b in range(surprise.size(0)):
            self.events.append(
                find_thresholds(
                    surprise[b],
                    threshold_param=self.gamma,
                    min_size=self.min_size,
                )[0]
            )
        self.prefill = False

    def _decompose_keys(self, keys, layer_idx):
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
        self.lr_keys[layer_idx] = lr_keys
        self.comp_ratio = self.calc_compression_ratio()

    def _reconstruct_keys(self, keys, layer_idx):
        lr_keys = self.lr_keys[layer_idx]
        recon_keys = []
        for b in range(len(lr_keys)):
            batch_recon_keys = []
            for A, B in lr_keys[b]:
                batch_recon_keys.append(A @ B)
            ed = self.events[b][-1]
            # TODO: move the below outside the loop as all batches will have the same ed
            if ed < keys.size(-2):    # TODO: update this logic to handle evicted tokens - ie. keys is now only the new tokens since prefill
                batch_recon_keys.append(keys[b, ..., ed:, :])
            recon_keys.append(torch.cat(batch_recon_keys, dim=-2))
        return torch.stack(recon_keys, dim=0)

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
            self._decompose_keys(keys, layer_idx)
            self.clear(layer_idx=layer_idx, end_idx=self.events[0][-1])
        recon_keys = self._reconstruct_keys(keys, layer_idx)
        cache_position = cache_kwargs.get("cache_position", None) 
        if cache_position is not None:
            assert recon_keys.size(-2) == cache_position.size(-1), f"Reconstructed keys have seq_len {recon_keys.size(-2)} but cache_position has size {cache_position.size(-1)}"
        return recon_keys


KEY_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "low_rank": LowRankKeysCache,
    "surprise_lr": SurpriseLRKCache,
}
