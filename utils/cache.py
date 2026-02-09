import torch 

from transformers.cache_utils import (
    DynamicCache as DC, Any, Iterable, PreTrainedConfig
)
# from transformers.models.llama.modeling_llama import LlamaAttention

from .matrix_decomposition import truncated_svd
from .segmentation import find_thresholds


class DynamicCache(DC):
    """This class simply intercepts kwargs for a more flexible base."""
    def __init__(
        self, 
        *args,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        **kwargs,
    ):
        super().__init__(
            ddp_cache_data,
            config,
            offloading,
            offload_only_non_sliding,
        )


class SVDCache(DynamicCache):
    def __init__(
        self, 
        *args,
        niter: int = 3,
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        rank_selection: str = 'comp_ratio',  # comp_ratio, energy
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.n = niter
        self.r = comp_ratio
        self.e = energy_threshold
        self.rank_selection = rank_selection
        self.prefill = True

        self.svd_keys = {}

    def calc_compression_ratio(self):
        if self.rank_selection == 'comp_ratio':
            return self.r
        else:
            crs = 0
            for US, V in self.svd_keys.values():
                m, k = US.shape[-2:]
                n = V.size(-1)
                crs += (m * n) / (k * (m + n))
            return crs / len(self.svd_keys)

    def update_events(self, *args, **kwargs):
        self.prefill = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )
        if self.prefill:
            U, S, V = truncated_svd(
                keys, self.rank_selection,
                cr=self.r, energy_threshold=self.e,
                niter=self.n
            )
            US = U * S.unsqueeze(-2)
            self.svd_keys[layer_idx] = (US, V)
            self.comp_ratio = self.calc_compression_ratio()
            return keys, values  # return full KVs for accurate optimal generation of KVs in next layers
        elif self.svd_keys.get(layer_idx, False):
            US, V = self.svd_keys[layer_idx]
            reconstructed_keys = US @ V
            if US.size(-2) < keys.size(-2):
                reconstructed_keys = torch.cat(
                    [reconstructed_keys, keys[..., US.size(-2):, :]],
                    dim=-2,
                )
            return reconstructed_keys, values
        else:
            raise Exception(
                "Prefill is set to False and no svd keys were found."
            )


class SurpriseSVDCache(DynamicCache):
    def __init__(
        self, 
        *args,
        niter: int = 3,
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        rank_selection: str = 'comp_ratio',  # comp_ratio, energy
        gamma: float = 3.0,
        min_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.n = niter
        self.r = comp_ratio
        self.e = energy_threshold
        self.rank_selection = rank_selection
        self.gamma = gamma
        self.min_size = min_size
        self.prefill = True

        self.svd_keys = {}

    def calc_compression_ratio(self):
        if self.rank_selection == 'comp_ratio':
            return self.r
        else:
            crs = 0
            num_events = 0
            for svd_keys in self.svd_keys.values():
                for b in range(len(svd_keys)):
                    for (US, V) in svd_keys[b]:
                        m, k = US.shape[-2:]
                        n = V.size(-1)
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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )
        if self.prefill:
            return keys, values
        
        elif self.svd_keys.get(layer_idx, False):
            recon_keys = []
            svd_keys = self.svd_keys[layer_idx]
            for b in range(len(svd_keys)):
                batch_recon_keys = []
                for (US, V) in svd_keys[b]:
                    batch_recon_keys.append(US @ V)
                ed = self.events[b][-1]
                if ed < keys.size(-2):
                    batch_recon_keys.append(
                        keys[b, ..., ed:, :]
                    )
                recon_keys.append(
                    torch.cat(batch_recon_keys, dim=-2)
                )
        else:
            recon_keys = []
            svd_keys = []
            for b in range(len(self.events)):
                batch_recon_keys = []
                batch_svd_keys = []
                for i, st in enumerate(self.events[b][:-1]):
                    ed = self.events[b][i+1]
                    keys_subset = keys[b, ..., st:ed, :]
                    U, S, V = truncated_svd(
                        keys_subset, self.rank_selection,
                        cr=self.r, energy_threshold=self.e,
                        niter=self.n
                    )
                    batch_svd_keys.append((U * S.unsqueeze(-2), V))
                    batch_recon_keys.append(
                        (U * S.unsqueeze(-2)) @ V
                    )
                if ed < keys.size(-2):
                    batch_recon_keys.append(
                        keys[b, ..., ed:, :]
                    )
                recon_keys.append(
                    torch.cat(batch_recon_keys, dim=-2)
                )
                svd_keys.append(batch_svd_keys)
            self.svd_keys[layer_idx] = svd_keys
            self.comp_ratio = self.calc_compression_ratio()

        recon_keys = torch.stack(recon_keys, dim=0)
        return recon_keys, values
