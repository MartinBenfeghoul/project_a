import torch 

from transformers.cache_utils import (
    DynamicCache as DC, Any, Iterable, PreTrainedConfig
)
# from transformers.models.llama.modeling_llama import LlamaAttention

from .matrix_decomposition import truncated_svd
from .segmentation import find_thresholds

def find_rank_wrt_cr(r, m, n):
    """Find the rank k to use for low-rank approximation of a (m x n) matrix 
        such that the compression ratio is ~r.
    """
    k = m * n / (r * (m + n))
    return int(round(k))


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.n = niter
        self.r = comp_ratio
        self.prefill = True

    @property
    def compression_ratio(self):
        U, _, V = self.svd_keys
        m, k = U.shape[-2:]
        n = V.size(-1)
        return (m * n) / (k * (m + n))

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
            return keys, values
        k = find_rank_wrt_cr(self.r, keys.size(-2), keys.size(-1))

        self.svd_keys = truncated_svd(keys, k, niter=self.n)

        U, S, V = self.svd_keys
        reconstructed_keys = (U * S.unsqueeze(-2)) @ V
        return reconstructed_keys, values


class SurpriseSVDCache(SVDCache):
    def __init__(
        self, 
        *args,
        gamma: float = 3.0,
        min_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.gamma = gamma
        self.min_size = min_size

    @property
    def compression_ratio(self):
        return self.r  # TODO: implement wrt events

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
        elif hasattr(self, 'svd_keys'):
            raise NotImplementedError
        else:
            recon_keys = []
            for b in range(len(self.events)):
                batch_recon_keys = []
                for i, st in enumerate(self.events[b][:-1]):
                    ed = self.events[b][i+1]
                    keys_subset = keys[b, ..., st:ed, :]
                    k = find_rank_wrt_cr(
                        self.r, keys_subset.size(-2), keys_subset.size(-1)
                    )
                    U, S, V = truncated_svd(keys_subset, k, niter=self.n)
                    batch_recon_keys.append(
                        (U * S.unsqueeze(-2)) @ V
                    )
                if ed < keys.size(-2):
                    # if layer_idx == 0 and b == 0:
                    #     print("Appending new keys to recon")
                    batch_recon_keys.append(
                        keys[b, ..., ed:, :]
                    )
                    recon_keys.append(
                        torch.cat(batch_recon_keys, dim=-2)
                    )
            recon_keys = torch.stack(recon_keys, dim=0)
            return recon_keys, values