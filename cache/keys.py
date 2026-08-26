import math
from typing import Any

import torch

from utils.matrix_decomposition import (
    decompose_grouped_xkv_to_segment_store,
    reconstruct_segments,
)
from utils.turboquant import (
    factor_dtype,
    factor_nbytes,
    factor_shape,
    quantise_factor,
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
    if cache_kwargs is not None and cache_kwargs.get("allow_sparse_kv", False):
        return
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
    def __init__(
        self,
        *args,
        rank_selection: str = "comp_ratio",
        comp_ratio: float = 2.0,
        energy_threshold: float = 0.95,
        unrope_keys: bool = True,
        quantise_a: bool = False,
        quantise_b: bool = False,
        compressor_bits: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rank_selection = rank_selection
        self.r = comp_ratio
        self.e = energy_threshold
        self.unrope_keys = unrope_keys
        self.quantise_a = quantise_a
        self.quantise_b = quantise_b
        self.compressor_bits = compressor_bits

        self.prefill = True
        self.lr_keys = {}
        self.comp_ratio = None
        self.compressed_len = 0

    def update_events(self, *args, **kwargs):
        self.prefill = False

    def _decomposition_kwargs(self):
        return {
            "rank_selection": self.rank_selection,
            "cr": self.r,
            "energy_threshold": self.e,
            "quantise_a": self.quantise_a,
            "quantise_b": self.quantise_b,
            "compressor_bits": self.compressor_bits,
        }

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
        xkv_svd_backend: str = "cholqr",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive.")
        if num_layers is not None and num_layers <= 0:
            raise ValueError("num_layers must be positive when provided.")

        self.layer_group_size = layer_group_size
        self.num_layers = num_layers
        self.xkv_svd_backend = xkv_svd_backend
        self.shared_a = {}
        self.group_metadata = {}
        self.packed_shared_a = {}
        self.packed_lr_keys = {}
        self.fused_lr_keys = {}
        self.fused_chunk_sizes = {}
        self.selective_bytes = {}
        from efficiency import FusedKeyReconstructor

        self.fused_reconstructor = FusedKeyReconstructor()

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
        group_start = (
            layer_idx // self.layer_group_size
        ) * self.layer_group_size
        group_last = group_start + self.layer_group_size - 1
        if self.num_layers is not None:
            group_last = min(group_last, self.num_layers - 1)
        return group_start, group_last

    def _get_decomposition_group(self, layer_idx, cache_name):
        group_start, group_last_layer = self._get_group_bounds(layer_idx)
        if group_last_layer != layer_idx:
            raise ValueError(
                f"{cache_name} decomposition should only be triggered from "
                f"the last layer of a group, got layer {layer_idx} for group "
                f"ending at layer {group_last_layer}."
            )

        group_layers = tuple(range(group_start, group_last_layer + 1))
        group_tensors = [self.layers[i].tensor for i in group_layers]
        assert all(
            t.size(-2) == group_tensors[-1].size(-2) for t in group_tensors
        ), (
            f"All layers in the {cache_name} group must share the same "
            "cached length."
        )
        return group_layers, group_last_layer, group_tensors

    def _get_group_prefix(
        self,
        group_layers,
        group_tensors,
        suffix_start,
        cache_kwargs,
    ):
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

        return torch.cat(prefix_tensors, dim=-1), split_sizes

    def _empty_xkv_segments(self, batch_size, group_layers):
        shared_segments = [[] for _ in range(batch_size)]
        per_layer_segments = [
            [[] for _ in range(batch_size)] for _ in group_layers
        ]
        return shared_segments, per_layer_segments

    def _xkv_decomposition_kwargs(self):
        decompose_kwargs = self._decomposition_kwargs()
        quantise_a = decompose_kwargs.pop("quantise_a")
        quantise_b = decompose_kwargs.pop("quantise_b")
        compressor_bits = decompose_kwargs.pop("compressor_bits")
        decompose_kwargs["svd_backend"] = self.xkv_svd_backend
        return decompose_kwargs, quantise_a, quantise_b, compressor_bits

    def set_selective_overhead(self, layer_idx, nbytes):
        """Record persistent selective-key bytes for rank budgeting."""
        self.selective_bytes[layer_idx] = nbytes

    def _align_fused_rank(self, decompose_kwargs, tensor, group_layers):
        """Fit aligned xKV factors into the total key budget."""
        if decompose_kwargs["rank_selection"] != "comp_ratio":
            return
        from efficiency import adjust_rank

        m, n = tensor.shape[-2:]
        rank = adjust_rank(
            m,
            n,
            decompose_kwargs["cr"],
            sum(
                self.selective_bytes.get(layer_idx, 0)
                for layer_idx in group_layers
            ),
            tensor.size(0),
            tensor.element_size(),
        )
        decompose_kwargs["cr"] = m * n / (rank * (m + n))

    def _split_grouped_xkv_segments(
        self,
        grouped_segments,
        split_sizes,
        group_layers,
        batch_size,
        quantise_b,
        compressor_bits,
    ):
        shared_segments, per_layer_segments = self._empty_xkv_segments(
            batch_size, group_layers
        )
        for batch_idx, batch_segments in enumerate(grouped_segments):
            for segment in batch_segments:
                shared_factor, grouped_right = segment["factors"]
                shared_segments[batch_idx].append(
                    {
                        "range": segment["range"],
                        "factor": shared_factor,
                    }
                )
                right_factors = torch.split(grouped_right, split_sizes, dim=-1)
                for layer_segments, right_factor in zip(
                    per_layer_segments, right_factors
                ):
                    if quantise_b:
                        right_factor = quantise_factor(
                            right_factor,
                            compressor_bits,
                        )
                    layer_segments[batch_idx].append(
                        {
                            "range": segment["range"],
                            "factor": right_factor,
                        }
                    )
        return shared_segments, per_layer_segments

    def _store_xkv_group_segments(
        self,
        group_layers,
        group_last_layer,
        suffix_start,
        shared_segments,
        per_layer_segments,
        extra_metadata=None,
    ):
        self.shared_a[group_last_layer] = shared_segments
        for group_layer_idx, layer_segments in zip(
            group_layers, per_layer_segments
        ):
            self.lr_keys[group_layer_idx] = layer_segments
            metadata = {
                "group_last_layer": group_last_layer,
                "compressed_len": suffix_start,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            self.group_metadata[group_layer_idx] = metadata
        self.comp_ratio = self.calc_compression_ratio()
        if not shared_segments[0]:
            return
        # packing avoids looping through batch elements and per-step stacking during decoding
        self.packed_shared_a[group_last_layer] = torch.stack(
            [segments[0]["factor"].squeeze(0) for segments in shared_segments]
        )
        for batch_idx, segments in enumerate(shared_segments):
            segments[0]["factor"] = self.packed_shared_a[group_last_layer][
                batch_idx
            ].unsqueeze(0)
        for group_layer_idx, layer_segments in zip(
            group_layers, per_layer_segments
        ):
            self.packed_lr_keys[group_layer_idx] = torch.stack(
                [
                    segments[0]["factor"].squeeze(0)
                    for segments in layer_segments
                ]
            )
            for batch_idx, segments in enumerate(layer_segments):
                segments[0]["factor"] = self.packed_lr_keys[group_layer_idx][
                    batch_idx
                ].unsqueeze(0)

    def calc_compression_ratio(self):
        layers_by_group: dict[int, list[int]] = {}
        for layer_idx, metadata in self.group_metadata.items():
            layers_by_group.setdefault(metadata["group_last_layer"], []).append(
                layer_idx
            )

        crs = 0.0
        num_segments = 0
        for group_last, group_layers in layers_by_group.items():
            for batch_idx, batch_shared in enumerate(self.shared_a[group_last]):
                for seg_idx, shared_segment in enumerate(batch_shared):
                    A = shared_segment["factor"]
                    a_shape = factor_shape(A)
                    leading = math.prod(a_shape[:-2])
                    m = a_shape[-2]
                    n = sum(
                        factor_shape(
                            self.lr_keys[l][batch_idx][seg_idx]["factor"]
                        )[-1]
                        for l in group_layers
                    )
                    original = (
                        leading
                        * m
                        * n
                        * torch.empty(
                            (),
                            dtype=factor_dtype(A),
                        ).element_size()
                    )
                    compressed = factor_nbytes(A) + sum(
                        factor_nbytes(
                            self.lr_keys[l][batch_idx][seg_idx]["factor"]
                        )
                        for l in group_layers
                    )
                    crs += original / compressed
                    num_segments += 1
        return crs / num_segments if num_segments > 0 else 0.0

    def _decompose_keys(self, layer_idx, cache_kwargs=None):
        group_layers, group_last_layer, group_tensors = (
            self._get_decomposition_group(layer_idx, "xKV")
        )
        batch_size = group_tensors[-1].size(0)

        suffix_start = group_tensors[-1].size(-2)
        self.compressed_len = suffix_start
        if suffix_start == 0:
            shared_segments, per_layer_segments = self._empty_xkv_segments(
                batch_size, group_layers
            )
        else:
            group_prefix, split_sizes = self._get_group_prefix(
                group_layers,
                group_tensors,
                suffix_start,
                cache_kwargs,
            )
            segment_ranges = [[(0, suffix_start)] for _ in range(batch_size)]
            decompose_kwargs, quantise_a, quantise_b, compressor_bits = (
                self._xkv_decomposition_kwargs()
            )
            self._align_fused_rank(decompose_kwargs, group_prefix, group_layers)
            grouped_segments = decompose_grouped_xkv_to_segment_store(
                group_prefix.unsqueeze(1),
                segment_ranges=segment_ranges,
                quantise_a=quantise_a,
                compressor_bits=compressor_bits,
                **decompose_kwargs,
            )
            shared_segments, per_layer_segments = (
                self._split_grouped_xkv_segments(
                    grouped_segments,
                    split_sizes,
                    group_layers,
                    batch_size,
                    quantise_b,
                    compressor_bits,
                )
            )

        self._store_xkv_group_segments(
            group_layers,
            group_last_layer,
            suffix_start,
            shared_segments,
            per_layer_segments,
        )

        return suffix_start

    def append_decode(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> None:
        SingleTensorCache.update(self, key_states, layer_idx, cache_kwargs)

    def supports_selective_retrieval(self, layer_idx: int) -> bool:
        return layer_idx in self.packed_lr_keys

    def _fused_right_factor(self, layer_idx, num_heads):
        """Reshape packed_lr_keys into expected shape for kernel"""
        right = self.fused_lr_keys.get(layer_idx)
        if right is None:
            packed = self.packed_lr_keys[layer_idx]
            batch_size, rank, flat_dim = packed.shape
            head_dim = flat_dim // num_heads
            right = (
                packed.reshape(batch_size, rank, num_heads, head_dim)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            self.fused_lr_keys[layer_idx] = right
        return right

    def prepare_selected_chunks(self, layer_idx, chunks, chunk_size):
        metadata = self.group_metadata[layer_idx]
        shared = self.packed_shared_a[metadata["group_last_layer"]]
        if (
            not self.unrope_keys
            or "inverse_permutation" in metadata
            or not self.fused_reconstructor.available()
        ):
            return None
        right = self._fused_right_factor(layer_idx, chunks.size(1))
        if not self.fused_reconstructor.supports(
            shared, right, chunks, chunk_size
        ):
            return None
        self.fused_chunk_sizes[layer_idx] = chunk_size
        return self.fused_reconstructor.reorder(layer_idx, chunks, chunk_size)

    @property
    def selective_reconstruction_nbytes(self):
        return self.fused_reconstructor.nbytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.fused_lr_keys.values()
        )

    @torch.no_grad()
    def retrieve_selected(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self.group_metadata[layer_idx]
        group_last = metadata["group_last_layer"]

        shared = self.packed_shared_a[group_last]
        right = self.packed_lr_keys[layer_idx]
        suffix = self.layers[layer_idx].tensor
        prefix_len = positions.size(-1) - suffix.size(-2)
        prefix_positions = positions[..., :prefix_len]
        batch_size, num_heads, selected_len = prefix_positions.shape
        rank = shared.size(-1)
        head_dim = right.size(-1) // num_heads
        fused_right = self.fused_lr_keys.get(layer_idx)
        if (
            fused_right is not None
            and layer_idx in self.fused_reconstructor.states
        ):
            cos, sin = self.layers[layer_idx]._resolve_rope_cos_sin(
                metadata["compressed_len"],
                head_dim,
                shared.device,
                shared.dtype,
            )
            keys = self.fused_reconstructor.reconstruct(
                layer_idx,
                shared,
                fused_right,
                cos,
                sin,
                self.fused_chunk_sizes[layer_idx],
            )
        else:
            selected_a = (
                shared[:, None]
                .expand(-1, num_heads, -1, -1)
                .gather(
                    2,
                    prefix_positions[..., None].expand(
                        -1, -1, selected_len, rank
                    ),
                )
            )
            right = right.reshape(
                batch_size, rank, num_heads, head_dim
            ).transpose(1, 2)
            keys = torch.matmul(selected_a, right)
            if self.unrope_keys:
                keys = self.layers[layer_idx]._rope_selected(
                    keys,
                    prefix_positions,
                    metadata["compressed_len"],
                    inverse=False,
                )

        if suffix.size(-2):
            suffix_positions = (
                positions[..., prefix_len:] - metadata["compressed_len"]
            )
            suffix_keys = suffix.gather(
                2,
                suffix_positions[..., None].expand(-1, -1, -1, suffix.size(-1)),
            )
            keys = torch.cat([keys, suffix_keys], dim=2)
        return keys

    def _reconstruct_keys(self, keys, layer_idx):
        metadata = self.group_metadata.get(layer_idx)
        if metadata is None:
            raise ValueError(f"No xKV metadata found for layer {layer_idx}.")

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

            suffix_start = self._decompose_keys(
                layer_idx, cache_kwargs=cache_kwargs
            )
            for group_layer_idx in range(group_start, group_last_layer + 1):
                self._evict(layer_idx=group_layer_idx, end_idx=suffix_start)
            return keys

        if layer_idx in self.lr_keys:
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys

        raise Exception(
            "Prefill is set to False and no xKV factors were found for "
            f"layer {layer_idx}. If this layer is part of a terminal partial "
            "group, pass num_layers through later plumbing so the group "
            "boundary is known."
        )


KEY_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "xkv": XKVKeysCache,
    "turboquant": TurboQuantCache,
}
