import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from utils.matrix_decomposition import (
    Factor,
    FactorPairSegment,
    FactorSegment,
    SVDDecompositionConfig,
    decompose_grouped_xkv_to_segment_store,
    reconstruct_segments,
)
from utils.turboquant import (
    dequantise_factor,
    factor_dtype,
    factor_nbytes,
    factor_shape,
    gather_factor_rows,
    is_quantised_factor,
    pack_factors,
    quantise_factor,
)
from .base import SharedRopeCache, SingleTensorCache


@dataclass
class XKVLayerState:
    selective_overhead_bytes: int = 0
    group_last_layer: int | None = None
    compressed_len: int = 0
    segments: list[list[FactorSegment]] | None = None
    packed_right: Factor | None = None
    fused_right: torch.Tensor | None = None
    fused_chunk_size: int | None = None
    inverse_permutation: torch.Tensor | None = None

    @property
    def is_compressed(self) -> bool:
        return (
            self.group_last_layer is not None
            and self.segments is not None
            and self.packed_right is not None
        )


@dataclass
class XKVGroupState:
    layer_indices: tuple[int, ...]
    compressed_len: int
    shared_segments: list[list[FactorSegment]]
    packed_shared: Factor | None = None


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
        ddp_cache_data=None,
        *,
        comp_ratio: float = 2.0,
        quantise_a: bool = False,
        quantise_b: bool = False,
        compressor_bits: int = 4,
        rope_cache: SharedRopeCache | None = None,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            rope_cache=rope_cache,
        )
        self.unrope_keys = True
        self.decomposition = SVDDecompositionConfig(
            compression_ratio=comp_ratio,
            quantise_a=quantise_a,
            quantise_b=quantise_b,
            compressor_bits=compressor_bits,
        )

        self.prefill = True
        self.comp_ratio = None

    def update_events(self, *args, **kwargs):
        self.prefill = False


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
        ddp_cache_data=None,
        *,
        layer_group_size: int = 2,
        num_layers: int | None = None,
        xkv_svd_backend: str = "cholqr",
        comp_ratio: float = 2.0,
        quantise_a: bool = False,
        quantise_b: bool = False,
        compressor_bits: int = 4,
        rope_cache: SharedRopeCache | None = None,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            comp_ratio=comp_ratio,
            quantise_a=quantise_a,
            quantise_b=quantise_b,
            compressor_bits=compressor_bits,
            rope_cache=rope_cache,
        )
        if layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive.")
        if num_layers is not None and num_layers <= 0:
            raise ValueError("num_layers must be positive when provided.")

        self.layer_group_size = layer_group_size
        self.num_layers = num_layers
        self.decomposition = replace(
            self.decomposition, svd_backend=xkv_svd_backend
        )
        self.layer_states: dict[int, XKVLayerState] = {}
        self.group_states: dict[int, XKVGroupState] = {}
        from efficiency import FusedKeyReconstructor

        self.fused_reconstructor = FusedKeyReconstructor()

    def _layer_state(self, layer_idx: int) -> XKVLayerState:
        return self.layer_states.setdefault(layer_idx, XKVLayerState())

    def _compressed_layer_state(self, layer_idx: int) -> XKVLayerState:
        state = self.layer_states.get(layer_idx)
        if state is None or not state.is_compressed:
            raise ValueError(f"No xKV state found for layer {layer_idx}.")
        return state

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

    def set_selective_overhead(self, layer_idx, nbytes):
        """Record persistent selective-key bytes for rank budgeting."""
        self._layer_state(layer_idx).selective_overhead_bytes = nbytes

    def _group_decomposition_config(
        self,
        tensor: torch.Tensor,
        group_layers,
    ) -> SVDDecompositionConfig:
        """Fit aligned xKV factors for one group into the total key budget."""
        from efficiency import adjust_rank

        m, n = tensor.shape[-2:]
        rank = adjust_rank(
            m,
            n,
            self.decomposition.compression_ratio,
            sum(
                self._layer_state(layer_idx).selective_overhead_bytes
                for layer_idx in group_layers
            ),
            tensor.size(0),
            tensor.element_size(),
        )
        return replace(
            self.decomposition,
            compression_ratio=m * n / (rank * (m + n)),
            quantise_b=False,
        )

    def _split_grouped_xkv_segments(
        self,
        grouped_segments: list[list[FactorPairSegment]],
        split_sizes,
        group_layers,
        batch_size,
    ):
        shared_segments, per_layer_segments = self._empty_xkv_segments(
            batch_size, group_layers
        )
        for batch_idx, batch_segments in enumerate(grouped_segments):
            for segment in batch_segments:
                shared_factor, grouped_right = segment.factors
                shared_segments[batch_idx].append(
                    FactorSegment(
                        token_range=segment.token_range,
                        factor=shared_factor,
                    )
                )
                right_factors = torch.split(grouped_right, split_sizes, dim=-1)
                for layer_segments, right_factor in zip(
                    per_layer_segments, right_factors
                ):
                    if self.decomposition.quantise_b:
                        right_factor = quantise_factor(
                            right_factor,
                            self.decomposition.compressor_bits,
                        )
                    layer_segments[batch_idx].append(
                        FactorSegment(
                            token_range=segment.token_range,
                            factor=right_factor,
                        )
                    )
        return shared_segments, per_layer_segments

    def _store_xkv_group_segments(
        self,
        group_layers,
        group_last_layer,
        suffix_start,
        shared_segments,
        per_layer_segments,
    ):
        group_state = XKVGroupState(
            layer_indices=tuple(group_layers),
            compressed_len=suffix_start,
            shared_segments=shared_segments,
        )
        self.group_states[group_last_layer] = group_state
        for group_layer_idx, layer_segments in zip(
            group_layers,
            per_layer_segments,
        ):
            state = self._layer_state(group_layer_idx)
            state.group_last_layer = group_last_layer
            state.compressed_len = suffix_start
            state.segments = layer_segments
        if not shared_segments[0]:
            self.comp_ratio = self.calc_compression_ratio()
            return
        # Packing avoids per-batch stacking during every decode step.
        group_state.packed_shared, shared_views = pack_factors(
            [segments[0].factor for segments in shared_segments]
        )
        for segments, view in zip(shared_segments, shared_views):
            segments[0].factor = view
        for group_layer_idx, layer_segments in zip(
            group_layers, per_layer_segments
        ):
            state = self._layer_state(group_layer_idx)
            state.packed_right, right_views = pack_factors(
                [segments[0].factor for segments in layer_segments]
            )
            for segments, view in zip(layer_segments, right_views):
                segments[0].factor = view
        self.comp_ratio = self.calc_compression_ratio()

    def calc_compression_ratio(self):
        crs = 0.0
        num_segments = 0
        for group_state in self.group_states.values():
            for batch_idx, batch_shared in enumerate(
                group_state.shared_segments
            ):
                for seg_idx, shared_segment in enumerate(batch_shared):
                    A = shared_segment.factor
                    a_shape = factor_shape(A)
                    leading = math.prod(a_shape[:-2])
                    m = a_shape[-2]
                    n = sum(
                        factor_shape(
                            self._compressed_layer_state(layer_idx)
                            .segments[batch_idx][seg_idx]
                            .factor
                        )[-1]
                        for layer_idx in group_state.layer_indices
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
                            self._compressed_layer_state(layer_idx)
                            .segments[batch_idx][seg_idx]
                            .factor
                        )
                        for layer_idx in group_state.layer_indices
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
            grouped_segments = decompose_grouped_xkv_to_segment_store(
                group_prefix.unsqueeze(1),
                segment_ranges=segment_ranges,
                config=self._group_decomposition_config(
                    group_prefix, group_layers
                ),
            )
            shared_segments, per_layer_segments = (
                self._split_grouped_xkv_segments(
                    grouped_segments,
                    split_sizes,
                    group_layers,
                    batch_size,
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
        state = self.layer_states.get(layer_idx)
        return state is not None and state.is_compressed

    def _fused_right_factor(self, layer_idx, num_heads):
        """Reshape the packed right factor into the kernel layout."""
        state = self._compressed_layer_state(layer_idx)
        if state.fused_right is None:
            packed = state.packed_right
            batch_size, rank, flat_dim = packed.shape
            head_dim = flat_dim // num_heads
            state.fused_right = (
                packed.reshape(batch_size, rank, num_heads, head_dim)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        return state.fused_right

    def prepare_selected_chunks(self, layer_idx, chunks, chunk_size):
        state = self._compressed_layer_state(layer_idx)
        group_state = self.group_states[state.group_last_layer]
        shared = group_state.packed_shared
        if (
            not self.unrope_keys
            or is_quantised_factor(shared)
            or is_quantised_factor(state.packed_right)
            or state.inverse_permutation is not None
            or not self.layers[layer_idx].supports_fused_rope
            or not self.fused_reconstructor.available()
        ):
            return None
        right = self._fused_right_factor(layer_idx, chunks.size(1))
        if not self.fused_reconstructor.supports(
            shared, right, chunks, chunk_size
        ):
            return None
        state.fused_chunk_size = chunk_size
        return self.fused_reconstructor.reorder(layer_idx, chunks, chunk_size)

    @property
    def selective_reconstruction_nbytes(self):
        return self.fused_reconstructor.nbytes + sum(
            state.fused_right.numel() * state.fused_right.element_size()
            for state in self.layer_states.values()
            if state.fused_right is not None
        )

    @torch.no_grad()
    def retrieve_selected(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        state = self._compressed_layer_state(layer_idx)
        group_state = self.group_states[state.group_last_layer]
        shared = group_state.packed_shared
        right = state.packed_right
        suffix = self.layers[layer_idx].tensor
        prefix_len = positions.size(-1) - suffix.size(-2)
        prefix_positions = positions[..., :prefix_len]
        batch_size, num_heads, _ = prefix_positions.shape
        rank = factor_shape(shared)[-1]
        head_dim = factor_shape(right)[-1] // num_heads
        if (
            state.fused_right is not None
            and layer_idx in self.fused_reconstructor.states
        ):
            packed_rope = self.layers[layer_idx]._fused_rope(
                state.compressed_len,
                shared,
            )
            keys = self.fused_reconstructor.reconstruct(
                layer_idx,
                shared,
                state.fused_right,
                packed_rope,
                state.fused_chunk_size,
            )
        else:
            selected_a = gather_factor_rows(shared, prefix_positions)
            right = (
                dequantise_factor(right)
                .reshape(batch_size, rank, num_heads, head_dim)
                .transpose(1, 2)
            )
            keys = torch.matmul(selected_a, right)
            if self.unrope_keys:
                keys = self.layers[layer_idx]._rope_selected(
                    keys,
                    prefix_positions,
                    state.compressed_len,
                    inverse=False,
                )
        if suffix.size(-2):
            suffix_positions = (
                positions[..., prefix_len:] - state.compressed_len
            )
            suffix_keys = suffix.gather(
                2,
                suffix_positions[..., None].expand(-1, -1, -1, suffix.size(-1)),
            )
            keys = torch.cat([keys, suffix_keys], dim=2)
        return keys

    def _reconstruct_keys(self, keys, layer_idx):
        state = self._compressed_layer_state(layer_idx)
        group_state = self.group_states[state.group_last_layer]
        shared_segments = group_state.shared_segments
        layer_segments = state.segments

        paired_segments = [
            [
                FactorPairSegment(
                    token_range=shared_segment.token_range,
                    factors=(
                        shared_segment.factor,
                        layer_segment.factor,
                    ),
                )
                for shared_segment, layer_segment in zip(
                    batch_shared,
                    batch_layer,
                )
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
                compressed_len=state.compressed_len,
            )
        recon_keys = torch.cat([prefix_keys, keys], dim=-2)
        return recon_keys

    def get_reconstructed_keys_only(
        self,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """Reconstruct a layer's compressed prefix without its live suffix.

        The value cache trains against the keys it will actually see at decode
        time, so it needs the lossy reconstruction rather than the exact keys.
        Returns None when the layer has not been decomposed yet.
        """
        state = self.layer_states.get(layer_idx)
        if state is None or not state.is_compressed:
            return None
        suffix_keys = self.layers[layer_idx].tensor
        return self._reconstruct_keys(suffix_keys[..., :0, :], layer_idx)

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

        state = self.layer_states.get(layer_idx)
        if state is not None and state.is_compressed:
            recon_keys = self._reconstruct_keys(keys, layer_idx)
            check_recon_length(recon_keys, cache_kwargs)
            return recon_keys

        raise Exception(
            "Prefill is set to False and no xKV factors were found for "
            f"layer {layer_idx}. If this layer is part of a terminal partial "
            "group, pass num_layers through later plumbing so the group "
            "boundary is known."
        )
