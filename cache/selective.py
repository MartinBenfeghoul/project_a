"""Selective reconstruction"""

import math
from dataclasses import dataclass

import torch

from efficiency import FusedLandmarkScorer
from .config import SelectiveCacheConfig


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


class SelectiveReconstruction:
    """Owns the per-layer landmark state and the decode-time selection."""

    def __init__(
        self,
        config: SelectiveCacheConfig,
        key_cache,
        value_cache,
    ):
        self.config = config
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.layers: dict[int, SelectiveLayerState] = {}
        self._scorer = FusedLandmarkScorer()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _build_chunk_landmarks(
        self,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mean-key landmarks and validity masks for fixed-size chunks"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        chunk_size = self.config.chunk_size
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
        local_start = max(0, seq_len - self.config.local_tokens)
        return num_chunks - local_start // self.config.chunk_size

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
        if local_chunks > 0:
            similarity[..., -local_chunks:, :] = torch.inf
        outlier_count = min(
            self.config.outlier_chunks,
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
            self.config.chunk_size,
            device=chunks.device,
        )
        return (
            chunks[..., None] * self.config.chunk_size + offsets
        ).flatten(2)

    def _record_overhead(self, layer_idx: int) -> None:
        """Cache persistent selective-key bytes and pass them to rank selection"""
        nbytes = self.layers[layer_idx].key_overhead_nbytes
        record = getattr(self.key_cache, "set_selective_overhead", None)
        if callable(record):
            record(layer_idx, nbytes)

    def store_landmarks(
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
        if local_chunks > 0:
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
                max(0, seq_len - self.config.local_tokens),
                seq_len,
                device=key_states.device,
            )
            .reshape(1, 1, -1)
            .expand(key_states.size(0), key_states.size(1), -1)
        )
        positions = torch.cat([outlier_positions, local_positions], dim=-1)

        gather_idx = positions[..., None].expand(-1, -1, -1, head_dim)
        self.layers[layer_idx] = SelectiveLayerState(
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
        self._record_overhead(layer_idx)

    def select_positions(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        state = self.layers[layer_idx]
        landmarks = state.landmarks
        batch_size, num_query_heads, _, head_dim = query_states.shape
        num_kv_heads = landmarks.size(1)
        num_groups = num_query_heads // num_kv_heads
        grouped_query = query_states.reshape(
            batch_size, num_kv_heads, num_groups, head_dim
        )
        scores = self._scorer.score(grouped_query, landmarks)
        scores = scores[..., : state.landmark_count]
        selected_chunks = math.ceil(
            self.config.token_budget / self.config.chunk_size
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
                self.config.chunk_size,
            )
            if reordered is not None:
                chunks = reordered
        offsets = torch.arange(
            self.config.chunk_size,
            device=scores.device,
        )
        positions = (
            chunks[..., None] * self.config.chunk_size + offsets
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

    def supports_retrieval(self, layer_idx: int) -> bool:
        value_supports = getattr(
            self.value_cache, "supports_selective_retrieval", None
        )
        key_supports = getattr(
            self.key_cache, "supports_selective_retrieval", None
        )
        return (
            layer_idx in self.layers
            and callable(value_supports)
            and value_supports(layer_idx)
            and (not callable(key_supports) or key_supports(layer_idx))
        )

    def retrieve(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.layers[layer_idx]
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

    @property
    def scorer_nbytes(self) -> int:
        return self._scorer.nbytes
