"""Shared RoPE state for the cache layers."""

import torch

from utils.rope import apply_packed_rope, inverse_packed_rope


class SharedRopeCache:
    """One packed RoPE table shared by every key/value cache layer."""

    def __init__(self):
        self.packed: torch.Tensor | None = None

    def capture(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        expected_shape = (cos.size(0), 1, cos.size(2), cos.size(3))
        if self.packed is not None and self.packed.shape == expected_shape:
            return

        half_dim = cos.size(-1) // 2
        self.packed = torch.cat(
            (cos[..., :half_dim], sin[..., :half_dim]),
            dim=-1,
        ).detach().contiguous()

    def _table(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.packed is None:
            raise ValueError(
                "RoPE state is unavailable. The model must provide 'cos' and "
                "'sin' in cache_kwargs during prefill."
            )
        return self.packed.to(device=device, dtype=dtype)

    def prefix(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        if positions is None:
            if length > table.size(2):
                raise ValueError(
                    f"Requested RoPE length {length}, but only "
                    f"{table.size(2)} positions were captured."
                )
            return table[:, :, :length]
        positions = positions[:length].to(device=table.device, dtype=torch.long)
        return table.index_select(2, positions)

    def selected(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        batch_size, num_heads = positions.shape[:2]
        if table.size(0) == 1 and batch_size > 1:
            table = table.expand(batch_size, -1, -1, -1)
        if table.size(0) != batch_size:
            raise ValueError(
                f"RoPE batch size {table.size(0)} does not match "
                f"requested batch size {batch_size}."
            )
        table = table.expand(-1, num_heads, -1, -1)
        gather_idx = positions[..., None].expand(
            -1, -1, -1, table.size(-1)
        )
        return table.gather(2, gather_idx)

    def fused(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self._table(device, dtype)
        if length > table.size(2):
            raise ValueError(
                f"Requested RoPE length {length}, but only "
                f"{table.size(2)} positions were captured."
            )
        return table[0, 0, :length].contiguous()

    @property
    def nbytes(self) -> int:
        if self.packed is None:
            return 0
        return self.packed.numel() * self.packed.element_size()

    @property
    def supports_fused(self) -> bool:
        return self.packed is not None and self.packed.size(0) == 1


class RopeLayerMixin:
    """Use shared packed RoPE state to un-rope/re-rope cached tensors."""

    def _init_rope(self, rope_cache: SharedRopeCache | None = None):
        self.rope_cache = rope_cache or SharedRopeCache()
        self.rope_positions: torch.Tensor | None = None

    def _resolve_packed_rope(
        self,
        prefix_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_kwargs: dict | None = None,
    ) -> torch.Tensor:
        if (
            cache_kwargs is not None
            and "cos" in cache_kwargs
            and "sin" in cache_kwargs
        ):
            self.rope_cache.capture(cache_kwargs["cos"], cache_kwargs["sin"])
            kept_positions = cache_kwargs.get("kept_positions")
            if kept_positions is not None:
                self.rope_positions = kept_positions.detach()

        return self.rope_cache.prefix(
            prefix_len,
            device,
            dtype,
            positions=self.rope_positions,
        )

    def _undo_rope(
        self,
        keys: torch.Tensor,
        cache_kwargs: dict | None = None,
        prefill: bool = True,
        compressed_len: int = 0,
    ) -> torch.Tensor:
        T = keys.shape[2]
        prefix_len = T if prefill else min(compressed_len, T)
        if prefix_len == 0:
            return keys

        packed = self._resolve_packed_rope(
            prefix_len,
            keys.device,
            keys.dtype,
            cache_kwargs=cache_kwargs if prefill else None,
        )

        prefix = inverse_packed_rope(keys[:, :, :prefix_len], packed)
        if prefix_len < T:
            return torch.cat([prefix, keys[:, :, prefix_len:]], dim=2)
        return prefix

    def _rope_selected(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        compressed_len: int,
        inverse: bool,
    ) -> torch.Tensor:
        positions = positions.to(device=keys.device, dtype=torch.long)
        if compressed_len <= 0:
            return keys
        valid = positions < compressed_len
        rope_positions = positions.clamp(min=0, max=compressed_len - 1)
        if self.rope_positions is not None:
            mapping = self.rope_positions.to(
                device=keys.device,
                dtype=torch.long,
            )
            rope_positions = mapping[rope_positions]
        packed = self.rope_cache.selected(
            rope_positions,
            keys.device,
            keys.dtype,
        )
        transformed = (
            inverse_packed_rope(keys, packed)
            if inverse
            else apply_packed_rope(keys, packed)
        )
        return torch.where(valid[..., None], transformed, keys)

    def _apply_rope(
        self,
        keys: torch.Tensor,
        compressed_len: int = 0,
    ) -> torch.Tensor:
        T = keys.shape[2]
        prefix_len = min(compressed_len, T)
        if prefix_len == 0:
            return keys

        packed = self._resolve_packed_rope(
            prefix_len,
            keys.device,
            keys.dtype,
        )

        prefix = keys[:, :, :prefix_len]
        prefix_roped = apply_packed_rope(prefix, packed)
        if prefix_len < T:
            return torch.cat([prefix_roped, keys[:, :, prefix_len:]], dim=2)
        return prefix_roped

    def _fused_rope(self, length: int, tensor: torch.Tensor) -> torch.Tensor:
        return self.rope_cache.fused(length, tensor.device, tensor.dtype)

    @property
    def supports_fused_rope(self) -> bool:
        return self.rope_positions is None and self.rope_cache.supports_fused
