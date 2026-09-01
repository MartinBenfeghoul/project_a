import math
import os

import pytest
import torch

from efficiency.xkv import FusedKeyReconstructor

from tests.helpers import rope_cos_sin


pytestmark = pytest.mark.cuda


def _fused_reference(
    shared: torch.Tensor,
    right: torch.Tensor,
    chunks: torch.Tensor,
    packed_rope: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch_size, num_heads, _ = chunks.shape
    head_dim = right.size(-2)
    dense = torch.einsum(
        "btr,bhdr->bhtd",
        shared.float(),
        right.float(),
    )
    positions = (
        chunks[..., None] * chunk_size
        + torch.arange(chunk_size, device=shared.device)
    ).reshape(batch_size, num_heads, -1)
    selected = dense.gather(
        2,
        positions[..., None].expand(-1, -1, -1, head_dim),
    )
    selected_rope = packed_rope[positions]
    half_dim = head_dim // 2
    first, second = selected[..., :half_dim], selected[..., half_dim:]
    cos = selected_rope[..., :half_dim].float()
    sin = selected_rope[..., half_dim:].float()
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )


def test_fused_reconstruction_matches_reference_and_reuses_cache():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("set RUN_CUDA_TESTS=1 to run CUDA extension tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not FusedKeyReconstructor.available():
        pytest.skip("fused xKV extension is not installed")

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size, num_heads = 1, 2
    seq_len, head_dim, rank = 1024, 128, 64
    select_sets, chunk_size = 128, 8
    shared = (
        torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype)
        / math.sqrt(rank)
    )
    right = (
        torch.randn(
            batch_size,
            num_heads,
            head_dim,
            rank,
            device=device,
            dtype=dtype,
        )
        / math.sqrt(rank)
    ).contiguous()
    cos, sin = rope_cos_sin(
        batch_size,
        seq_len,
        head_dim,
        dtype=dtype,
        device=device,
        theta=10_000_000.0,
    )
    packed_rope = torch.cat((cos[0, :, :64], sin[0, :, :64]), dim=-1)
    chunks = torch.stack(
        [
            torch.randperm(seq_len // chunk_size, device=device)[:select_sets]
            for _ in range(num_heads)
        ]
    )[None].long()

    reconstructor = FusedKeyReconstructor()
    reordered = reconstructor.reorder(0, chunks, chunk_size)
    assert reconstructor.hit_counts(0).eq(0).all()
    actual = reconstructor.reconstruct(
        0,
        shared,
        right,
        packed_rope,
        chunk_size,
    )
    torch.cuda.synchronize()
    expected = _fused_reference(
        shared,
        right,
        reordered,
        packed_rope,
        chunk_size,
    )
    relative_error = (
        (actual.float() - expected).norm() / expected.norm()
    ).item()
    assert relative_error < 0.02
    first_result = actual.clone()

    reconstructor.reorder(0, chunks, chunk_size)
    assert reconstructor.hit_counts(0).eq(select_sets).all()
    repeated = reconstructor.reconstruct(
        0,
        shared,
        right,
        packed_rope,
        chunk_size,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(repeated, first_result, atol=0, rtol=0)


def _requires_fused_ops():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("set RUN_CUDA_TESTS=1 to run CUDA extension tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not FusedKeyReconstructor.available():
        pytest.skip("fused xKV extension is not installed")


@pytest.mark.parametrize("quantise", (False, True))
def test_selective_retrieval_on_the_fused_path(quantise):
    """Selected keys match a full reconstruction, fused or not.

    Quantised factors cannot feed the fused kernels, so the cache must fall
    back to the dense matmul path instead of handing them over.
    """
    _requires_fused_ops()

    from cache.base import SharedRopeCache
    from cache.keys import XKVKeysCache
    from tests.helpers import apply_model_rope, gather_tokens

    torch.manual_seed(8)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size, num_heads = 1, 2
    seq_len, head_dim, chunk_size = 2048, 128, 8
    selected_chunks = 128

    cos, sin = rope_cos_sin(
        batch_size, seq_len, head_dim, dtype=dtype, device=device
    )
    keys = apply_model_rope(
        torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device, dtype=dtype,
        ),
        cos,
        sin,
    )

    cache = XKVKeysCache(
        layer_group_size=1,
        num_layers=1,
        xkv_svd_backend="linalg",
        comp_ratio=2.0,
        quantise_a=quantise,
        quantise_b=quantise,
        compressor_bits=8,
        rope_cache=SharedRopeCache(),
    )
    cache.update(keys, 0, {"cos": cos, "sin": sin})
    cache.update_events()

    num_chunks = seq_len // chunk_size
    chunks = torch.randperm(num_chunks, device=device)[:selected_chunks]
    chunks = chunks.reshape(1, 1, -1).expand(batch_size, num_heads, -1)
    chunks = chunks.contiguous()

    reordered = cache.prepare_selected_chunks(0, chunks, chunk_size)
    if quantise:
        assert reordered is None, "quantised factors reached the fused kernels"
    else:
        assert reordered is not None, "the fused path was not taken"
        chunks = reordered

    offsets = torch.arange(chunk_size, device=device)
    positions = (chunks[..., None] * chunk_size + offsets).reshape(
        batch_size, num_heads, -1
    )

    selected = cache.retrieve_selected(0, positions)
    expected = gather_tokens(cache.get_reconstructed_keys_only(0), positions)

    selected, expected = selected.float(), expected.float()
    error = (selected - expected).norm() / expected.norm()
    assert error < 5e-2, f"selected keys deviate by {error:.4f}"
