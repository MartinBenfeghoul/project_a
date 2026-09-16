"""xKV key cache behaviour with a baseline value cache."""

import pytest
import torch

from cache.rope import SharedRopeCache
from cache.backends.xkv import XKVKeysCache
from numerics.quantisation import factor_nbytes, is_quantised_factor
from utils.rope import inverse_rope

from tests.helpers import apply_model_rope, low_rank_keys, rope_cos_sin


def _flatten_heads(keys: torch.Tensor) -> torch.Tensor:
    """[B, H, T, D] -> the [B, 1, T, H * D] layout the decomposer receives."""
    batch_size, _, seq_len, _ = keys.shape
    return keys.transpose(1, 2).reshape(batch_size, seq_len, -1).unsqueeze(1)


def _capture_decomposition_inputs(monkeypatch):
    """Record every tensor handed to the segment-store decomposer."""
    import cache.backends.xkv as keys_module

    captured = []
    original = keys_module.decompose_grouped_xkv_to_segment_store

    def spy(tensor, *args, **kwargs):
        captured.append(tensor.clone())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(
        keys_module, "decompose_grouped_xkv_to_segment_store", spy
    )
    return captured


def _build_cache(**kwargs) -> XKVKeysCache:
    defaults = dict(
        layer_group_size=1,
        num_layers=1,
        xkv_svd_backend="linalg",
        comp_ratio=2.0,
        rope_cache=SharedRopeCache(),
    )
    return XKVKeysCache(**{**defaults, **kwargs})


@pytest.mark.parametrize("quantise_a", [False, True])
def test_compression_ratio_aggregates_bytes_across_sequences_and_groups(quantise_a):
    torch.manual_seed(41)
    batch_size, num_heads, seq_len, head_dim = 2, 2, 96, 16
    valid_lengths = [96, 24]
    mask = torch.arange(seq_len)[None, :] >= (
        seq_len - torch.tensor(valid_lengths)[:, None]
    )
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    cache = _build_cache(
        layer_group_size=2, num_layers=3, comp_ratio=2.5,
        quantise_a=quantise_a,
    )
    assert cache.calc_compression_ratio() == 0
    for layer in range(3):
        keys = apply_model_rope(
            torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
        )
        cache.update(keys, layer, {
            "cos": cos, "sin": sin, "padding_mask": mask,
        })

    originals, compressed = [], []
    for group in cache.group_states.values():
        for batch_idx, valid in enumerate(valid_lengths):
            originals.append(
                valid * len(group.layer_indices) * num_heads * head_dim
                * keys.element_size()
            )
            shared = group.shared_segments[batch_idx][0].factor
            compressed.append(
                factor_nbytes(shared) * valid / seq_len
                + sum(
                    factor_nbytes(
                        cache.layer_states[layer].segments[batch_idx][0].factor
                    ) for layer in group.layer_indices
                )
            )
    expected = sum(originals) / sum(compressed)
    old_mean = sum(o / c for o, c in zip(originals, compressed)) / len(originals)
    assert expected != pytest.approx(old_mean)
    assert cache.calc_compression_ratio() == pytest.approx(expected)
    assert cache.comp_ratio == pytest.approx(expected)


def test_keys_are_unroped_before_decomposition(monkeypatch):
    torch.manual_seed(10)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    unroped = torch.randn(batch_size, num_heads, seq_len, head_dim)
    roped = apply_model_rope(unroped, cos, sin)

    captured = _capture_decomposition_inputs(monkeypatch)
    cache = _build_cache()
    cache.update(roped, 0, {"cos": cos, "sin": sin})

    assert len(captured) == 1
    # The decomposer must see the un-roped keys, not the cached roped ones.
    torch.testing.assert_close(
        captured[0], _flatten_heads(unroped), atol=1e-5, rtol=1e-5
    )
    assert not torch.allclose(
        captured[0], _flatten_heads(roped), atol=1e-3, rtol=1e-3
    )


def test_unroped_input_matches_an_independent_inverse_rope(monkeypatch):
    """The un-roping uses the same transform as utils.rope.inverse_rope."""
    torch.manual_seed(11)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    roped = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )

    captured = _capture_decomposition_inputs(monkeypatch)
    _build_cache().update(roped, 0, {"cos": cos, "sin": sin})

    expected = inverse_rope(roped, cos.unsqueeze(1), sin.unsqueeze(1))
    torch.testing.assert_close(
        captured[0], _flatten_heads(expected), atol=1e-5, rtol=1e-5
    )


def test_value_cache_role_keeps_keys_roped(monkeypatch):
    """With unrope_keys disabled the tensor is decomposed exactly as cached."""
    torch.manual_seed(12)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    roped = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )

    captured = _capture_decomposition_inputs(monkeypatch)
    cache = _build_cache()
    cache.unrope_keys = False
    cache.update(roped, 0, {"cos": cos, "sin": sin})

    torch.testing.assert_close(
        captured[0], _flatten_heads(roped), atol=1e-6, rtol=1e-6
    )


def test_reconstruction_restores_rope():
    """Reconstructed keys come back in the roped frame they were stored in."""
    torch.manual_seed(13)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank=4)
    roped = apply_model_rope(keys, cos, sin)

    cache = _build_cache(comp_ratio=0.5)
    cache.update(roped, 0, {"cos": cos, "sin": sin})
    reconstructed = cache.get_reconstructed_keys_only(0)

    torch.testing.assert_close(reconstructed, roped, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", ("linalg", "cholqr"))
def test_full_rank_decomposition_is_near_lossless(backend):
    torch.manual_seed(14)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 64, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank=8)
    roped = apply_model_rope(keys, cos, sin)

    cache = _build_cache(xkv_svd_backend=backend, comp_ratio=0.5)
    cache.update(roped, 0, {"cos": cos, "sin": sin})

    reconstructed = cache.get_reconstructed_keys_only(0)
    error = (reconstructed - roped).norm() / roped.norm()
    assert error < 5e-2, f"{backend} relative error {error:.4f}"


def test_cholqr_decomposition_matches_linalg():
    """Both SVD backends recover the same low-rank structure."""
    torch.manual_seed(15)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 128, 16
    rank = 6
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank=rank)
    roped = apply_model_rope(keys, cos, sin)

    # A compression ratio whose rank comfortably covers the true rank.
    comp_ratio = (seq_len * num_heads * head_dim) / (
        2 * rank * (seq_len + num_heads * head_dim)
    )

    reconstructions = {}
    for backend in ("linalg", "cholqr"):
        cache = _build_cache(xkv_svd_backend=backend, comp_ratio=comp_ratio)
        cache.update(roped, 0, {"cos": cos, "sin": sin})
        reconstructions[backend] = cache.get_reconstructed_keys_only(0)

    for backend, reconstructed in reconstructions.items():
        error = (reconstructed - roped).norm() / roped.norm()
        assert error < 2e-2, f"{backend} relative error {error:.4f}"

    difference = (
        reconstructions["cholqr"] - reconstructions["linalg"]
    ).norm() / reconstructions["linalg"].norm()
    assert difference < 2e-2, f"backends differ by {difference:.4f}"


def _group_last_layer(layer_idx: int, group_size: int, num_layers: int) -> int:
    group_start = (layer_idx // group_size) * group_size
    return min(group_start + group_size - 1, num_layers - 1)


@pytest.mark.parametrize(
    ("group_size", "num_layers"),
    [(1, 4), (2, 4), (3, 4), (4, 4)],
)
def test_decomposition_only_runs_at_group_boundaries(
    monkeypatch,
    group_size,
    num_layers,
):
    torch.manual_seed(16)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)

    captured = _capture_decomposition_inputs(monkeypatch)
    cache = _build_cache(
        layer_group_size=group_size,
        num_layers=num_layers,
    )

    boundaries = [
        idx
        for idx in range(num_layers)
        if idx == _group_last_layer(idx, group_size, num_layers)
    ]
    expected_calls = 0

    for layer_idx in range(num_layers):
        keys = apply_model_rope(
            torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
        )
        cache.update(keys, layer_idx, {"cos": cos, "sin": sin})

        if layer_idx in boundaries:
            expected_calls += 1
        assert len(captured) == expected_calls, (
            f"layer {layer_idx} triggered {len(captured)} decompositions, "
            f"expected {expected_calls}"
        )

        group_last = _group_last_layer(layer_idx, group_size, num_layers)
        group_start = (layer_idx // group_size) * group_size
        for idx in range(num_layers):
            state = cache.layer_states.get(idx)
            compressed = state is not None and state.is_compressed
            if idx > layer_idx or group_start <= idx <= layer_idx < group_last:
                # Not yet decomposed: the exact keys are still cached.
                assert not compressed, f"layer {idx} compressed too early"
                if idx <= layer_idx:
                    assert cache.layers[idx].tensor.size(-2) == seq_len
            else:
                assert compressed, f"layer {idx} was not decomposed"
                # The exact prefix is dropped once factors exist.
                assert cache.layers[idx].tensor.size(-2) == 0

    # Every layer belongs to exactly one decomposed group by the end.
    assert len(captured) == len(boundaries)
    assert sorted(cache.group_states) == boundaries
    covered = sorted(
        idx
        for state in cache.group_states.values()
        for idx in state.layer_indices
    )
    assert covered == list(range(num_layers))


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_group_members_share_one_decomposition_call(
    monkeypatch,
    group_size,
):
    """One SVD per group, over the concatenated keys of all its layers."""
    torch.manual_seed(17)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    num_layers = 4
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)

    captured = _capture_decomposition_inputs(monkeypatch)
    cache = _build_cache(
        layer_group_size=group_size,
        num_layers=num_layers,
    )
    for layer_idx in range(num_layers):
        cache.update(
            apply_model_rope(
                torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
            ),
            layer_idx,
            {"cos": cos, "sin": sin},
        )

    assert len(captured) == num_layers // group_size
    for tensor in captured:
        # Heads of every layer in the group are concatenated side by side.
        assert tensor.shape == (
            batch_size,
            1,
            seq_len,
            group_size * num_heads * head_dim,
        )

    # The shared left factor is stored once per group, not once per layer.
    for group_last, group_state in cache.group_states.items():
        assert len(group_state.layer_indices) == group_size
        assert group_state.layer_indices[-1] == group_last
        assert group_state.packed_shared is not None


def _quantised_cache(**kwargs):
    return _build_cache(quantise_a=True, compressor_bits=4, **kwargs)


def test_folded_rotation_matches_decoding_every_gathered_row():
    """The folded right factor reproduces the plain dequantise-then-GEMM path.
    """
    from numerics.quantisation import (
        fold_rotation,
        gather_factor_rows,
        gather_factor_rows_unrotated,
    )

    torch.manual_seed(21)
    batch_size, num_heads, seq_len, head_dim, rank = 1, 2, 64, 16, 24
    cache = _quantised_cache()
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank)
    cache.update(
        apply_model_rope(keys, cos, sin),
        0,
        {"cos": cos, "sin": sin, "cache_position": torch.arange(seq_len)},
    )

    shared = cache.group_states[0].packed_shared
    right = cache.layer_states[0].packed_right
    assert is_quantised_factor(shared)

    positions = torch.randint(0, seq_len, (batch_size, num_heads, 16))
    stored_rank = right.shape[1]

    reference = torch.matmul(
        gather_factor_rows(shared, positions),
        right.reshape(batch_size, stored_rank, num_heads, head_dim).transpose(
            1, 2
        ),
    )
    rows, norms = gather_factor_rows_unrotated(shared, positions)
    folded = torch.matmul(
        rows,
        fold_rotation(shared, right)
        .reshape(batch_size, stored_rank, num_heads, head_dim)
        .transpose(1, 2),
    ) * norms[..., None].to(rows.dtype)

    torch.testing.assert_close(folded, reference, atol=2e-2, rtol=2e-2)


def test_right_factor_is_folded_in_place_without_extra_memory():
    """The rotation replaces the stored right factor rather than joining it.

    Recovering `B` from the stored `R @ B` (R is orthogonal, so
    `B == R.T @ folded`) lets us check the folded storage against the plain
    dequantise-then-multiply semantics it stands in for.
    """
    from numerics.quantisation import (
        gather_factor_rows,
        get_turboquant_compressor,
    )

    torch.manual_seed(23)
    batch_size, num_heads, seq_len, head_dim, rank = 1, 2, 64, 16, 24
    cache = _quantised_cache()
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank)
    cache.update(
        apply_model_rope(keys, cos, sin),
        0,
        {"cos": cos, "sin": sin, "cache_position": torch.arange(seq_len)},
    )
    state = cache.layer_states[0]
    shared = cache.group_states[0].packed_shared
    assert is_quantised_factor(shared)
    assert state.right_is_folded

    # Nothing is cached alongside the stored factors for decode.
    assert cache.selective_reconstruction_nbytes == 0

    positions = (
        torch.arange(seq_len)
        .reshape(1, 1, -1)
        .expand(batch_size, num_heads, -1)
    )
    retrieved = cache.retrieve_selected(0, positions)

    stored_rank = state.packed_right.shape[1]
    rotation = get_turboquant_compressor(
        stored_rank, shared.bits, state.packed_right.device
    ).rotation
    unfolded = torch.matmul(rotation.T, state.packed_right.float())
    reference = torch.matmul(
        gather_factor_rows(shared, positions).float(),
        unfolded.reshape(batch_size, stored_rank, num_heads, head_dim)
        .transpose(1, 2),
    ).to(retrieved.dtype)
    reference = cache.layers[0]._rope_selected(
        reference, positions, state.compressed_len, inverse=False
    )
    torch.testing.assert_close(retrieved, reference, atol=2e-2, rtol=2e-2)


def test_selective_and_full_reconstruction_agree_when_folded():
    """Both decode routes read the folded factor; they must not diverge."""
    torch.manual_seed(24)
    batch_size, num_heads, seq_len, head_dim, rank = 1, 2, 64, 16, 24
    cache = _quantised_cache()
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank)
    kwargs = {"cos": cos, "sin": sin, "cache_position": torch.arange(seq_len)}
    cache.update(apply_model_rope(keys, cos, sin), 0, kwargs)
    assert cache.layer_states[0].right_is_folded

    positions = (
        torch.arange(seq_len)
        .reshape(1, 1, -1)
        .expand(batch_size, num_heads, -1)
    )
    selective = cache.retrieve_selected(0, positions)
    # The non-selective route reconstructs through reconstruct_segments.
    full = cache.get_reconstructed_keys_only(0)
    torch.testing.assert_close(
        selective, full[..., :seq_len, :], atol=2e-2, rtol=2e-2
    )
