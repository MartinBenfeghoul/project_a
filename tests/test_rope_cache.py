import torch

from cache.backends.tensor import SingleTensorDynamicLayer
from cache.rope import SharedRopeCache
from utils.rope import apply_packed_rope, inverse_packed_rope

from tests.helpers import apply_model_rope, rope_cos_sin


def test_packed_rope_matches_model_rope_and_is_invertible():
    torch.manual_seed(0)
    batch_size, num_heads, seq_len, head_dim = 2, 3, 19, 16
    tensor = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)

    rope_cache = SharedRopeCache()
    rope_cache.capture(cos, sin)
    packed = rope_cache.prefix(seq_len, tensor.device, tensor.dtype)

    expected = apply_model_rope(tensor, cos, sin)
    actual = apply_packed_rope(tensor, packed)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inverse_packed_rope(actual, packed), tensor)


def test_evicted_positions_use_their_original_rope_rows():
    torch.manual_seed(1)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 23, 16
    unroped = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    roped = apply_model_rope(unroped, cos, sin)
    kept_positions = torch.tensor([0, 3, 7, 11, 18, 22])

    layer = SingleTensorDynamicLayer(rope_cache=SharedRopeCache())
    kept_roped = roped.index_select(2, kept_positions)
    restored = layer._undo_rope(
        kept_roped,
        {
            "cos": cos,
            "sin": sin,
            "kept_positions": kept_positions,
        },
    )

    torch.testing.assert_close(
        restored,
        unroped.index_select(2, kept_positions),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        layer._apply_rope(restored, compressed_len=kept_positions.numel()),
        kept_roped,
        atol=2e-6,
        rtol=2e-6,
    )
