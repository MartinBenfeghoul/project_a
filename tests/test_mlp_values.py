"""MLP value cache: residual initialisation is collected and applied."""

import pytest
import torch

from cache.rope import SharedRopeCache
from cache.backends.mlp_values import MLPValueCache
from model.mlp import MLP
from utils.model import extract_kv_linear_init

from tests.helpers import apply_model_rope, build_llama, rope_cos_sin


def _project(layer, hidden_states, head_dim):
    """Reproduce the model's per-KV-head key and value projections."""
    batch_size, seq_len, _ = hidden_states.shape
    shape = (batch_size, seq_len, -1, head_dim)
    keys = layer.self_attn.k_proj(hidden_states).view(shape).transpose(1, 2)
    values = layer.self_attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    return keys, values


def _apply_residual(keys, w_linear):
    return torch.einsum("bhtd,hde->bhte", keys, w_linear)


def test_linear_init_matches_pinv_of_key_projection():
    model = build_llama(3, num_attention_heads=4, num_key_value_heads=2)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads

    weights = extract_kv_linear_init(model)
    assert len(weights) == model.config.num_hidden_layers

    for layer_idx, layer in enumerate(model.model.layers):
        assert weights[layer_idx].shape == (num_kv_heads, head_dim, head_dim)
        w_k = layer.self_attn.k_proj.weight.detach().T.float()
        w_v = layer.self_attn.v_proj.weight.detach().T.float()
        for head in range(num_kv_heads):
            columns = slice(head * head_dim, (head + 1) * head_dim)
            expected = torch.linalg.pinv(w_k[:, columns]) @ w_v[:, columns]
            torch.testing.assert_close(
                weights[layer_idx][head].float(),
                expected,
                atol=1e-4,
                rtol=1e-4,
            )

    # Each layer gets its own matrix.
    assert not torch.allclose(weights[0], weights[1])


def test_linear_init_maps_keys_onto_values_exactly_when_invertible():
    """With a square key projection the residual reproduces values exactly."""
    torch.manual_seed(20)
    head_dim = 16
    model = build_llama(
        2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
    )
    weights = extract_kv_linear_init(model)
    hidden_states = torch.randn(1, 12, model.config.hidden_size)

    for layer_idx, layer in enumerate(model.model.layers):
        keys, values = _project(layer, hidden_states, head_dim)
        predicted = _apply_residual(keys, weights[layer_idx])
        torch.testing.assert_close(predicted, values, atol=1e-3, rtol=1e-3)


def test_mlp_adds_the_residual_branch():
    torch.manual_seed(21)
    num_heads, head_dim = 2, 8
    mlp = MLP(head_dim=head_dim, num_heads=num_heads, use_residual=True)
    w_linear = torch.randn(num_heads, head_dim, head_dim)
    with torch.no_grad():
        mlp.W_linear.copy_(w_linear)

    inputs = torch.randn(1, num_heads, 5, head_dim)
    with torch.no_grad():
        with_residual = mlp(inputs)
        for parameter in list(mlp.weights) + list(mlp.biases):
            parameter.zero_()
        residual_only = mlp(inputs)

    torch.testing.assert_close(residual_only, _apply_residual(inputs, w_linear))
    assert not torch.allclose(with_residual, residual_only)


def test_mlp_without_residual_has_no_linear_branch():
    mlp = MLP(head_dim=8, num_heads=2, use_residual=False)
    assert not hasattr(mlp, "W_linear")
    assert mlp.residual_eq is None


def _prefill(cache, keys, values, cos, sin, layer_idx=0):
    batch_size, _, seq_len, _ = keys.shape
    return cache.update(
        values,
        layer_idx,
        {
            "keys": keys,
            "cos": cos,
            "sin": sin,
            "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        },
    )


@pytest.mark.parametrize("as_callable", (False, True))
def test_residual_weights_reach_each_layer_mlp(as_callable):
    torch.manual_seed(22)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 8
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    per_layer = [torch.randn(num_heads, head_dim, head_dim) for _ in range(3)]

    cache = MLPValueCache(
        target_cr=2.0,
        num_epochs=0,
        use_residual=True,
        W_linear_per_layer=(lambda: per_layer) if as_callable else per_layer,
        rope_cache=SharedRopeCache(),
    )
    assert cache.W_linear_per_layer is not None

    for layer_idx in range(3):
        keys = apply_model_rope(
            torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
        )
        _prefill(
            cache,
            keys,
            torch.randn_like(keys),
            cos,
            sin,
            layer_idx=layer_idx,
        )
        torch.testing.assert_close(
            cache.layers[layer_idx].mlp.W_linear.detach(),
            per_layer[layer_idx],
        )


def test_linear_weights_are_ignored_without_use_residual():
    torch.manual_seed(23)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 8
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    cache = MLPValueCache(
        target_cr=2.0,
        num_epochs=0,
        use_residual=False,
        W_linear_per_layer=[torch.randn(num_heads, head_dim, head_dim)],
        rope_cache=SharedRopeCache(),
    )
    assert cache.W_linear_per_layer is None

    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    _prefill(cache, keys, torch.randn_like(keys), cos, sin)
    assert not hasattr(cache.layers[0].mlp, "W_linear")


def test_residual_reconstructs_values_it_was_derived_from():
    """End to end: the residual carries real signal from keys to values.

    The key projection is square here, so the exact residual solution exists
    and an untrained MLP with a zero residual budget must find it.
    """
    torch.manual_seed(24)
    head_dim = 16
    seq_len = 64
    model = build_llama(
        1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
    )
    weights = extract_kv_linear_init(model)
    hidden_states = torch.randn(1, seq_len, model.config.hidden_size)
    keys, values = _project(model.model.layers[0], hidden_states, head_dim)

    cos, sin = rope_cos_sin(1, seq_len, head_dim)
    roped_keys = apply_model_rope(keys, cos, sin)

    errors = {}
    for use_residual in (False, True):
        cache = MLPValueCache(
            # A ratio too high to afford residual rows, so the reported
            # values come from the MLP alone.
            target_cr=1e6,
            num_epochs=0,
            use_residual=use_residual,
            W_linear_per_layer=weights if use_residual else None,
            rope_cache=SharedRopeCache(),
        )
        _prefill(cache, roped_keys, values, cos, sin)
        assert cache.layers[0].indices.numel() == 0
        predicted = cache.layers[0].decompress(
            cache.layers[0]._undo_rope(
                roped_keys,
                {"cos": cos, "sin": sin},
                prefill=False,
                compressed_len=seq_len,
            )
        )
        errors[use_residual] = (
            (predicted - values).norm() / values.norm()
        ).item()

    assert errors[True] < 0.5 * errors[False], errors
    assert errors[True] < 0.1, errors


@pytest.mark.parametrize("turboquant_residuals", (False, True))
def test_selected_residual_rows_match_a_full_decode(turboquant_residuals):
    """Selective retrieval decodes rows one by one, quantised or not.

    Quantised residuals are stored as CompressorParams rather than a tensor,
    so the row lookup has to slice the compressed store and decode that,
    which is what this pins.
    """
    torch.manual_seed(25)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    values = torch.randn_like(keys)

    cache = MLPValueCache(
        # Low enough to afford a residual row for every token.
        target_cr=1e-3,
        num_epochs=0,
        turboquant_residuals=turboquant_residuals,
        compressor_bits=4,
        rope_cache=SharedRopeCache(),
    )
    _prefill(cache, keys, values, cos, sin)

    layer = cache.layers[0]
    assert layer.indices.numel() > 0, "no residuals were stored"

    positions = (
        torch.arange(seq_len // 2)
        .reshape(1, 1, -1)
        .expand(batch_size, num_heads, -1)
    )
    rows, _ = layer._lookup_stored_rows(positions)

    torch.testing.assert_close(
        layer._decode_residual_rows(rows),
        layer._decode_residuals().index_select(0, rows),
    )


@pytest.mark.parametrize("turboquant_residuals", (False, True))
def test_selective_retrieval_reads_stored_residuals(turboquant_residuals):
    """The selective decode path reaches the residual store without blowing up."""
    torch.manual_seed(26)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    values = torch.randn_like(keys)

    cache = MLPValueCache(
        target_cr=1e-3,
        num_epochs=0,
        turboquant_residuals=turboquant_residuals,
        compressor_bits=4,
        rope_cache=SharedRopeCache(),
    )
    _prefill(cache, keys, values, cos, sin)
    layer = cache.layers[0]
    layer.prefill = False
    assert cache.supports_selective_retrieval(0)
    assert layer.indices.numel() > 0

    selected = 32
    positions = (
        torch.arange(selected)
        .reshape(1, 1, -1)
        .expand(batch_size, num_heads, -1)
    )
    retrieved = cache.retrieve_selected(keys[:, :, :selected], positions, 0)

    assert retrieved.shape == (batch_size, num_heads, selected, head_dim)
    assert torch.isfinite(retrieved).all()


def _layer_with(learned_weights, *, use_residual, w_linear_init=None):
    """Lazily initialise one value layer against a given learned init."""
    from cache.backends.mlp_values import MLPValueLayer
    from model.meta_learning import LearnedLayerInit

    layer = MLPValueLayer(
        target_cr=4.0,
        learned_init=LearnedLayerInit(weights=learned_weights),
        use_residual=use_residual,
        W_linear_init=w_linear_init,
    )
    layer.lazy_initialization(torch.zeros(1, 2, 4, 8))
    return layer


def test_checkpoint_w_linear_wins_over_the_model_derived_one():
    """MLPValueCache drops W_linear_init for such checkpoints; the layer has
    to agree, or a layer built directly resolves W_linear the other way."""
    checkpoint = MLP(num_heads=2, head_dim=8, use_residual=True).state_dict()
    checkpoint["W_linear"] = torch.full_like(checkpoint["W_linear"], 3.0)
    derived = torch.full_like(checkpoint["W_linear"], 7.0)

    layer = _layer_with(
        checkpoint, use_residual=True, w_linear_init=derived
    )

    torch.testing.assert_close(layer.mlp.W_linear, checkpoint["W_linear"])


def test_a_no_residual_checkpoint_loads_under_use_residual():
    """Meta-training with use_residual=false must still evaluate with it on."""
    checkpoint = MLP(num_heads=2, head_dim=8, use_residual=False).state_dict()
    assert "W_linear" not in checkpoint
    derived = torch.full((2, 8, 8), 5.0)

    layer = _layer_with(checkpoint, use_residual=True, w_linear_init=derived)

    torch.testing.assert_close(layer.mlp.W_linear, derived)
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(layer.mlp.state_dict()[name], tensor)


def test_a_residual_checkpoint_loads_without_use_residual():
    checkpoint = MLP(num_heads=2, head_dim=8, use_residual=True).state_dict()

    layer = _layer_with(checkpoint, use_residual=False)

    assert not hasattr(layer.mlp, "W_linear")
    for name, tensor in checkpoint.items():
        if name != "W_linear":
            torch.testing.assert_close(layer.mlp.state_dict()[name], tensor)
