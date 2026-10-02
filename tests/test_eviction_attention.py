"""Evicted padding rows must remain invisible during ordinary/selective decode."""

import os

import pytest
import torch

from cache import CompressedCache
from cache.config import (
    CompressedCacheConfig,
    XKVCacheConfig,
    MLPValueCacheConfig,
    SelectiveCacheConfig,
)
from model.attention_predictor import (
    AttentionPredictor,
    install_attention_predictor_hooks,
)
from model.selective_attention import install_selective_attention
from model.unpadded_attention import install_unpadded_sdpa
from tests.helpers import build_llama


@pytest.mark.parametrize(
    "backend", ["sdpa", "eager", "unpadded_sdpa", "flash_attention_2"]
)
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_evicted_padding_cannot_change_logits(backend, selective, batch):
    gpu = backend == "flash_attention_2"
    if gpu:
        if (
            os.environ.get("RUN_CUDA_TESTS") != "1"
            or not torch.cuda.is_available()
        ):
            pytest.skip("Enable RUN_CUDA_TESTS=1 with a CUDA GPU for FA2")
        pytest.importorskip("flash_attn")
    device = "cuda" if gpu else "cpu"
    dtype = torch.bfloat16 if gpu else torch.float32
    model = build_llama(4).to(device=device, dtype=dtype)
    if backend == "unpadded_sdpa":
        install_unpadded_sdpa(model)
    else:
        model.set_attn_implementation(backend)
    predictor = AttentionPredictor().to(device).eval()
    # lm_eval installs selective attention before the predictor; the benchmark
    # installs it afterwards. Exercise the former ordering here.
    handles = install_selective_attention(model) if selective else []
    handles += install_attention_predictor_hooks(model, predictor)
    torch.manual_seed(19)
    ids = torch.randint(3, 64, (batch, 256), device=device)
    mask = torch.ones_like(ids)
    for row in range(batch):
        mask[row, : row * 24] = 0
        ids[row, : row * 24] = 0
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    config = CompressedCacheConfig(
        key=XKVCacheConfig(
            layer_group_size=2,
            num_layers=4,
            svd_backend="linalg",
            compression_ratio=2,
        ),
        value=MLPValueCacheConfig(target_compression_ratio=2, num_epochs=0),
        selective=SelectiveCacheConfig(
            enabled=selective,
            token_budget=32,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=2,
        ),
        eviction_keep_ratio=0.625,
    )
    results = []
    checked = []
    try:
        for poison in (False, True):
            torch.manual_seed(7)
            cache = CompressedCache(
                config=config,
                cache_context={"padding_mask": mask},
                verbose=False,
            )
            step_index = [0]

            def valid_rows(layer):
                kept = cache.kept_positions[layer].to(device)
                # Include an explicitly masked generated token, ensuring that
                # suffix validity is preserved rather than blindly using ones.
                suffix = mask.new_ones((batch, step_index[0] + 1))
                suffix[-1, 0] = 0
                return torch.cat(
                    (mask.index_select(1, kept), suffix), -1
                ).bool()

            def check(module, args, kwargs):
                if kwargs["hidden_states"].shape[1] != 1:
                    return
                expected = valid_rows(module.layer_idx)
                actual = kwargs["attention_mask"]
                if actual is None:
                    actual = torch.ones_like(expected)
                elif actual.ndim == 4:
                    actual = actual[:, 0, 0]
                    if actual.dtype != torch.bool:
                        actual = actual == 0
                torch.testing.assert_close(actual, expected)
                checked.append(module.layer_idx)

            checks = [
                layer.self_attn.register_forward_pre_hook(
                    check, with_kwargs=True
                )
                for layer in model.model.layers
            ]
            original_update = cache.update
            original_retrieve = cache.retrieve_selected

            def update(keys, values, layer, cache_kwargs=None):
                decoded = keys.shape[-2] == 1
                k, v = original_update(keys, values, layer, cache_kwargs)
                if poison and decoded:
                    invalid = ~valid_rows(layer)[:, None, :, None]
                    k = k.masked_fill(invalid, 100)
                    v = v.masked_fill(invalid, -100)
                return k, v

            def retrieve(layer, selected):
                k, v = original_retrieve(layer, selected)
                if poison:
                    valid = (
                        valid_rows(layer)[:, None, :]
                        .expand(batch, selected.size(1), -1)
                        .gather(2, selected)
                    )
                    k = k.masked_fill(~valid[..., None], 100)
                    v = v.masked_fill(~valid[..., None], -100)
                return k, v

            cache.update = update
            cache.retrieve_selected = retrieve
            try:
                with torch.no_grad():
                    model(
                        ids,
                        attention_mask=mask,
                        position_ids=positions,
                        past_key_values=cache,
                    )
                    outputs = []
                    for step in range(3):
                        step_index[0] = step
                        suffix = mask.new_ones((batch, step + 1))
                        suffix[-1, 0] = 0
                        out = model(
                            ids[:, -1:],
                            attention_mask=torch.cat((mask, suffix), -1),
                            position_ids=positions[:, -1:] + step + 1,
                            past_key_values=cache,
                        )
                        outputs.append(out.logits.detach())
                    results.append(torch.stack(outputs))
            finally:
                for handle in checks:
                    handle.remove()
        assert len(checked) == 2 * 3 * 4
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    finally:
        for handle in reversed(handles):
            handle.remove()


def test_all_valid_evicted_cache_keeps_mask_free_decode():
    cache = CompressedCache(
        config=CompressedCacheConfig(eviction_keep_ratio=0.5), verbose=False
    )
    mask = torch.ones(1, 160, dtype=torch.bool)
    cache.set_generation_attention_mask(mask)
    cache.set_value_importance(0, torch.ones(1, 2, 160))
    cache.update(torch.randn(1, 2, 160, 8), torch.randn(1, 2, 160, 8), 0)
    cache.set_generation_attention_mask(torch.ones(1, 161, dtype=torch.bool))
    assert cache.eviction_decode_mask(0) is None
