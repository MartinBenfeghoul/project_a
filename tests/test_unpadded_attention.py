"""unpadded_sdpa must reproduce the sdpa path on left-padded batches."""

import pytest
import torch
from transformers import DynamicCache

from cache.config import (
    CompressedCacheConfig,
    MLPValueCacheConfig,
    SelectiveCacheConfig,
    XKVCacheConfig,
)
from cache.core import CompressedCache
from model.selective_attention import install_selective_attention
from model.unpadded_attention import install_unpadded_sdpa
from tests.helpers import build_llama

NUM_LAYERS = 4
PROMPT_LEN = 48
PADS = (0, 5, 17)


def _stock_and_unpadded_models():
    stock = build_llama(NUM_LAYERS)
    unpadded = build_llama(NUM_LAYERS)
    install_unpadded_sdpa(unpadded)
    return stock, unpadded


def _left_padded_batch(seed):
    torch.manual_seed(seed)
    input_ids = torch.randint(3, 64, (len(PADS), PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    for row, pad in enumerate(PADS):
        input_ids[row, :pad] = 0
        attention_mask[row, :pad] = 0
    return input_ids, attention_mask


def test_prefill_and_chunked_continuation_match_stock_sdpa():
    input_ids, attention_mask = _left_padded_batch(seed=0)
    chunk = torch.randint(3, 64, (len(PADS), 2))
    chunk_mask = torch.cat([attention_mask, torch.ones_like(chunk)], dim=-1)

    outputs = []
    for model in _stock_and_unpadded_models():
        cache = DynamicCache()
        with torch.no_grad():
            prefill = model(
                input_ids, attention_mask=attention_mask, past_key_values=cache
            ).logits
            # q_len != kv_len takes the small-4D-mask fallback.
            step = model(
                chunk, attention_mask=chunk_mask, past_key_values=cache
            ).logits
        outputs.append((prefill[attention_mask.bool()], step))

    (ref_prefill, ref_step), (prefill, step) = outputs
    torch.testing.assert_close(prefill, ref_prefill, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(step, ref_step, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("selective", (False, True))
def test_compressed_generation_matches_stock_sdpa(selective):
    input_ids, attention_mask = _left_padded_batch(seed=1)
    config = CompressedCacheConfig(
        key=XKVCacheConfig(
            layer_group_size=2,
            num_layers=NUM_LAYERS,
            svd_backend="linalg",
            compression_ratio=2.0,
        ),
        value=MLPValueCacheConfig(target_compression_ratio=2.0, num_epochs=5),
        selective=SelectiveCacheConfig(
            enabled=selective,
            token_budget=24,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=2,
        ),
    )

    logits = []
    for model in _stock_and_unpadded_models():
        handles = install_selective_attention(model) if selective else []
        cache = CompressedCache(
            config=config,
            cache_context={"padding_mask": attention_mask},
            verbose=False,
        )
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                max_new_tokens=4,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=0,
            )
        for handle in handles:
            handle.remove()
        logits.append(torch.stack(output.logits))

    torch.testing.assert_close(logits[1], logits[0], atol=1e-4, rtol=1e-4)
