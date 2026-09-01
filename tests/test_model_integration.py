import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)

from cache.cache import CompressedCache
from cache.config import (
    BaselineCacheConfig,
    CompressedCacheConfig,
    XKVCacheConfig,
)


@pytest.mark.parametrize(
    ("model_cls", "config_cls", "extra_config"),
    [
        (LlamaForCausalLM, LlamaConfig, {}),
        (MistralForCausalLM, MistralConfig, {"sliding_window": 32}),
    ],
    ids=("llama", "mistral"),
)
def test_compressed_cache_prefill_and_decode(
    model_cls,
    config_cls,
    extra_config,
):
    torch.manual_seed(4)
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **extra_config,
    )
    model = model_cls(config).eval()
    cache = CompressedCache(
        config=CompressedCacheConfig(
            key=XKVCacheConfig(
                compression_ratio=2.0,
                layer_group_size=2,
                num_layers=2,
                svd_backend="linalg",
            ),
            value=BaselineCacheConfig(),
        ),
        verbose=False,
    )
    prompt = torch.randint(3, config.vocab_size, (1, 12))

    with torch.no_grad():
        prefill = model(prompt, past_key_values=cache, use_cache=True)
        decode = model(
            torch.randint(3, config.vocab_size, (1, 1)),
            past_key_values=cache,
            use_cache=True,
        )

    assert prefill.logits.shape == (1, 12, config.vocab_size)
    assert decode.logits.shape == (1, 1, config.vocab_size)
    assert torch.isfinite(decode.logits).all()
    assert cache.get_seq_length(0) == 13
    assert not cache.key_cache.prefill
