import torch

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria

from .cache import CompressedCache


class CompressedCacheHFLM(HFLM):
    """
    HFLM subclass that injects a new CompressedCache into every model call.
    Note: once CompressiveCache is implemented in transformers, subclassing won't be needed
    simply pass "cache_implementation": "compressive_cache" in generation kwargs
    """

    def __init__(
        self,
        key_cache_kwargs,
        value_cache_kwargs,
        eviction_keep_ratio,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._key_cache_kwargs = key_cache_kwargs
        self._value_cache_kwargs = value_cache_kwargs
        self._eviction_keep_ratio = eviction_keep_ratio

    def _make_cache(self, cache_context=None):
        return CompressedCache(
            key_cache_kwargs=self._key_cache_kwargs,
            value_cache_kwargs=self._value_cache_kwargs,
            cache_context=cache_context,
            eviction_keep_ratio=self._eviction_keep_ratio,
            verbose=False,
        )

    def _model_call(self, inps, attn_mask=None, labels=None):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            cache = self._make_cache(
                {"padding_mask": torch.ones_like(inps, dtype=torch.bool)}
            )
            output = self.model(inps, past_key_values=cache)
            return output.logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        cache = self._make_cache(
            {"padding_mask": generation_kwargs["attention_mask"]}
        )
        generation_kwargs["past_key_values"] = cache
        generation_kwargs["temperature"] = generation_kwargs.get(
            "temperature", 0.0
        )
        do_sample = generation_kwargs.get("do_sample")

        if (
            temp := generation_kwargs.get("temperature")
        ) == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and temp == 0.0:
            generation_kwargs.pop("temperature", None)
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            output = self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs,
            )
        return output
