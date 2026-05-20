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
        logger,
        eviction_keep_ratio,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._key_cache_kwargs = key_cache_kwargs
        self._value_cache_kwargs = value_cache_kwargs
        self._eviction_keep_ratio = eviction_keep_ratio
        self._logger = logger
        self._previous_value_mlps = None

    def _make_cache(self, cache_context=None):
        value_cache_kwargs = dict(self._value_cache_kwargs)
        if value_cache_kwargs.get("prev_sample_init"):
            value_cache_kwargs["previous_sample_mlps"] = self._previous_value_mlps
        return CompressedCache(
            config=self.model.config,
            key_cache_kwargs=self._key_cache_kwargs,
            value_cache_kwargs=value_cache_kwargs,
            cache_context=cache_context,
            eviction_keep_ratio=self._eviction_keep_ratio,
            verbose=False,
        )

    def _remember_value_mlps(self, cache):
        if self._value_cache_kwargs.get("prev_sample_init"):
            get_mlps = getattr(cache.value_cache, "get_layer_mlps", None)
            if callable(get_mlps):
                self._previous_value_mlps = get_mlps()

    def generate_until(self, requests, disable_tqdm=False):
        self._current_task_name = requests[0].task_name if requests else None
        try:
            return super().generate_until(requests, disable_tqdm=disable_tqdm)
        finally:
            self._current_task_name = None

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
            cache = self._make_cache()
            output = self.model(inps, past_key_values=cache)
            self._remember_value_mlps(cache)
            return output.logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        self._logger.recorded_cr = False
        self._logger.recorded_k_timing = False
        task_name = getattr(self, "_current_task_name", None)
        cache_context = {"task_name": task_name} if task_name is not None else None
        cache = self._make_cache(cache_context)
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
        self._remember_value_mlps(cache)
        return output
