"""Align decode masks with evicted KV rows."""

import inspect

import torch


def _argument(args, kwargs, names, name):
    if name in kwargs:
        return kwargs[name]
    index = names.index(name) if name in names else len(args)
    return args[index] if index < len(args) else None


def install_eviction_attention_hooks(model):
    from .attention_predictor import (
        _attention_backend_specs,
        _get_attention_backend,
    )

    root_names = list(inspect.signature(model.forward).parameters)

    def capture(module, args, kwargs):
        cache = _argument(args, kwargs, root_names, "past_key_values")
        if getattr(cache, "eviction_keep_ratio", 1.0) < 1:
            cache.set_generation_attention_mask(
                _argument(args, kwargs, root_names, "attention_mask")
            )

    handles = [model.register_forward_pre_hook(capture, with_kwargs=True)]
    specs = _attention_backend_specs()
    for module in model.modules():
        spec = _get_attention_backend(module, specs)
        if spec is None:
            continue
        names = list(inspect.signature(spec["base_cls"].forward).parameters)[1:]

        def align(module, args, kwargs, names=names):
            cache = _argument(args, kwargs, names, "past_key_values")
            if cache is None:
                cache = _argument(args, kwargs, names, "past_key_value")
            if getattr(cache, "eviction_keep_ratio", 1.0) >= 1:
                return
            if module.layer_idx not in cache.kept_positions:
                return
            hidden = _argument(args, kwargs, names, "hidden_states")
            if hidden.shape[1] != 1:
                raise NotImplementedError(
                    "Evicted caches support single-token decode only."
                )
            if getattr(module.config, "sliding_window", None) is not None:
                raise NotImplementedError(
                    "Eviction mask alignment does not support sliding-window attention."
                )
            mask = cache.eviction_decode_mask(module.layer_idx)
            backend = module.config._attn_implementation
            if backend not in (
                "sdpa",
                "eager",
                "unpadded_sdpa",
                "flash_attention_2",
            ):
                raise NotImplementedError(
                    f"Eviction mask alignment is unsupported for {backend}."
                )
            if mask is not None:
                mask = mask.to(hidden.device)
                if backend in ("sdpa", "eager"):
                    mask = mask[:, None, None, :]
                if backend == "eager":
                    mask = torch.zeros_like(
                        mask, dtype=hidden.dtype
                    ).masked_fill(~mask, torch.finfo(hidden.dtype).min)
            index = names.index("attention_mask")
            if index < len(args):
                args = list(args)
                args[index] = mask
                return tuple(args), kwargs
            return args, {**kwargs, "attention_mask": mask}

        handles.append(
            module.register_forward_pre_hook(align, with_kwargs=True)
        )
    return handles
