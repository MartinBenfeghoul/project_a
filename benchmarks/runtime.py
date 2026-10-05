from contextlib import contextmanager
from pathlib import Path

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def load_model(model_name, dtype="bfloat16"):
    import torch
    from utils.model import get_model_and_tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible. Check nvidia-smi and your execution environment."
        )
    model, tokenizer = get_model_and_tokenizer(
        model_name, torch_dtype=getattr(torch, dtype)
    )
    if model.config.model_type not in ("llama", "mistral"):
        raise ValueError(
            "This demo currently supports Llama and Mistral architectures."
        )
    return model, tokenizer


def init_weights(model_name, method, meta_weights=None):
    name = Path(model_name).name
    if method == "fast":
        path = Path(
            meta_weights or CHECKPOINTS / "meta_init" / name / "meta_mlps.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path, None
    if method == "lowram":
        path = CHECKPOINTS / "value_mlps" / name / "value_mlps.pt"
        if not path.is_file():
            import gc
            import torch
            import train_value_mlps

            print(f"No value MLPs at {path}; training them first.")
            train_value_mlps.train(
                train_value_mlps.build_arg_parser().parse_args(
                    ["--model_name", model_name, "--output_path", str(path)]
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return None, path
    return None, None


def cache_config(
    model,
    method,
    ratio=4,
    steps=150,
    meta_weights=None,
    linear_weights=None,
    svd_backend="cholqr",
    token_budget=2048,
    value_weights=None,
):
    from cache import (
        CompressedCacheConfig,
        XKVCacheConfig,
        MLPValueCacheConfig,
        SelectiveCacheConfig,
    )
    from model.meta_learning import LearnedInit

    if method == "baseline":
        return CompressedCacheConfig()
    if method == "fast" and not meta_weights:
        raise ValueError(
            "The fast preset requires --meta-weights from this model."
        )
    init_path = {"fast": meta_weights, "lowram": value_weights}.get(method)
    learned_init = (
        LearnedInit.from_checkpoint(str(init_path)) if init_path else None
    )

    return CompressedCacheConfig(
        key=XKVCacheConfig(
            compression_ratio=ratio,
            layer_group_size=4,
            num_layers=model.config.num_hidden_layers,
            svd_backend=svd_backend,
        ),
        value=MLPValueCacheConfig(
            target_compression_ratio=ratio,
            num_epochs=2 if method == "fast" else steps,
            learned_init=learned_init,
            use_residual=True,
            linear_weights=linear_weights,
        ),
        selective=SelectiveCacheConfig(
            enabled=method == "fast", token_budget=token_budget
        ),
    )


@contextmanager
def attention_mode(model, method):
    from model.selective_attention import install_selective_attention

    handles = install_selective_attention(model) if method == "fast" else []
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
