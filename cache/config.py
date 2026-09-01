from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch

from .rope import SharedRopeCache


@dataclass(frozen=True)
class BaselineCacheConfig:
    cache_type: Literal["baseline"] = field(default="baseline", init=False)


@dataclass(frozen=True)
class XKVCacheConfig:
    layer_group_size: int = 2
    num_layers: int | None = None
    svd_backend: str = "cholqr"
    compression_ratio: float = 2.0
    quantise_a: bool = False
    quantise_b: bool = False
    compressor_bits: int = 4
    cache_type: Literal["xkv"] = field(default="xkv", init=False)


@dataclass(frozen=True)
class TurboQuantCacheConfig:
    compressor_bits: int = 4
    cache_type: Literal["turboquant"] = field(
        default="turboquant",
        init=False,
    )


@dataclass(frozen=True)
class MLPValueCacheConfig:
    target_compression_ratio: float
    num_epochs: int = 5
    meta_weights_path: str | None = None
    value_mlp_weights_path: str | None = None
    use_residual: bool = False
    linear_weights: (
        list[torch.Tensor] | Callable[[], list[torch.Tensor]] | None
    ) = None
    turboquant_residuals: bool = False
    compressor_bits: int = 3
    cache_type: Literal["mlp"] = field(default="mlp", init=False)


@dataclass(frozen=True)
class SelectiveCacheConfig:
    enabled: bool = False
    token_budget: int = 2048
    chunk_size: int = 8
    local_tokens: int = 32
    outlier_chunks: int = 48


KeyCacheConfig = BaselineCacheConfig | XKVCacheConfig | TurboQuantCacheConfig
ValueCacheConfig = (
    BaselineCacheConfig
    | XKVCacheConfig
    | TurboQuantCacheConfig
    | MLPValueCacheConfig
)


@dataclass(frozen=True)
class CompressedCacheConfig:
    key: KeyCacheConfig = field(default_factory=BaselineCacheConfig)
    value: ValueCacheConfig = field(default_factory=BaselineCacheConfig)
    selective: SelectiveCacheConfig = field(
        default_factory=SelectiveCacheConfig
    )
    eviction_keep_ratio: float = 1.0


def _print_backend(config, verbose: bool) -> None:
    if verbose:
        print(f"Loading cache type {config.cache_type}")


def build_key_cache(
    config: KeyCacheConfig,
    *,
    ddp_cache_data: Iterable[torch.Tensor] | None,
    rope_cache: SharedRopeCache,
    verbose: bool,
):
    from .backends.tensor import SingleTensorCache
    from .backends.turboquant import TurboQuantCache
    from .backends.xkv import XKVKeysCache

    _print_backend(config, verbose)
    if isinstance(config, BaselineCacheConfig):
        return SingleTensorCache(
            ddp_cache_data=ddp_cache_data,
            rope_cache=rope_cache,
        )
    if isinstance(config, XKVCacheConfig):
        return XKVKeysCache(
            ddp_cache_data=ddp_cache_data,
            layer_group_size=config.layer_group_size,
            num_layers=config.num_layers,
            xkv_svd_backend=config.svd_backend,
            comp_ratio=config.compression_ratio,
            quantise_a=config.quantise_a,
            quantise_b=config.quantise_b,
            compressor_bits=config.compressor_bits,
            rope_cache=rope_cache,
        )
    if isinstance(config, TurboQuantCacheConfig):
        return TurboQuantCache(
            ddp_cache_data=ddp_cache_data,
            compressor_bits=config.compressor_bits,
            rope_cache=rope_cache,
        )
    raise TypeError(f"Unsupported key cache config: {type(config).__name__}")


def build_value_cache(
    config: ValueCacheConfig,
    *,
    ddp_cache_data: Iterable[torch.Tensor] | None,
    rope_cache: SharedRopeCache,
    verbose: bool,
):
    from .backends.mlp_values import MLPValueCache
    from .backends.tensor import SingleTensorCache
    from .backends.turboquant import TurboQuantCache
    from .backends.xkv import XKVKeysCache

    _print_backend(config, verbose)
    if isinstance(config, BaselineCacheConfig):
        return (
            SingleTensorCache(
                ddp_cache_data=ddp_cache_data,
                rope_cache=rope_cache,
            ),
            True,
        )
    if isinstance(config, MLPValueCacheConfig):
        return (
            MLPValueCache(
                ddp_cache_data=ddp_cache_data,
                target_cr=config.target_compression_ratio,
                num_epochs=config.num_epochs,
                meta_weights_path=config.meta_weights_path,
                value_mlp_weights_path=config.value_mlp_weights_path,
                use_residual=config.use_residual,
                W_linear_per_layer=config.linear_weights,
                turboquant_residuals=config.turboquant_residuals,
                compressor_bits=config.compressor_bits,
                rope_cache=rope_cache,
            ),
            True,
        )
    if isinstance(config, TurboQuantCacheConfig):
        return (
            TurboQuantCache(
                ddp_cache_data=ddp_cache_data,
                compressor_bits=config.compressor_bits,
                rope_cache=rope_cache,
            ),
            True,
        )
    if isinstance(config, XKVCacheConfig):
        cache = XKVKeysCache(
            ddp_cache_data=ddp_cache_data,
            layer_group_size=config.layer_group_size,
            num_layers=config.num_layers,
            xkv_svd_backend=config.svd_backend,
            comp_ratio=config.compression_ratio,
            quantise_a=config.quantise_a,
            quantise_b=config.quantise_b,
            compressor_bits=config.compressor_bits,
            rope_cache=rope_cache,
        )
        cache.unrope_keys = False
        return cache, False
    raise TypeError(f"Unsupported value cache config: {type(config).__name__}")


def _build_xkv_config(args, num_layers: int) -> XKVCacheConfig:
    return XKVCacheConfig(
        compression_ratio=args.comp_ratio,
        layer_group_size=args.xkv_layer_group_size,
        svd_backend=args.xkv_svd_backend,
        num_layers=num_layers,
        quantise_a=args.k_quantise_a,
        quantise_b=args.k_quantise_b,
        compressor_bits=args.k_compressor_bits,
    )


def build_key_cache_config(args, num_layers: int) -> KeyCacheConfig:
    if args.k_cache_type == "xkv":
        return _build_xkv_config(args, num_layers)
    if args.k_cache_type == "turboquant":
        return TurboQuantCacheConfig(compressor_bits=args.k_compressor_bits)
    return BaselineCacheConfig()


def build_value_cache_config(args, model, num_layers: int) -> ValueCacheConfig:
    if args.v_cache_type == "mlp":
        from utils.model import extract_kv_linear_init

        return MLPValueCacheConfig(
            target_compression_ratio=args.target_cr,
            num_epochs=args.num_epochs,
            meta_weights_path=args.meta_weights_path,
            value_mlp_weights_path=args.value_mlp_weights_path,
            use_residual=args.use_residual,
            linear_weights=(
                extract_kv_linear_init(model) if args.use_residual else None
            ),
            turboquant_residuals=args.v_turboquant_residuals,
            compressor_bits=args.v_compressor_bits,
        )
    if args.v_cache_type == "turboquant":
        return TurboQuantCacheConfig(compressor_bits=args.v_compressor_bits)
    if args.v_cache_type == "xkv":
        return _build_xkv_config(args, num_layers)
    return BaselineCacheConfig()


def build_cache_config(args, model) -> CompressedCacheConfig:
    """Assemble the full cache config from parsed CLI arguments."""
    num_layers = model.config.num_hidden_layers
    return CompressedCacheConfig(
        key=build_key_cache_config(args, num_layers),
        value=build_value_cache_config(args, model, num_layers),
        selective=SelectiveCacheConfig(enabled=args.selective_reconstruction),
        eviction_keep_ratio=args.eviction_keep_ratio,
    )
