"""Compression-ratio accounting across the key and value backends."""

from dataclasses import fields, is_dataclass

import torch

from numerics.quantisation import (
    MSECompressor, TurboQuantFactor, factor_nbytes, get_turboquant_compressor,
)


def _prefill_retention(cache, padding_mask):
    mask = padding_mask.bool()
    policy = getattr(cache, "eviction", None)
    positions = {} if policy is None else policy.kept_positions
    force = None
    if policy is not None and policy.enabled:
        force = policy._forced_positions(mask.shape[1], mask, mask.device)
    counts = {}
    for idx in range(len(cache.key_cache.layers)):
        kept = positions.get(idx)
        selected = mask
        selected_force = force
        if kept is not None:
            kept = kept.to(mask.device)
            selected = mask.index_select(1, kept)
            if force is not None:
                selected_force = force.index_select(0, kept)
        counts[idx] = {
            "valid": int(selected.sum().item()),
            "rows": selected.numel(),
            "forced": (
                int((selected & selected_force[None, :]).sum().item())
                if selected_force is not None else 0
            ),
        }
    return counts


def prefill_storage(cache, padding_mask) -> dict:
    valid_tokens = int(padding_mask.sum().item())
    original_bytes = 0
    padding_bytes = 0
    retention = _prefill_retention(cache, padding_mask)
    payload = []
    quantizers = []
    metadata = []
    policy = getattr(cache, "eviction", None)
    if policy is not None:
        metadata.extend([policy.kept_positions, policy._group_keep_positions,
                         policy._value_importance])
    for backend in (cache.key_cache, cache.value_cache):
        for idx, layer in enumerate(backend.layers):
            quantizers.append(getattr(layer, "compressor", None))
            metadata.append(getattr(layer, "rope_positions", None))
            tensor = layer.tensor
            original_bytes += (
                valid_tokens * tensor.shape[1] * tensor.shape[-1]
                * tensor.element_size()
            )
            payload.append(tensor)
            counts = retention[idx]
            invalid = counts["rows"] - counts["valid"]
            if tensor.shape[-2]:
                assert tensor.shape[0] * tensor.shape[-2] == counts["rows"]
                padding_bytes += (
                    invalid * tensor.shape[1] * tensor.shape[-1]
                    * tensor.element_size()
                )
            params = getattr(layer, "compressed_params", None)
            if params is not None:
                encoded_bytes = (
                    params.indices.numel() * params.indices.element_size()
                    + params.norms.numel() * params.norms.element_size()
                )
                padding_bytes += encoded_bytes // counts["rows"] * invalid
            for name in ("compressed_params", "indices", "value_residuals"):
                payload.append(getattr(layer, name, None))
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                payload.extend(mlp.parameters())
        for group in getattr(backend, "group_states", {}).values():
            metadata.append(getattr(group, "valid_lens", None))
            payload.append(group.packed_shared)
            counts = retention[group.layer_indices[0]]
            padding_bytes += (
                factor_nbytes(group.packed_shared) // counts["rows"]
                * (counts["rows"] - counts["valid"])
            )
        for state in getattr(backend, "layer_states", {}).values():
            payload.append(state.packed_right)
    payload.extend(cache.selective.layers.values())

    storages = {}

    def visit(value, gpu_only=False):
        if isinstance(value, torch.Tensor):
            if gpu_only and value.device.type != "cuda":
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, gpu_only)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item, gpu_only)
        elif isinstance(value, MSECompressor):
            visit(vars(value), gpu_only)
        elif is_dataclass(value) and not isinstance(value, type):
            if isinstance(value, TurboQuantFactor):
                quantizers.append(get_turboquant_compressor(
                    value.params.shape[-1], value.bits, value.params.indices.device
                ))
            for field in fields(value):
                visit(getattr(value, field.name), gpu_only)

    for value in payload:
        visit(value)
    representation_bytes = sum(storages.values())
    visit(metadata, gpu_only=True)
    eviction_metadata_bytes = sum(storages.values()) - representation_bytes
    before_quantizers = sum(storages.values())
    visit(quantizers, gpu_only=True)
    quantizer_buffer_bytes = sum(storages.values()) - before_quantizers
    physical_bytes = sum(storages.values())
    compressed_bytes = physical_bytes - padding_bytes
    original_layer_tokens = valid_tokens * len(retention)
    retained_layer_tokens = sum(c["valid"] for c in retention.values())
    return {
        "num_sequences": int(padding_mask.shape[0]),
        "num_valid_tokens": valid_tokens,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "physical_compressed_bytes": physical_bytes,
        "discounted_padding_bytes": padding_bytes,
        "eviction_metadata_bytes": eviction_metadata_bytes,
        "quantizer_buffer_bytes": quantizer_buffer_bytes,
        "physical_compression_ratio": (
            original_bytes / physical_bytes if physical_bytes else None
        ),
        "compression_ratio": (
            original_bytes / compressed_bytes if compressed_bytes else None
        ),
        "eviction": {
            "original_token_layer_count": original_layer_tokens,
            "retained_token_layer_count": retained_layer_tokens,
            "forced_token_layer_count": sum(c["forced"] for c in retention.values()),
            "actual_ratio": (
                original_layer_tokens / retained_layer_tokens
                if retained_layer_tokens else None
            ),
        },
    }


def _backend_comp_ratio(cache) -> float | None:
    calc = getattr(cache, "calc_compression_ratio", None)
    if not callable(calc):
        return None
    return calc() or None


def _selective_overhead_nbytes(selective, key_cache, rope_cache) -> int:
    nbytes = sum(
        state.key_overhead_nbytes + state.exact_value_nbytes
        for state in selective.layers.values()
    )
    nbytes += selective.scorer_nbytes
    nbytes += rope_cache.nbytes
    nbytes += getattr(key_cache, "selective_reconstruction_nbytes", 0)
    return nbytes


def compression_ratio(
    key_cache,
    value_cache,
    selective,
    rope_cache,
    eviction=None,
) -> float | None:
    """Combined compression ratio, or whichever side reports one."""
    eviction_ratio = 1.0 if eviction is None else eviction.compression_ratio
    key_cr = _backend_comp_ratio(key_cache)
    value_cr = _backend_comp_ratio(value_cache)

    if key_cr is None or value_cr is None:
        backend_cr = key_cr if key_cr is not None else value_cr
        return None if backend_cr is None else backend_cr * eviction_ratio

    comp_ratio = 2 / ((1 / key_cr) + (1 / value_cr))
    if not selective.layers:
        return comp_ratio * eviction_ratio

    original_bytes = 2 * sum(
        state.original_key_nbytes for state in selective.layers.values()
    )
    backend_bytes = original_bytes / eviction_ratio
    selective_bytes = _selective_overhead_nbytes(
        selective, key_cache, rope_cache
    )
    return original_bytes / ((backend_bytes / comp_ratio) + selective_bytes)
