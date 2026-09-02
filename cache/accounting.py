"""Compression-ratio accounting across the key and value backends."""


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
