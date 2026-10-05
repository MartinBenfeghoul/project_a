def list_of_strings(arg):
    return arg.split(",")

def parse_layers(layer_spec: str | None, num_layers: int) -> list[int]:
    if layer_spec is None or layer_spec == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))

    bad = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range: {bad}")
    return sorted(layers)


def add_common_training_args(
    parser,
    *,
    model_name: str,
    seq_len: int,
    max_batches: int,
    lr: float = 1e-3,
    batch_size: int | None = None,
):
    """Add the arguments every training script shares.

    Defaults are per-script, so each caller passes its own.
    """
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--seq_len", type=int, default=seq_len)
    parser.add_argument("--max_batches", type=int, default=max_batches)
    parser.add_argument("--lr", type=float, default=lr)
    if batch_size is not None:
        parser.add_argument("--batch_size", type=int, default=batch_size)
    return parser
