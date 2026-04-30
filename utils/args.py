import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected. Got {v}, type: {type(v)}"
        )


def list_of_strings(arg):
    return arg.split(",")


def list_of_floats(arg):
    return [float(x) for x in list_of_strings(arg)]


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["false", "true"].index(str(x).lower()))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


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
