import itertools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import (
    KEY_CACHE_CLASSES,
    PackedTokens,
    load_data,
    collate,
    get_model_and_tokenizer,
)


def measure_perplexity(logits, target):
    nll = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    ppl = torch.exp(nll)
    return nll, ppl


def get_cache(cache_type):
    if cache_type not in KEY_CACHE_CLASSES:
        raise ValueError(
            f"{cache_type} not in KEY_CACHE_CLASSES. Please select one of the following: {KEY_CACHE_CLASSES.keys()}"
        )
    print(f"Loading cache type {cache_type}")
    return KEY_CACHE_CLASSES[cache_type]


def main(
    model_name,
    dataset,
    seq_len: int = 1024,
    micro_bs: int = 4,
    cache_type: str = "low_rank",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    if not os.path.exists("model_inputs.pt"):
        ds = load_data(dataset)
        packed = PackedTokens(
            ds,
            tokenizer,
            seq_len=seq_len,
            eos_id=tokenizer.eos_token_id,
            buffer_tokens=2 * micro_bs * seq_len,
        )
        dl = DataLoader(
            packed,
            batch_size=micro_bs,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        batches = list(itertools.islice(dl, micro_bs))
        inputs = batches[0]  # Take first batch only for saving
    else:
        inputs = torch.load("model_inputs.pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    cache_position = torch.arange(
        inputs["input_ids"].shape[1], device=model.device
    )

    cache = get_cache(cache_type)
    past_key_values = cache(
        config=model.config,
        comp_ratio=1.5,
        n_iter=8,
        gamma=3.0,
        min_size=8.0,
    )

    pos = 512
    chunk_input_ids = inputs["input_ids"][..., :pos]
    chunk_position = cache_position[:pos]
    out = model(
        chunk_input_ids,
        labels=chunk_input_ids,
        past_key_values=past_key_values,
        cache_position=chunk_position,
        use_cache=True,
    )
    past_key_values = out.past_key_values
    nll = out.loss
    ppl = torch.exp(out.loss)
    print(
        f"Prefill over {pos} positions, nll={nll.item():.1f}, ppl={ppl.item():.1f}"
    )
    past_key_values.update_events(
        out.logits, inputs["input_ids"][..., 1 : pos + 1]
    )

    ppls = []
    for _ in range(16):
        out = model(
            inputs["input_ids"][..., pos : pos + 1],
            past_key_values=past_key_values,
            cache_position=cache_position[pos : pos + 1],
            use_cache=True,
        )
        past_key_values = out.past_key_values
        target = inputs["input_ids"][:, pos + 1]  # shape (B,)
        logits = out.logits[:, -1, :]  # shape (B, V)
        nll, ppl = measure_perplexity(logits, target)
        ppls.append(ppl.item())
        cr = past_key_values.comp_ratio
        print(
            f"pos={pos}, CR={cr:.1f}, nll={nll.item():.1f}, ppl={ppl.item():.1f}"
        )

        pos += 1
    print(f"Average ppl={np.mean(ppls):.1f}+-{np.std(ppls):.1f}")


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # "/home/ma-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    dataset = "HuggingFaceFW/fineweb-edu"  # "example_dataset", "HuggingFaceFW/fineweb-edu",
    cache_type = "surprise_lr"  # low_rank, surprise_lr

    main(model_name, dataset, cache_type=cache_type)
