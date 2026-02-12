import argparse
import numpy as np
import torch
from datasets import load_from_disk

from utils import (
    KEY_CACHE_CLASSES,
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


@torch.no_grad()
def main(
    model_name: str,
    dataset: str,
    n_samples: int,
    cache_type: str,
    comp_ratio: float,
    energy_threshold: float,
    rank_selection: str,
    max_new_tokens: int,
):
    """ThE cOdE iS tHe DoCsTrInG - Fredericoco 2026"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    eos_id = tokenizer.eos_token_id

    ds = load_from_disk(dataset)
    n_samples = min(n_samples, len(ds))
    print(f"Testing NIAH over {n_samples} samples.")
    n_correct = 0
    crs = []
    for i, batch in enumerate(ds):
        prompt = batch["prompt"]
        answer = batch["answer"]

        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, device=device
        )["input_ids"].to(device)

        _, T = input_ids.shape
        cache_position = torch.arange(T, device=device)

        cache = get_cache(cache_type)
        past_key_values = cache(
            config=model.config,
            comp_ratio=comp_ratio,
            energy_threshold=energy_threshold,
            rank_selection=rank_selection,
            n_iter=8,
            gamma=3.0,
            min_size=8.0,
        )

        # Prefill
        out = model(
            input_ids,
            labels=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        nll = out.loss
        ppl = torch.exp(out.loss)
        print(
            f"Prefill: nll={nll.item():.1f}, ppl={ppl.item():.1f}, seq_len={T}"
        )
        if hasattr(past_key_values, "update_events"):
            past_key_values.update_events(out.logits, input_ids)

        output_ids = []
        input_id = input_ids[..., -1:]  # (B, 1)
        cache_position = cache_position[
            ..., -1:
        ]  # (B, 1) or (1,) depending on how you built it
        for j in range(max_new_tokens):
            out = model(
                input_ids=input_id,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if j == 0:
                cr = past_key_values.comp_ratio
                print(f"Compression ratio: {cr:.2f}")
                crs.append(cr)

            logits = out.logits[:, -1, :]  # (B, V)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            output_ids.append(next_id)

            input_id = next_id
            cache_position = cache_position + 1

            if eos_id is not None:
                if (next_id == eos_id).all():
                    print(f"EOS detected. Stopping.")
                    break

        gen_ids = torch.cat(output_ids, dim=-1)
        print(f"Answer: {answer}")
        output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        correct = answer in output
        n_correct += 1 if correct else 0
        print(f"Output: {output}", f"{'\u2705' if correct else '\u274C'}")
        if i >= n_samples - 1:
            print(f"Finished testing on {n_samples}. Stopping.")
            break
    success_rate = n_correct / n_samples
    print(f"Success rate: {success_rate * 100:.1f}%")
    cr_avg = np.mean(crs)
    cr_std = np.std(crs)
    print(f"Compression ratio: {cr_avg:.2f}+-{cr_std:.2f}")
    return success_rate, (cr_avg, cr_std)


def get_parser():
    parser = argparse.ArgumentParser(description="Training script for LLM.")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="/home/ma-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="data/NIAH/multi-keys/1k"
    )
    parser.add_argument("-c", "--cache_type", type=str, default="surprise_lr")
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
    parser.add_argument("-e", "--energy_threshold", type=float, default=0.95)
    parser.add_argument(
        "--rank_selection", type=str, default="comp_ratio"  # comp_ratio, energy
    )
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    main(**vars(args))
