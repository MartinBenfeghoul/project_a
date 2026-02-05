import argparse
import torch
from datasets import load_from_disk

from utils import (
    CACHE_CLASSES,
    get_model_and_tokenizer,
)

def measure_perplexity(logits, target):
    nll = torch.nn.functional.cross_entropy(
        logits, target, reduction="mean"
    )
    ppl = torch.exp(nll)
    return nll, ppl

def get_cache(cache_type):
    if cache_type not in CACHE_CLASSES:
        raise ValueError(f"{cache_type} not in CACHE_CLASSES. Please select one of the following: {CACHE_CLASSES.keys()}")
    print(f"Loading cache type {cache_type}")
    return CACHE_CLASSES[cache_type]

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(args.model_name, device)
    eos_id = tokenizer.eos_token_id

    ds = load_from_disk(args.dataset)
    n_samples = min(args.n_samples, len(ds))
    print(f"Testing NIAH over {n_samples} samples.")
    n_correct = 0
    for i, batch in enumerate(ds):
        prompt = batch['prompt']
        answer = batch['answer']

        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )['input_ids']

        B, T = input_ids.shape
        cache_position = torch.arange(
            T, device=device
        )

        cache = get_cache(args.cache_type)
        past_key_values = cache(
            config=model.config,
            comp_ratio=args.compression_ratio,
            niter=8,
            gamma=3.0,
            min_size=8.0,
        )

        # Prefill
        out = model(
            input_ids, 
            labels=input_ids,
            past_key_values=past_key_values, 
            cache_position=cache_position, 
            use_cache=True
        )
        past_key_values = out.past_key_values
        nll = out.loss
        ppl= torch.exp(out.loss)
        print(
            f"Prefill: nll={nll.item():.1f}, ppl={ppl.item():.1f}, seq_len={T}"
        )
        if hasattr(past_key_values, 'update_events'):
            print("Updating events within the KV cache.")
            past_key_values.update_events(
                out.logits, input_ids
            )

        output_ids = []
        input_id = input_ids[..., -1:]                 # (B, 1)
        cache_position = cache_position[..., -1:]       # (B, 1) or (1,) depending on how you built it
        for _ in range(args.max_new_tokens):
            out = model(
                input_ids=input_id,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = out.past_key_values

            logits = out.logits[:, -1, :]              # (B, V)
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
        print(
            f"Output: {output}",
            f"{'\u2705' if correct else '\u274C'}"
        )
        if i >= n_samples - 1:
            print(f"Finished testing on {n_samples}. Stopping.")
            break
    success_rate = n_correct / n_samples
    print(f"Success rate: {success_rate * 100:.1f}%")
    return success_rate

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LLM.")
    parser.add_argument(
        "-m", "--model_name", type=str, default="/home/ma-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="data/NIAH/multi-keys/1k"
    )
    parser.add_argument(
        "-c", "--cache_type", type=str, default="surprise_svd"
    )
    parser.add_argument(
        "-r", "--compression_ratio", type=float, default=2.0
    )
    parser.add_argument(
        "-n", "--n_samples", type=int, default=100
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = parse_args()
    main(args)
