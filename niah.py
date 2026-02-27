import argparse
import torch
from datasets import load_from_disk
from cache import CompressedCache
from utils import (
    Logger,
    get_model_and_tokenizer,
)

def make_hook(logger, uncompressed_window=0):
    """
    :param logger: persistent logger object to record metrics
    :param uncompressed_window: number of initial tokens to keep uncompressed
        (to get a more accurate estimate of initial perplexity)
        NOTE: this doesn't do anything for non-surprise-based methods
    """

    def hook(module, args, kwargs, output):
        input_ids = kwargs["input_ids"]
        seq_len = input_ids.size(-1)
        pkv = output.past_key_values
        if seq_len > 1:
            nll = output.loss
            logits = output.logits
            if nll is None:
                nll = module.loss_function(
                    logits=logits,
                    labels=input_ids,
                    vocab_size=module.config.vocab_size,
                    **kwargs,
                )
            ppl = torch.exp(nll)
            print(
                f"Prefill: nll={nll.item():.1f}, ppl={ppl.item():.1f}",
                f", seq_len={seq_len}",
            )
            if hasattr(pkv, "update_events"):
                if uncompressed_window > 0:
                    label_ed = -uncompressed_window
                else:
                    label_ed = None
                pkv.update_events(
                    logits[..., : -1 - uncompressed_window, :],
                    input_ids[..., 1:label_ed],
                )
        else:
            if hasattr(pkv, "comp_ratio") and not logger.recorded_cr:
                cr = pkv.comp_ratio
                if cr is not None:
                    print(f"Compression ratio: {cr:.2f}")
                    logger.add_log("crs", cr)
                    logger.recorded_cr = True

    return hook


def register_hooks(model):
    logger = Logger()
    model.register_forward_hook(make_hook(logger), with_kwargs=True)
    return model, logger


@torch.no_grad()
def main(
    model_name: str,
    dataset: str,
    n_samples: int,
    key_cache_kwargs: dict,
    value_cache_kwargs: dict,
    max_new_tokens: int,
):
    """Evaluate NIAH retrieval accuracy with a custom compressed KV cache setup.

    Args:
        model_name: Hugging Face model ID or local model path.
        dataset: Path to a disk-backed NIAH dataset.
        n_samples: Number of dataset samples to evaluate.
        key_cache_kwargs: Dictionary of key-cache compression parameters passed to CompressedCache.
        value_cache_kwargs: Dictionary of value-cache compression parameters passed to CompressedCache.
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        Tuple of `(success_rate, (compression_ratio_mean, compression_ratio_std))`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    model, logger = register_hooks(model)

    ds = load_from_disk(dataset)
    n_samples = min(n_samples, len(ds))
    print(f"Testing NIAH over {n_samples} samples.")
    n_correct = 0
    for i, batch in enumerate(ds):
        logger.recorded_cr = False

        prompt = batch["prompt"]
        answer = batch["answer"]

        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

        past_key_values = CompressedCache(
            config=model.config,
            key_cache_kwargs=key_cache_kwargs,
            value_cache_kwargs=value_cache_kwargs,
        )

        out = model.generate(
            input_ids,
            labels=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=0,
            max_new_tokens=max_new_tokens,
        )

        print(f"Answer: {answer}")
        input_len = input_ids.size(-1)
        output = tokenizer.decode(out[:, input_len:], skip_special_tokens=True)
        output = output[0]
        correct = answer in output
        n_correct += 1 if correct else 0
        print(f"Output: {output}", f"{'\u2705' if correct else '\u274C'}")
        if i >= n_samples - 1:
            print(f"Finished testing on {n_samples}. Stopping.")
            break
    success_rate = n_correct / n_samples
    print(f"Success rate: {success_rate * 100:.1f}%")
    cr_avg, cr_std = logger.get_log_mean("crs", std=True)
    print(f"Compression ratio: {cr_avg:.2f}+-{cr_std:.2f}")
    return success_rate, (cr_avg, cr_std)


def get_parser():
    parser = argparse.ArgumentParser(description="Training script for LLM.")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",  # "/home/ma-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="data/NIAH/multi-keys/1k"
    )
    parser.add_argument("-n", "--n_samples", type=int, default=100)

    parser.add_argument("--max_new_tokens", type=int, default=4)

    # key cache
    parser.add_argument("--k_cache_type", type=str, default="surprise_lr")
    parser.add_argument("--decomposition_method", type=str, default="svd", choices=["svd", "lora"])
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
    parser.add_argument("-e", "--energy_threshold", type=float, default=0.95)
    parser.add_argument("--rank_selection", type=str, default="comp_ratio", choices=["comp_ratio", "energy"])
    parser.add_argument("--k_lr", type=float, default=1e-2)
    parser.add_argument("--n_iter", type=int, default=3)

    # value cache
    parser.add_argument("--v_cache_type", type=str, default="mlp")
    parser.add_argument("--num_layers_per_mlp", type=int, nargs="+", default=[2]*16)
    parser.add_argument("--hidden_factors_per_mlp", type=int, nargs="+", default=[1]*16)
    parser.add_argument("--num_heads_per_mlp", type=int, nargs="+", default=[1]*16)
    parser.add_argument("--per_sequence", action="store_true")
    parser.add_argument("--target_perc", type=int, nargs="+", default=[100]*8+[85]*8)
    parser.add_argument("--target_model_num_heads", type=int, default=8)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_func", type=str, default="mse")
    parser.add_argument("--num_epochs", type=int, default=1)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    key_cache_kwargs = {
        "cache_type": args.k_cache_type,
        "decomposition_method": args.decomposition_method,
        "comp_ratio": args.comp_ratio,
        "energy_threshold": args.energy_threshold,
        "rank_selection": args.rank_selection,
        "lr": args.k_lr,
        "n_iter": args.n_iter,
        "gamma": 3.0,
        "min_size": 8.0,
    }

    value_cache_kwargs = {
        "cache_type": args.v_cache_type,
        "num_layers_per_mlp": args.num_layers_per_mlp,
        "hidden_factors_per_mlp": args.hidden_factors_per_mlp,
        "num_heads_per_mlp": args.num_heads_per_mlp,
        "per_sequence": args.per_sequence,
        "target_perc": args.target_perc,
        "target_model_num_heads": args.target_model_num_heads,
        "lr": args.v_lr,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "optimizer": args.optimizer,
        "loss_func": args.loss_func,
        "num_epochs": args.num_epochs,
    }
    main(
        model_name=args.model_name,
        dataset=args.dataset,
        n_samples=args.n_samples,
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        max_new_tokens=args.max_new_tokens,
    )
