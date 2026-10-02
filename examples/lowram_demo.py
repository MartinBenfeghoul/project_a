"""An end-to-end LowRAM example."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method", choices=["baseline", "lowram", "fast"], default="lowram"
    )
    parser.add_argument(
        "--meta_weights",
        help="Meta init for fast. Default: checkpoints/meta_init/<model>/.",
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    import torch
    from cache import CompressedCache
    from benchmarks.runtime import (
        load_model,
        init_weights,
        cache_config,
        attention_mode,
    )
    from benchmarks.tasks import retrieval_examples, score_answer
    from utils.model import extract_kv_linear_init, get_device

    meta_weights, value_weights = init_weights(
        args.model, args.method, args.meta_weights
    )
    torch.manual_seed(42)
    model, tokenizer = load_model(args.model)
    example = retrieval_examples(tokenizer, args.context, samples=1, seed=42)[0]
    inputs = torch.tensor([example["input_ids"]], device=get_device(model))
    mask = torch.ones_like(inputs)
    config = cache_config(
        model,
        args.method,
        args.ratio,
        args.steps,
        meta_weights,
        extract_kv_linear_init(model) if args.method != "baseline" else None,
        value_weights=value_weights,
    )
    cache = CompressedCache(
        config=config, cache_context={"padding_mask": mask}, verbose=False
    )
    with attention_mode(model, args.method), torch.no_grad():
        output = model.generate(
            inputs,
            attention_mask=mask,
            past_key_values=cache,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            logits_to_keep=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(
        output[0, inputs.shape[1] :], skip_special_tokens=True
    )
    print("Question:", example["question"])
    print("Reference:", ", ".join(example["references"]))
    print("Answer:", answer)
    print("Fact recall:", score_answer(answer, example["references"])["score"])


if __name__ == "__main__":
    main()
