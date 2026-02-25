import argparse
import random
import json
from pathlib import Path
from typing import List

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

DEFAULT_FILLER_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
    "Water is wet.",
    "Fire is hot.",
    "Time flies quickly.",
    "This is filler text.",
]


def generate_passkey_sample(
    tokenizer,
    seq_len: int = 256,
    key_depth: float = 0.5,
    filler_sentences: List[str] = None,
) -> dict:
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if filler_sentences is None:
        filler_sentences = DEFAULT_FILLER_SENTENCES

    key = "".join([str(random.randint(0, 9)) for _ in range(5)])

    intro = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorise it. I will quiz you about the important information there.\n"
    )
    key_sentence = (
        f"The pass key is {key}. Remember it. {key} is the pass key.\n"
    )

    body = []
    while True:
        body.append(random.choice(filler_sentences))
        tmp_prompt = intro + " ".join(body) + "\n"
        tokens = tokenizer(tmp_prompt + key_sentence, return_tensors="pt")[
            "input_ids"
        ].shape[-1]
        if tokens >= seq_len - 30:
            break

    insert_idx = min(int(len(body) * key_depth), len(body))
    body_with_key = body[:insert_idx] + [key_sentence] + body[insert_idx:]

    query = "What is the pass key? The pass key is"
    prompt = intro + " ".join(body_with_key) + "\n" + query

    return {"prompt": prompt, "answer": key}


def generate_passkey_dataset(
    output_path: str,
    tokenizer_name: str = "gpt2",
    num_samples: int = 1000,
    seq_length: int = 1024,
    depth: float = None,
    filler_sentences: List[str] = None,
    seed: int = 42,
):
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if filler_sentences is None:
        filler_sentences = DEFAULT_FILLER_SENTENCES

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            key = "".join([str(random.randint(0, 9)) for _ in range(5)])

            intro = (
                "There is an important info hidden inside a lot of irrelevant text. "
                "Find it and memorise it. I will quiz you about the important information there.\n"
            )
            key_sentence = (
                f"The pass key is {key}. Remember it. {key} is the pass key.\n"
            )

            body = []
            while True:
                body.append(random.choice(filler_sentences))
                tmp_prompt = intro + " ".join(body) + "\n"
                tokens = tokenizer(
                    tmp_prompt + key_sentence, return_tensors="pt"
                )["input_ids"].shape[-1]
                if tokens >= seq_length - 30:
                    break

            if depth is not None:
                insert_idx = min(int(len(body) * depth), len(body))
            else:
                insert_idx = random.randint(0, len(body))
            body_with_key = (
                body[:insert_idx] + [key_sentence] + body[insert_idx:]
            )

            query = "What is the pass key? The pass key is"
            prompt = intro + " ".join(body_with_key) + "\n" + query

            f.write(json.dumps({"prompt": prompt, "answer": key}) + "\n")

    print(f"Saved {num_samples} examples to {output_path}")


def generate_multi_passkey_dataset(
    output_path: str,
    tokenizer_name: str = "gpt2",
    num_samples: int = 100,
    seq_length: int = 1024,
    filler_text: str = None,
    num_keys: int = 3,  # Number of keys to hide
    num_queries: int = 2,  # Number of keys to ask for
    seed: int = 42,
):
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if filler_text is None or not isinstance(filler_text, str):
        raise ValueError("filler_text must be a single string")

    intro = (
        f"There are {num_keys} different pass keys hidden in the text below. "
        "Memorize all of them and their associated labels.\n"
    )

    # Pre-tokenize static parts
    intro_ids = tokenizer(intro, add_special_tokens=False)["input_ids"]
    filler_ids = tokenizer(filler_text, add_special_tokens=False)["input_ids"]

    # Estimate space for keys and queries (approximate)
    # Each key sentence: "\nKey [name] is [value].\n"
    reserved_tokens = 50 * num_keys + 50 * num_queries + len(intro_ids)
    max_filler_len = seq_length - reserved_tokens

    if max_filler_len <= 100:
        raise ValueError(
            f"Sequence length {seq_length} is too small for {num_keys} keys."
        )

    new_ds = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            # 1. Generate unique keys and values
            keys_data = []
            for i in range(num_keys):
                label = f"ALPHA_{i}"  # You can randomize these labels too
                val = "".join(str(random.randint(0, 9)) for _ in range(5))
                keys_data.append({"label": label, "value": val})

            # 2. Prepare filler slice
            start = random.randint(0, len(filler_ids) - max_filler_len)
            filler_slice_text = tokenizer.decode(
                filler_ids[start : start + max_filler_len]
            )

            # 3. Insert keys at random positions
            # We split the filler into chunks to ensure keys aren't all in one spot
            text_segments = []
            last_idx = 0
            # Get random insertion points in sorted order
            insert_points = sorted(
                [
                    random.randint(0, len(filler_slice_text))
                    for _ in range(num_keys)
                ]
            )

            current_text = ""
            for i, split_point in enumerate(insert_points):
                current_text += filler_slice_text[last_idx:split_point]
                current_text += f"\n[MEMORIZE] The pass key for {keys_data[i]['label']} is {keys_data[i]['value']}.\n"
                last_idx = split_point
            current_text += filler_slice_text[last_idx:]

            # 4. Select which keys to query
            queried_keys = random.sample(keys_data, num_queries)

            query_text = "\nBased on the text, what are the pass keys for the following?\n"
            answers = ""
            for q in queried_keys:
                query_text += f"- {q['label']}: "
                # answers[q['label']] = q['value']
                answers += q["value"] + " "

            # query_text += "\nAnswer in JSON format."

            prompt = intro + current_text + query_text

            entry = {
                "prompt": prompt,
                # "answer": json.dumps(answers)
                "answer": answers[:-1],
            }
            f.write(json.dumps(entry) + "\n")
            new_ds.append(entry)

    return Dataset.from_list(new_ds)


def parse_args():
    """
    Example usage:
    python passkey_data.py passkey --output out.jsonl --seq-length 4000 --depth 0.5
    python passkey_data.py multi --seq-length 4000 --filler-file data/PaulGrahamEssays.json
    """
    parser = argparse.ArgumentParser(description="Generate passkey datasets")
    subparsers = parser.add_subparsers(dest="task", required=True)

    p = subparsers.add_parser(
        "passkey", help="Single-key passkey retrieval dataset"
    )
    p.add_argument(
        "--output", type=str, required=True, help="Output .jsonl path"
    )
    p.add_argument(
        "--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Key depth (0.0–1.0); random if omitted",
    )
    p.add_argument("--seed", type=int, default=42)

    m = subparsers.add_parser(
        "multi", help="Multi-key passkey retrieval dataset"
    )
    m.add_argument(
        "--output", type=str, required=True, help="Output .jsonl path"
    )
    m.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    m.add_argument("--num-samples", type=int, default=100)
    m.add_argument("--seq-length", type=int, default=1024)
    m.add_argument(
        "--filler-file",
        type=str,
        required=True,
        help="Path to JSON file with filler text (key: 'text')",
    )
    m.add_argument("--num-keys", type=int, default=3)
    m.add_argument("--num-queries", type=int, default=1)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument(
        "--save-to-disk",
        type=str,
        default=None,
        help="Optional path to save DatasetDict to disk",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "passkey":
        generate_passkey_dataset(
            output_path=args.output,
            tokenizer_name=args.tokenizer,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            depth=args.depth,
            seed=args.seed,
        )

    elif args.task == "multi":
        with open(args.filler_file, "r") as f:
            filler_text = json.load(f)["text"]

        ds = generate_multi_passkey_dataset(
            output_path=args.output,
            tokenizer_name=args.tokenizer,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            filler_text=filler_text,
            num_keys=args.num_keys,
            num_queries=args.num_queries,
            seed=args.seed,
        )

        if args.save_to_disk:
            DatasetDict({"train": ds}).save_to_disk(args.save_to_disk)
            print(f"Dataset saved to {args.save_to_disk}")


if __name__ == "__main__":
    main()
