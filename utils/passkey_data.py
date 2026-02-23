import random
import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer


def generate_passkey_sample(
    tokenizer,
    seq_len: int = 256,
    key_depth: float = 0.5,
    filler_sentences: List[str] = None,
) -> dict:
    """
    Generate a single passkey retrieval sample.

    Args:
        tokenizer: HuggingFace tokenizer or tokenizer name string.
        seq_len: Target sequence length in tokens.
        key_depth: Position of the key in the sequence as a fraction (0.0 = beginning, 1.0 = end).
        filler_sentences: Pool of filler sentences to sample from.

    Returns:
        dict with keys 'prompt' and 'answer'.
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if filler_sentences is None:
        filler_sentences = [
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

    key = "".join([str(random.randint(0, 9)) for _ in range(5)])

    intro = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorise it. I will quiz you about the important information there.\n"
    )
    key_sentence = f"The pass key is {key}. Remember it. {key} is the pass key.\n"

    body = []
    while True:
        body.append(random.choice(filler_sentences))
        tmp_prompt = intro + " ".join(body) + "\n"
        tokens = tokenizer(tmp_prompt + key_sentence, return_tensors="pt")["input_ids"].shape[-1]
        if tokens >= seq_len - 30:
            break

    insert_idx = int(len(body) * key_depth)
    insert_idx = min(insert_idx, len(body))
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
    """
    Stream a synthetic passkey retrieval dataset to JSONL.

    Args:
        output_path (str): Where to save the .jsonl file.
        tokenizer_name (str): HuggingFace tokenizer to control sequence length.
        num_samples (int): Number of examples to generate.
        seq_length (int): Target sequence length in tokens (prompt + key + filler + query).
        depth (float): Position of the key in the sequence as a fraction (0.0 = beginning,
                       0.5 = middle, 1.0 = end). If None, position is randomized.
        filler_sentences (List[str]): Pool of filler sentences to sample from.
        seed (int): Random seed.
    """

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if filler_sentences is None:
        filler_sentences = [
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            key = "".join([str(random.randint(0, 9)) for _ in range(5)])

            intro = (
                "There is an important info hidden inside a lot of irrelevant text. "
                "Find it and memorise it. I will quiz you about the important information there.\n"
            )
            key_sentence = f"The pass key is {key}. Remember it. {key} is the pass key.\n"

            # Build filler until near target token length (without placing key yet)
            body = []
            while True:
                body.append(random.choice(filler_sentences))
                tmp_prompt = intro + " ".join(body) + "\n"
                tokens = tokenizer(tmp_prompt + key_sentence, return_tensors="pt")["input_ids"].shape[-1]
                if tokens >= seq_length - 30:  # leave space for query
                    break

            # Insert key at specified depth or random position
            if depth is not None:
                insert_idx = int(len(body) * depth)
                insert_idx = min(insert_idx, len(body))  # clamp to valid range
            else:
                insert_idx = random.randint(0, len(body))
            body_with_key = body[:insert_idx] + [key_sentence] + body[insert_idx:]

            query = "What is the pass key? The pass key is"
            prompt = intro + " ".join(body_with_key) + "\n" + query

            ex = {"prompt": prompt, "answer": key}
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {num_samples} examples to {output_path}")


if __name__ == "__main__":
    sample = generate_passkey_sample("Qwen/Qwen2.5-0.5B-Instruct", seq_len=256)
    print(sample)

    lengths = {
        "128": 128,
        "256": 256,
        "4k": 4000,
        "8k": 8000,
        "16k": 16000,
        "32k": 32000,
        "64k": 64000,
        "128k": 128000,
        "256k": 256000,
    }
    depths = {
        "0": 0.0,
        "25": 0.25,
        "50": 0.5,
        "75": 0.75,
        "100": 1.0,
    }
    for length_name, length in lengths.items():
        for depth_name, depth in depths.items():
            print(f"Generating {length_name} dataset with depth {depth_name}%...")
            generate_passkey_dataset(
                output_path=f"data/passkey_retrieval/passkey_retrieval_{length_name}_depth{depth_name}.jsonl",
                tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
                num_samples=100,
                seq_length=length,
                depth=depth,
            )
