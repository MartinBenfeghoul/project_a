import random
import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

def generate_multi_passkey_dataset(
    output_path: str,
    tokenizer_name: str = "gpt2",
    num_samples: int = 100,
    seq_length: int = 1024,
    filler_text: str = None,
    num_keys: int = 3,       # Number of keys to hide
    num_queries: int = 2,    # Number of keys to ask for
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
        raise ValueError(f"Sequence length {seq_length} is too small for {num_keys} keys.")

    new_ds = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            # 1. Generate unique keys and values
            keys_data = []
            for i in range(num_keys):
                label = f"ALPHA_{i}" # You can randomize these labels too
                val = "".join(str(random.randint(0, 9)) for _ in range(5))
                keys_data.append({"label": label, "value": val})

            # 2. Prepare filler slice
            start = random.randint(0, len(filler_ids) - max_filler_len)
            filler_slice_text = tokenizer.decode(filler_ids[start : start + max_filler_len])

            # 3. Insert keys at random positions
            # We split the filler into chunks to ensure keys aren't all in one spot
            text_segments = []
            last_idx = 0
            # Get random insertion points in sorted order
            insert_points = sorted([random.randint(0, len(filler_slice_text)) for _ in range(num_keys)])
            
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
                #answers[q['label']] = q['value']
                answers += q['value'] + ' '
            
            #query_text += "\nAnswer in JSON format."

            prompt = intro + current_text + query_text

            entry = {
                "prompt": prompt,
                #"answer": json.dumps(answers)
                "answer": answers[:-1]
            }
            f.write(json.dumps(entry) + "\n")
            new_ds.append(entry)

    return Dataset.from_list(new_ds)

# --- Execution Logic ---
if __name__ == "__main__":
    lengths = {
        #"128": 128,
        #"256": 256,
        "512": 512,
        "1k": 1000,
        "2k": 2000,
        "4k": 4000,
        "6k": 6000,
        "8k": 8000,
        "10k": 10000,
        "12k": 12000,
        "15k": 15000,
        "20k": 20000,
        }
    
    filler_sentences = []

    #ds_name = "HHazard/multi-keys"
    ds_name = ""

    with open("data/PaulGrahamEssays.json", "r") as f:
        data = json.load(f)
        filler_sentences.append(data['text'])

    print(len(filler_sentences[0]))

    filler_sentences = filler_sentences[0][:100000]

    print(filler_sentences)

    new_dics = {}

    for name, length in lengths.items():
        print(f"Generating {name}...")
        new_dics[name] = generate_multi_passkey_dataset(
            output_path=f"data/multi_passkey_{name}.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            num_samples=100,
            seq_length=length,
            filler_text=filler_sentences,
            num_keys=3,    
            num_queries=1  
        )

    ds = DatasetDict(new_dics)
    ds.push_to_hub(ds_name)