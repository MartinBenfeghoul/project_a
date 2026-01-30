import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset

def load_data(
        dataset_path: str = "HuggingFaceFW/fineweb-edu",
        subset_name: str = "sample-100BT",
    ):
    if dataset_path == 'example_dataset':
        # Example dataset
        return [{"prompt": "Hello, how are you?"}, {"prompt": "What is the capital of France?"}]
    ds = load_dataset(
        dataset_path,
        subset_name,
        split="train",
        streaming=True,
    )
    return ds

def collate(batch):
    # batch is list of dicts containing 1D tensors of same length
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}


class PackedTokens(IterableDataset):
    """
    Streams text samples, tokenizes, concatenates, and yields fixed-length blocks.
    """
    def __init__(self, hf_dataset, tokenizer, seq_len: int, eos_id: int, buffer_tokens: int = 1_000_000):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.buffer_tokens = buffer_tokens

    def __iter__(self):
        buf = []
        buf_len = 0

        for ex in self.ds:
            text = ex.get("text", None)
            if not text:
                continue

            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue

            # Optional: delimiter between docs
            ids.append(self.eos_id)

            buf.extend(ids)
            buf_len += len(ids)

            # Emit as many seq_len blocks as possible
            while buf_len >= self.seq_len:
                x = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                buf_len -= self.seq_len

                input_ids = torch.tensor(x, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),  # causal LM
                }

            # Keep buffer bounded (avoid pathological growth)
            if buf_len > self.buffer_tokens:
                # drop oldest tokens (rare in practice if seq_len draining happens)
                drop = buf_len - self.buffer_tokens
                buf = buf[drop:]
                buf_len = len(buf)


class MetaLearningDataset(PackedTokens):
    def __init__(self, hf_dataset, tokenizer, seq_len, eos_id, support_ratio=0.8, buffer_tokens=1_000_000):
        super().__init__(hf_dataset, tokenizer, seq_len, eos_id, buffer_tokens)
        self.support_ratio = support_ratio

    def __iter__(self):
        for sample in super().__iter__():
            input_ids = sample["input_ids"]
            seq_len = len(input_ids)
            split_idx = int(seq_len * self.support_ratio)

            yield {
                "support_input_ids": input_ids[:split_idx],
                "support_labels": input_ids[:split_idx].clone(),
                "query_input_ids": input_ids[split_idx:],
                "query_labels": input_ids[split_idx:].clone(),
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones_like(input_ids),
            }


def meta_collate(batch):
    return {
        "support_input_ids": torch.stack([b["support_input_ids"] for b in batch]),
        "support_labels": torch.stack([b["support_labels"] for b in batch]),
        "query_input_ids": torch.stack([b["query_input_ids"] for b in batch]),
        "query_labels": torch.stack([b["query_labels"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }
