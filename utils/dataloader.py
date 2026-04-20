import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset


def load_data(
    dataset_path: str = "HuggingFaceFW/fineweb-edu",
    subset_name: str = "sample-100BT",
    shuffle_buffer_size: int = 10_000,
):
    if dataset_path == "example_dataset":
        # Example dataset
        return [
            {"prompt": "Hello, how are you?"},
            {"prompt": "What is the capital of France?"},
        ]
    ds = load_dataset(
        dataset_path,
        subset_name,
        split="train",
        streaming=True,
    )
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    return ds


def collate(batch):
    # batch is list of dicts containing 1D tensors of same length
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    attn_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
    }


class PackedTokens(IterableDataset):
    """
    Streams text samples, tokenizes, concatenates, and yields fixed-length blocks.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int,
        eos_id: int,
        buffer_tokens: int = 1_000_000,
    ):
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


class Dataset(PackedTokens):

    def __iter__(self):
        for sample in super().__iter__():
            input_ids = sample["input_ids"]
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones_like(input_ids),
            }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


class PairedDataset(IterableDataset):
    """Wraps a Dataset and yields consecutive pairs as (support, query) tasks."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __iter__(self):
        it = iter(self.base)
        for support in it:
            query = next(it, None)
            if query is None:
                break
            yield {"support": support, "query": query}


def collate_pairs(batch):
    return {
        "support_input_ids": torch.stack(
            [b["support"]["input_ids"] for b in batch]
        ),
        "support_attention_mask": torch.stack(
            [b["support"]["attention_mask"] for b in batch]
        ),
        "query_input_ids": torch.stack(
            [b["query"]["input_ids"] for b in batch]
        ),
        "query_attention_mask": torch.stack(
            [b["query"]["attention_mask"] for b in batch]
        ),
    }
