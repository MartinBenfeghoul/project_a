import torch
from torch.utils.data import DataLoader

from utils import (
    PackedTokens,
    load_data,
    collate,
    get_model_and_tokenizer, 
    generate_outputs_single_pass,
)


def calculate_surprise(loss):
    """Calculate surprise from loss."""
    return torch.exp(loss)

def save_inputs_outputs(inputs, outputs, save_path):
    """Save model inputs and outputs to a file."""
    print(f"Saving inputs and outputs to {save_path}")
    torch.save({**inputs, **outputs}, save_path)

def main(
    model_name,
    dataset,
    save_path="model_outputs.pt",
    seq_len=1024,
    micro_bs=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    ds = load_data(dataset)
    eos_id = tokenizer.eos_token_id
    packed = PackedTokens(ds, tokenizer, seq_len=seq_len, eos_id=eos_id)
    dl = DataLoader(
        packed,
        batch_size=micro_bs,
        collate_fn=collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    inputs = next(iter(dl))
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    # save outputs to a file
    save_inputs_outputs(inputs, outputs, save_path)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset = "HuggingFaceFW/fineweb-edu"  # "example_dataset"
    save_path = "model_outputs.pt"

    main(
        model_name, dataset, 
        save_path=save_path
    )