import torch

from utils import get_model_and_tokenizer, generate_outputs_single_pass


from torch.nn import CrossEntropyLoss

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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    # Example dataset  # TODO: Replace with actual dataset loading
    ds = [{"prompt": "Hello, how are you?"}, {"prompt": "What is the capital of France?"}]

    inputs, outputs = generate_outputs_single_pass(
        ds, model, tokenizer, device
    )

    # save outputs to a file
    save_inputs_outputs(inputs, outputs, save_path)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset = "example_dataset"
    save_path = "model_outputs.pt"

    main(model_name, dataset, save_path=save_path)