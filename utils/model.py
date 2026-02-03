import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side='left'
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.pad_token = tokenizer.eos_token if pad_token is None else pad_token
    tokenizer.padding_side = pad_token_side
    model.eval()
    return model, tokenizer


class VectorizedIndependentHeadMLP(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_factor=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        total_in = num_heads * head_dim
        total_hidden = num_heads * (head_dim * hidden_factor)
        
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(total_in, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            torch.nn.Conv1d(total_hidden, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            torch.nn.Conv1d(total_hidden, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            torch.nn.Conv1d(total_hidden, total_in, kernel_size=1, groups=num_heads),
        )

    def forward(self, x):
        b, h, t, d = x.shape
        x_reshaped = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
        out = self.net(x_reshaped)
        return out.view(b, h, d, t).permute(0, 1, 3, 2)
    
def clone_mlp_params(layer_mlps):
    return [[p.clone() for p in mlp.parameters()] for mlp in layer_mlps]


def load_mlp_params(layer_mlps, params_list):
    for mlp, params in zip(layer_mlps, params_list):
        for p, saved_p in zip(mlp.parameters(), params):
            p.data.copy_(saved_p.data)


def load_pretrained_mlps(checkpoint_path, layer_mlps, device):
    """Load pre-trained MLP parameters from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for i, mlp in enumerate(layer_mlps):
        layer_key = f"layer_{i}"
        if layer_key in checkpoint:
            mlp.load_state_dict(checkpoint[layer_key])
    print(f"Loaded pre-trained MLP parameters from {checkpoint_path}")
    return layer_mlps

