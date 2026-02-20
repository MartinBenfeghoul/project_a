import math
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from dotenv import load_dotenv

load_dotenv()


def download_model_to_hub(model_name, **kwargs):
    snapshot_download(model_name, **kwargs)


ACTIVATION_FN = {
    'none': nn.Identity(),
    'gelu': nn.GELU(),
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'one_plus_elu': lambda x: 1 + torch.nn.functional.elu(x),
    'exponential': torch.exp,
    'softmax': nn.Softmax(dim=-1),
}

def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side='left', torch_dtype=None
):
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )


    tokenizer.pad_token = (
        tokenizer.eos_token if pad_token is None else pad_token
    )
    tokenizer.padding_side = pad_token_side
    model.eval()
    return model, tokenizer


class MLP(nn.Module):
    def __init__(self,
                 num_heads: int = 8, 
                 head_dim: int = 128, 
                 hidden_factor: int = 2, 
                 num_layers: int = 4, 
                 device = "cuda",
                 deterministic_init: bool = True,
                 intermediate_activation: str = 'gelu'):
        
        super().__init__()
        self.intermediate_activation = ACTIVATION_FN[intermediate_activation]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.num_layers = num_layers
        self.device = device

        curr_dim = head_dim
        hidden_dim = hidden_factor * head_dim

        if deterministic_init:
            torch.manual_seed(1)

        for i in range(num_layers):
            out_dim = head_dim if i == num_layers - 1 else hidden_dim
            w = nn.Parameter(torch.empty(num_heads, curr_dim, out_dim)).to(self.device)
            b = nn.Parameter(torch.empty(num_heads, 1, out_dim)).to(self.device)

            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            nn.init.zeros_(b)
            
            self.weights.append(w)
            self.biases.append(b)
            curr_dim = out_dim
    
    
    def forward(self, x):
        # x: [B, H, T, D]
        for i in range(self.num_layers):

            x = torch.matmul(x, self.weights[i]) + self.biases[i]
            
            if i < self.num_layers - 1:
                x = self.intermediate_activation(x)
        return x

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
    inner_lr_params = checkpoint.get("inner_lr_params", None)
    if inner_lr_params is not None:
        inner_lr_params = [lr.to(device) for lr in inner_lr_params]
        print(f"Loaded meta-learned inner LR params from {checkpoint_path}")
    print(f"Loaded pre-trained MLP parameters from {checkpoint_path}")
    return layer_mlps, inner_lr_params

