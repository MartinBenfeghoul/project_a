import torch
import torch.nn as nn
import math

ACTIVATION_FN = {
    "none": nn.Identity(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "one_plus_elu": lambda x: 1 + torch.nn.functional.elu(x),
    "exponential": torch.exp,
    "softmax": nn.Softmax(dim=-1),
}


class MLP(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        num_layers: int = 4,
        hidden_factor: int = 2,
        num_heads: int = 8,
        device="cuda",
        deterministic_init: bool = True,
        intermediate_activation: str = "gelu",
    ):

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
            w = nn.Parameter(torch.empty(num_heads, curr_dim, out_dim)).to(
                self.device
            )
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
