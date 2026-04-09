import torch
import torch.nn as nn
import math

ACTIVATION_FN = {
    "none": nn.Identity(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "one_plus_elu": lambda x: 1 + torch.nn.functional.elu(x),
}


class MLP(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        num_layers: int = 4,
        hidden_factor: int = 2,
        num_heads: int = 8,
        per_sequence: bool = False,
        batch_size: int | None = None,
        deterministic_init: bool = True,
        intermediate_activation: str = "gelu",
        use_residual: bool = False,
    ):
        super().__init__()

        self.intermediate_activation = ACTIVATION_FN[intermediate_activation]
        self.num_layers = num_layers
        self.per_sequence = per_sequence
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.use_residual = use_residual

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        curr_dim = head_dim
        hidden_dim = hidden_factor * head_dim

        if deterministic_init:
            torch.manual_seed(1)

        for i in range(num_layers):
            out_dim = head_dim if i == num_layers - 1 else hidden_dim

            w_shape = (batch_size if per_sequence else 1, num_heads, curr_dim, out_dim)
            b_shape = (batch_size if per_sequence else 1, num_heads, 1, out_dim)
            w = nn.Parameter(torch.zeros(*w_shape))
            b = nn.Parameter(torch.zeros(*b_shape))

            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            nn.init.zeros_(b)

            self.weights.append(w)
            self.biases.append(b)
            curr_dim = out_dim

        if use_residual:
            self.W_linear = nn.Parameter(torch.zeros(num_heads, head_dim, num_heads, head_dim))

    def forward(self, x):
        # x: [B, H, T, D]
        x_in = x
        for i in range(self.num_layers):
            w = self.weights[i]
            b = self.biases[i]

            x = torch.matmul(x, w) + b

            if i < self.num_layers - 1:
                x = self.intermediate_activation(x)

        if self.use_residual:
            x = x + torch.einsum('bhtd,hdqe->bqte', x_in, self.W_linear)

        return x