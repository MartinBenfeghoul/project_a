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
        num_layers: int = 2,
        hidden_factor: int = 1,
        num_heads: int = 8,
        per_sequence: bool = False,
        batch_size: int | None = None,
        deterministic_init: bool = True,
        intermediate_activation: str = "relu",
        use_residual: bool = False,
        per_head_residual: bool = False,
    ):
        super().__init__()

        self.intermediate_activation = ACTIVATION_FN[intermediate_activation]
        self.num_layers = num_layers
        self.residual_eq = None

        batch_dim = batch_size if per_sequence else 1

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        curr_dim = head_dim
        hidden_dim = hidden_factor * head_dim

        if deterministic_init:
            torch.manual_seed(1)

        for i in range(num_layers):
            out_dim = head_dim if i == num_layers - 1 else hidden_dim

            w = torch.empty(1, num_heads, curr_dim, out_dim)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.weights.append(
                nn.Parameter(w.repeat(batch_dim, 1, 1, 1))
            )
            self.biases.append(
                nn.Parameter(torch.zeros(batch_dim, num_heads, 1, out_dim))
            )
            curr_dim = out_dim

        if use_residual:
            batch = "b" if per_sequence else ""
            self.residual_eq = (
                f"bhtd,{batch}hde->bhte"
                if per_head_residual
                else f"bhtd,{batch}hdqe->bqte"
            )
            linear_shape = (
                (num_heads, head_dim, head_dim)
                if per_head_residual
                else (num_heads, head_dim, num_heads, head_dim)
            )
            if per_sequence:
                linear_shape = (batch_dim, *linear_shape)
            self.W_linear = nn.Parameter(torch.zeros(linear_shape))

    def forward(self, x):
        # x: [B, H, T, D]
        x_in = x
        for i in range(self.num_layers):
            w = self.weights[i]
            b = self.biases[i]

            x = torch.matmul(x, w) + b

            if i < self.num_layers - 1:
                x = self.intermediate_activation(x)

        if self.residual_eq is not None:
            x = x + torch.einsum(self.residual_eq, x_in, self.W_linear)

        return x