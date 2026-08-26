import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        per_sequence: bool = False,
        batch_size: int | None = None,
        deterministic_init: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()

        self.num_layers = 2
        self.residual_eq = None

        batch_dim = batch_size if per_sequence else 1

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        curr_dim = head_dim
        hidden_dim = head_dim

        if deterministic_init:
            torch.manual_seed(1)

        for i in range(self.num_layers):
            out_dim = head_dim if i == self.num_layers - 1 else hidden_dim

            w = torch.empty(1, num_heads, curr_dim, out_dim)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.weights.append(nn.Parameter(w.repeat(batch_dim, 1, 1, 1)))
            self.biases.append(
                nn.Parameter(torch.zeros(batch_dim, num_heads, 1, out_dim))
            )
            curr_dim = out_dim

        if use_residual:
            batch = "b" if per_sequence else ""
            self.residual_eq = f"bhtd,{batch}hde->bhte"
            linear_shape = (num_heads, head_dim, head_dim)
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
                x = torch.relu(x)

        if self.residual_eq is not None:
            x = x + torch.einsum(self.residual_eq, x_in, self.W_linear)

        return x
