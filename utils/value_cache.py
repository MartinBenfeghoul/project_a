from .cache import SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Optimizer, Adam
from typing import Type
from torch.nn.functional import mse_loss
import torch
from .model.mlp import MLP

LOSS_FUNC = {
    'mse': mse_loss
}

OPTIMIZER = {
    'adam': Adam
}

class MLPValueLayer(SingleTensorDynamicLayer):
    def __init__(
        self,
        mlp_num_layers: int,
        mlp_hidden_factor: int,
        mlp_num_heads: int,
        lr: float,
        optimizer_cls: Type[Optimizer],
        loss_func,
    ):
        super().__init__()

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_factor = mlp_hidden_factor
        self.mlp_num_heads = mlp_num_heads

        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.loss_func = loss_func

        self.mlp = None
        self.optimizer = None

        self.indices = None
        self.compressed_values = None
        self.is_compressed = False
        
    def lazy_initialization(
            self,
            value_states: torch.Tensor,
            ) -> None:
        
        super().lazy_initialization(value_states)
        
        _, self.num_heads, _, self.head_dim = value_states.shape

        self.indices = torch.tensor([], dtype=torch.long, device=value_states.device)

        self.compressed_values = torch.tensor([], dtype=value_states.dtype, device=value_states.device)

        self.mlp = MLP(
            head_dim=self.head_dim, 
            num_layers=self.mlp_num_layers, 
            hidden_factor=self.mlp_hidden_factor, 
            num_heads=self.num_heads, 
            device=value_states.device
            )
        
    def eval_layer(self, keys) -> None:
        values = self.tensor # [1, num_head, num_token, head_dim]
        mlp = self.mlp
        v_approx = mlp(keys)                 
        # shape: [1, num_head, num_token]
        errors = self.loss_func(v_approx, values, reduction='none').mean(dim=-1)
        return errors, v_approx   

    def compress(self, threshold_val, keys) -> int:
        v_approx = self.mlp(keys)
        errors = self.loss_func(self.tensor, v_approx, reduction='none').mean(dim=-1)
        mask = errors > threshold_val
        self.indices = mask.nonzero(as_tuple=True)
        b, h, t = self.indices
        self.compressed_values = self.tensor[b, h, t]
        self.tensor = torch.tensor([], dtype=torch.float, device=self.tensor.device)
        self.is_compressed = True

    def temp_decompress(self, keys) -> torch.Tensor:
        values = self.mlp(keys)
        b, h, t = self.indices
        values[b, h, t] = self.compressed_values
        return values
    
    def decompress(self, keys) -> None:
        if self.is_compressed == False:
            return
        values = self.mlp(keys)
        b, h, t = self.indices
        values[b, h, t] = self.compressed_values
        self.tensor = values
        self.is_compressed = False
        self.compressed_values = torch.tensor([], dtype=torch.float, device=self.tensor.device)
        self.indices = torch.tensor([], dtype=torch.long, device=self.tensor.device)

    def update(self, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        
        keys = cache_kwargs["keys"]

        super().update(value_states, cache_kwargs)

        if self.is_compressed == False:
            return self.tensor

        else:
            decomp_values = self.temp_decompress(keys)

            decomp_values = torch.cat([decomp_values[..., :self.tensor.shape[-2], :], self.tensor], dim=-2)

            return decomp_values
            

    def crop(self, max_length: int) -> None:

        if self.is_compressed == True:
            self.decompress()
        
        super().crop(max_length)


class MLPValueCache(SingleTensorCache):
    def __init__(
        self,
        *args,
        num_layers_per_mlp: list[int],
        hidden_factors_per_mlp: list[int],
        num_heads_per_mlp: list[int],
        lr: float = 1e-3,
        device: str = "cuda",
        optimizer: str = "adam",
        loss_func: str = "mse",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert len(num_layers_per_mlp) == len(hidden_factors_per_mlp) == len(num_heads_per_mlp)

        self.num_layers_per_mlp = num_layers_per_mlp
        self.hidden_factors_per_mlp = hidden_factors_per_mlp
        self.num_heads_per_mlp = num_heads_per_mlp

        self.lr = lr
        self.device = device

        self.optimizer_cls = OPTIMIZER[optimizer]
        self.loss_func = LOSS_FUNC[loss_func]

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        return MLPValueLayer(
            mlp_num_layers=self.num_layers_per_mlp[layer_idx],
            mlp_hidden_factor=self.hidden_factors_per_mlp[layer_idx],
            mlp_num_heads=self.num_heads_per_mlp[layer_idx],
        )

    def update(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ):
        while len(self.layers) <= layer_idx:
            new_idx = len(self.layers)
            self.layers.append(self._build_layer(new_idx))

        values = self.layers[layer_idx].update(
            value_states=value_states,
            cache_kwargs=cache_kwargs,
        )

        return values

    def calc_compression_ratio(self):
        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            _, h, t, d = layer.keys.shape

            original = h * t * d

            num_params = layer.mlp.num_params
            num_stored = layer.indices[0].numel() if layer.is_compressed else 0

            compressed = (
                num_params
                + num_stored * d
                + num_stored * 3
            )

            original_total += original
            compressed_total += compressed

        return original_total / compressed_total

    def train(self, num_epochs: int = 5):
        all_params = [p for layer in self.layers for p in layer.mlp.parameters()]
        optimizer = self.optimizer_cls(all_params, lr=self.lr)
        for epoch in range(num_epochs):
                optimizer.zero_grad()
                total_loss = 0
                
                for layer in self.layers:
                    keys = layer.keys               # NEED TO PASS THE KEYS SOMEHOW
                    values = layer.tensor
                    # keys/values shape: [1, num_head, num_token, head_dim]                    
                    v_hat = layer.mlp(keys)
                    loss = self.loss_func(v_hat, values)
                    loss.backward()
                    total_loss += loss.item()
                
                optimizer.step()

                print(f'Epoch {epoch} loss is {total_loss}')

    def compress(self, thresh) -> None:
        # runs train and then evict
        self.train()
        for layer in self.layers:
            layer.compress(thresh)
    
    def decompress(self) -> None:  
        for layer in self.layers:
            layer.decompress()
    
    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, layer.mlp, getattr(layer, "_sliding_window_tensor", None)


VALUE_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "mlp": MLPValueCache,
}