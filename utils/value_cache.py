from .cache import SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Optimizer, Adam
from typing import Type
from torch.nn.functional import mse_loss
import torch
from model.mlp import MLP
from transformers.cache_utils import Any
import numpy as np

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
        optimizer_cls: str = "adam",
        num_epochs: int = 5,
        lr: float = 1.e-3,
        loss_func: str = "mse",
    ):
        super().__init__()

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_factor = mlp_hidden_factor
        self.mlp_num_heads = mlp_num_heads

        self.loss_func = LOSS_FUNC[loss_func]

        
        self.optimizer_cls = OPTIMIZER[optimizer_cls]
        self.num_epochs = num_epochs
        self.lr = lr

        self.mlp = None
        self.indices = None
        self.compressed_values = None
        self.is_compressed = False
        self.prefill = True
        self.compressed_len = 0
        
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
            ).to(device=value_states.device, dtype=value_states.dtype)
        
    def eval(self, keys) -> None:
        values = self.tensor # [1, num_head, num_token, head_dim]
        mlp = self.mlp
        v_approx = mlp(keys)                 
        # shape: [1, num_head, num_token]
        errors = self.loss_func(v_approx, values, reduction='none').mean(dim=-1)
        return errors, v_approx   
    
    def train_mlp(self, keys):
        with torch.enable_grad():
            values = self.tensor.detach()
            keys = keys.detach()
            all_params = [p for p in self.mlp.parameters()]
            optimizer = self.optimizer_cls(all_params, lr=self.lr)
            for _ in range(self.num_epochs):
                optimizer.zero_grad()
                # keys/values shape: [1, num_head, num_token, head_dim]                    
                v_hat = self.mlp(keys)
                loss = self.loss_func(v_hat, values)
                loss.backward()                
                optimizer.step()

    def compress(self, threshold_val, keys):
        v_approx = self.mlp(keys)
        errors = self.loss_func(self.tensor, v_approx, reduction='none').mean(dim=-1)
        mask = errors > threshold_val
        self.indices = mask.nonzero(as_tuple=True)
        b, h, t = self.indices
        num_kept = mask.sum().item()
        #print(f"Number of values kept (error > {threshold_val}): {num_kept}")
        self.compressed_values = self.tensor[b, h, t]
        #print(self.compressed_values.numel())
        self.compressed_len = self.tensor.shape[2]
        self.tensor = torch.tensor([], dtype=keys.dtype, device=self.tensor.device)
        self.is_compressed = True
        
    def temp_decompress(self, keys) -> torch.Tensor:
        values = self.mlp(keys[:, :, :self.compressed_len, :])
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
        self.compressed_values = torch.tensor([], dtype=keys.dtype, device=self.tensor.device)
        self.indices = torch.tensor([], dtype=torch.long, device=self.tensor.device)

    def update(self, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        
        if cache_kwargs is None or "keys" not in cache_kwargs:
            raise ValueError("MLPValueLayer requires keys in cache_kwargs")
        
        keys = cache_kwargs["keys"]

        values = super().update(value_states)

        #print(f'Keys: {keys.shape}')
        #print(f'Values: {values.shape}')
        
        if self.prefill:
            #print(self.compressed_values.numel())
            #print("We are in prefill")
            self.train_mlp(keys)
            self.compress(threshold_val=0.001, keys=keys)
            self.prefill = False
            #print(f'Indices {self.indices[0].shape}')
            #print(f'Compressed values {self.compressed_values.shape}')
            return values
        elif self.is_compressed:
            #print("We are generating")
            decomp_values = self.temp_decompress(keys)
            decomp_values = torch.cat([decomp_values, self.tensor], dim=-2)
            #print(f'Decompressed values {decomp_values.shape}')
            return decomp_values
        else:
            raise Exception(
                "Prefill is set to False but the values where not compressed."
            ) 

    def crop(self, max_length: int) -> None:
        if self.compressed_len < max_length: 
            # If the cache isn't compressed or max_length is larger just crop the suffix tensor
            new_max_length = max_length - self.compressed_len
            super().crop(new_max_length)
            return
        
        # Otherwise we generate a mask of the idx to keep and update the different tensors
        self.compressed_len = max_length
        self.tensor = torch.tensor([], dtype=self.tensor.dtype, device=self.tensor.device)
        b, h, t = self.indices

        if self.indices[0].numel() == 0:
            return
    
        keep_idx = t < max_length

        self.indices = (b[keep_idx], h[keep_idx], t[keep_idx])
        self.compressed_values = self.compressed_values[keep_idx]

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
        num_epochs: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert len(num_layers_per_mlp) == len(hidden_factors_per_mlp) == len(num_heads_per_mlp)

        self.num_layers_per_mlp = num_layers_per_mlp
        self.hidden_factors_per_mlp = hidden_factors_per_mlp
        self.num_heads_per_mlp = num_heads_per_mlp

        self.lr = lr
        self.device = device

        self.optimizer_cls = optimizer
        self.loss_func = loss_func
        self.num_epochs = num_epochs

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        return MLPValueLayer(
            mlp_num_layers=self.num_layers_per_mlp[layer_idx],
            mlp_hidden_factor=self.hidden_factors_per_mlp[layer_idx],
            mlp_num_heads=self.num_heads_per_mlp[layer_idx],
            loss_func=self.loss_func,
            num_epochs=self.num_epochs
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
        
        #print(f'This is layer {layer_idx}')

        values = self.layers[layer_idx].update(
            value_states=value_states,
            cache_kwargs=cache_kwargs,
        )

        #keys = cache_kwargs["keys"]

        #print(self.calc_compression_ratio(keys))

        return values

    def calc_compression_ratio(self, keys):

        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            _, h, t, d = keys.shape

            original = h * t * d

            if not layer.is_compressed:
                original_total += original
                compressed_total += original
                continue

            num_params = sum(p.numel() for p in layer.mlp.parameters())
            #print(num_params)
            num_stored = layer.indices[0].numel() if layer.is_compressed else 0

            compressed = (
                num_params
                + num_stored * d
                + num_stored * 3
            )

            original_total += original
            compressed_total += compressed

        return original_total / compressed_total

    def compress(self, thresh) -> None:
        # runs train and then evict
        self.train()
        for layer in self.layers:
            layer.compress(thresh)
    
    def decompress(self) -> None:  
        for layer in self.layers:
            layer.decompress()
    
    #def __iter__(self):
    #    for layer in self.layers:
    #        yield layer.keys, layer.values, layer.mlp, getattr(layer, "_sliding_window_tensor", None)


VALUE_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "mlp": MLPValueCache,
}