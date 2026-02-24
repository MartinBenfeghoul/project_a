from .cache import SingleTensorCache, SingleTensorDynamicLayer
from torch.optim import Adam
from torch.nn.functional import mse_loss
import torch
from model.mlp import MLP
from typing import Any

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
        per_sequence: bool = False,
        target_perc: int = None,
        threshold: int = None,
        optimizer_cls: str = "adam",
        num_epochs: int = 5,
        lr: float = 1.e-3,
        loss_func: str = "mse"
    ):
        super().__init__()

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_factor = mlp_hidden_factor
        self.mlp_num_heads = mlp_num_heads
        self.per_sequence = per_sequence
        self.target_perc = target_perc
        self.threshold = threshold

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
        
    def lazy_initialization(self, value_states: torch.Tensor) -> None:
        
        super().lazy_initialization(value_states)
        
        _, self.num_heads, _, self.head_dim = value_states.shape

        self.indices = torch.tensor([], dtype=torch.long, device=value_states.device)

        self.compressed_values = torch.tensor([], dtype=value_states.dtype, device=value_states.device)

        self.mlp = MLP(
            head_dim=self.head_dim, 
            num_layers=self.mlp_num_layers, 
            hidden_factor=self.mlp_hidden_factor, 
            num_heads=self.num_heads,
            per_sequence=self.per_sequence,
            max_batch_size=value_states.shape[0] if self.per_sequence else None,
            ).to(device=value_states.device, dtype=value_states.dtype)  
        
    def train_mlp(self, keys: torch.Tensor) -> None:
        with torch.enable_grad():
            values = self.tensor.detach()
            keys = keys.detach()
            all_params = [p for p in self.mlp.parameters()]
            optimizer = self.optimizer_cls(all_params, lr=self.lr)
            for _ in range(self.num_epochs):
                optimizer.zero_grad()
                # keys/values shape: [num_sequences, num_head, num_token, head_dim]                    
                v_hat = self.mlp(keys)
                loss = self.loss_func(v_hat, values)
                loss.backward()                
                optimizer.step()

    def compress(self, keys: torch.Tensor) -> None:
        v_approx = self.mlp(keys)
        errors = self.loss_func(self.tensor, v_approx, reduction='none').mean(dim=-1)
        if self.threshold == None and self.target_perc == None:
            raise ValueError("MLPValueLayer requires either a threshold or target_perc to compress values")
        
        if self.target_perc is not None:
            if self.per_sequence:
                B = errors.shape[0]
                errors_b = errors.view(B, -1)
                k = int(errors_b.shape[1] * (self.target_perc / 100))
                thresh = torch.topk(errors_b, k, largest=False).values[:, -1]
                mask = errors > thresh[:, None, None]
            else:
                flatten_errors = errors.view(-1)
                k = int(len(flatten_errors) * (self.target_perc / 100))
                self.threshold = torch.topk(flatten_errors, k, largest=False).values[-1]
                mask = errors > self.threshold
        elif self.threshold is not None:
            mask = errors > self.threshold

        self.indices = mask.nonzero(as_tuple=True)
        b, h, t = self.indices
        self.compressed_values = self.tensor[b, h, t]
        self.compressed_len = self.tensor.shape[2]
        B, H, _, D = self.tensor.shape
        self.tensor = self.tensor.new_empty((B, H, 0, D))
        self.seq_len = 0
        self.is_compressed = True
        
    def temp_decompress(self, keys: torch.Tensor) -> torch.Tensor:
        values = self.mlp(keys[:, :, :self.compressed_len, :])
        b, h, t = self.indices
        values[b, h, t] = self.compressed_values
        return values
    
    def decompress(self, keys: torch.Tensor) -> None:
        if self.is_compressed == False:
            return
        values = self.mlp(keys)
        b, h, t = self.indices
        values[b, h, t] = self.compressed_values
        self.tensor = values
        self.is_compressed = False
        self.compressed_values = self.compressed_values.new_empty(0)
        self.indices = (
            self.indices[0].new_empty(0),
            self.indices[1].new_empty(0),
            self.indices[2].new_empty(0),
        )

    def update(self, 
               value_states: torch.Tensor, 
               cache_kwargs: dict[str, Any] | None = None
               ) -> torch.Tensor:
        
        if cache_kwargs is None or "keys" not in cache_kwargs:
            raise ValueError("MLPValueLayer requires keys in cache_kwargs")
        
        keys = cache_kwargs["keys"]

        values = super().update(value_states)
        
        if self.prefill:
            self.train_mlp(keys)
            self.compress(keys)
            self.prefill = False
            return values
        elif self.is_compressed:
            decomp_values = self.temp_decompress(keys)
            decomp_values = torch.cat([decomp_values, self.tensor], dim=-2)
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
        B, H, _, D = self.tensor.shape
        self.tensor = self.tensor.new_empty((B, H, 0, D))
        self.seq_len = 0
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
        target_perc: list[int],
        target_model_num_heads: int = 8,
        per_sequence: bool = False,
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
        self.target_perc = target_perc
        self.per_sequence = per_sequence

        self.target_model_num_heads = target_model_num_heads

        self.lr = lr
        self.device = device

        self.optimizer_cls = optimizer
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        
        self.comp_ratio = 0

    def _build_layer(self, layer_idx: int) -> MLPValueLayer:
        return MLPValueLayer(
            mlp_num_layers=self.num_layers_per_mlp[layer_idx],
            mlp_hidden_factor=self.hidden_factors_per_mlp[layer_idx],
            mlp_num_heads=self.num_heads_per_mlp[layer_idx],
            target_perc=self.target_perc[layer_idx],
            per_sequence=self.per_sequence,
            loss_func=self.loss_func,
            num_epochs=self.num_epochs
        )

    def update(
        self,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        while len(self.layers) <= layer_idx:
            new_idx = len(self.layers)
            self.layers.append(self._build_layer(new_idx))  

        values = self.layers[layer_idx].update(
            value_states=value_states,
            cache_kwargs=cache_kwargs,
        )

        return values

    def calc_compression_ratio(self) -> float:

        original_total = 0
        compressed_total = 0

        for layer in self.layers:
            h, d = self.target_model_num_heads, layer.head_dim
            t = layer.compressed_len + layer.tensor.shape[2] if layer.tensor.numel() else layer.compressed_len

            original = h * t * d

            if not layer.is_compressed:
                original_total += original
                compressed_total += original
                continue

            num_params = sum(p.numel() for p in layer.mlp.parameters())
            num_stored = layer.indices[0].numel() if layer.is_compressed else 0

            compressed = (
                num_params
                + num_stored * d
                + num_stored * 3
            )

            original_total += original
            compressed_total += compressed

        return original_total / compressed_total
    

VALUE_CACHE_CLASSES = {
    "baseline": SingleTensorCache,
    "mlp": MLPValueCache,
}