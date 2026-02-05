from mlp import MLP
from transformers import DynamicCache, DynamicLayer, Cache
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
import torch.nn.functional as F


MEMORY_FUNCTIONS = {
    'mlps': MLP
}

class CompressibleLayer(DynamicLayer):
    def __init__(
              self,
              memory_func: str = 'mlps',
              mem_num_layers: int = 2,
              mem_hidden_factor: int = 2
              ) -> None:
        
        super().__init__()
        self.mem_num_layers, self.mem_hidden_factor = mem_num_layers, mem_hidden_factor
        self.memory_func = MEMORY_FUNCTIONS[memory_func]

        self.indices = None
        self.compressed_values = None
        self.memory = None
        self.is_compressed = False
         
    def lazy_initialization(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            ) -> None:
        
        super().lazy_initialization(key_states, value_states)
        
        _, self.num_heads, _, self.head_dim = key_states.shape

        self.indices = torch.tensor([], dtype=torch.long, device=key_states.device)
        self.compressed_values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        self.memory = self.memory_func(
            head_dim=self.head_dim, 
            num_layers=self.mem_num_layers, 
            hidden_factor=self.mem_hidden_factor, 
            num_heads=self.num_heads, 
            device=key_states.device)
        
    def eval_layer(
            self,
            loss_func = F.mse_loss
    ) -> None:
        keys = self.keys     # [1, num_head, num_token, head_dim]
        values = self.values # [1, num_head, num_token, head_dim]
        memory = self.memory
        v_approx = memory(keys)                 
        # shape: [1, num_head, num_token]
        errors = loss_func(v_approx, values, reduction='none').mean(dim=-1)
        return errors, v_approx   

    def compress(
            self,
            threshold_val,
            loss_func = F.mse_loss
        ) -> int:
        v_approx = self.memory(self.keys)
        errors = loss_func(self.values, v_approx, reduction='none').mean(dim=-1)
        mask = errors <= threshold_val
        self.indices = mask.nonzero(as_tuple=True)
        b, h, t = self.indices
        self.compressed_values = self.values[b, h, t]
        self.values = None
        self.is_compressed = True
        return mask.sum().item()
    
    def replace(
            self,
            threshold_val,
            loss_func = F.mse_loss
        ) -> int:
        v_approx = self.memory(self.keys)
        errors = loss_func(self.values, v_approx, reduction='none').mean(dim=-1)
        mask = errors <= threshold_val
        self.values[mask] = v_approx[mask]
        return mask.sum().item()

    def decompress(
            self,
    ) -> None:
        values = self.memory(self.keys)
        b, h, t = self.indices
        values[b, h, t] = self.compressed_values
        self.values = values
        self.is_compressed = False
        self.compressed_values = None
        self.indices = None

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.is_compressed == True:
            self.decompress()
        
        super().update(key_states, value_states, cache_kwargs)

    def crop(
            self, 
            max_length: int,
        ) -> None:

        if self.is_compressed == True:
            self.decompress()
        
        super().update(max_length)


class CompressibleCache(Cache):
    def __init__(self, 
                 num_layers: list[int],
                 hidden_factors: list[int],
                 device,
                 memory_func = None,
                 ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
                 config: PreTrainedConfig | None = None,
                 offloading: bool = False,
                 offload_only_non_sliding: bool = False,
                 ):
        
        self.num_layers = num_layers
        self.device = device

        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type, num_layer, hidden_factor in zip(layer_types, num_layers, hidden_factors):
                    layers.append(CompressibleLayer(memory_func=memory_func, mem_num_layers=num_layer, mem_hidden_factor=hidden_factor))

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            assert (len(ddp_cache_data)== len(num_layers) == len(hidden_factors))
            for layer_idx, (kv_and_optional_sliding, num_layer, hidden_factor) in enumerate(zip(ddp_cache_data, num_layers, hidden_factors)):
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                    # contain a third element, which is an optional sliding window tensor.
                    layers.append(CompressibleLayer(memory_func=memory_func, mem_num_layers=num_layer, mem_hidden_factor=hidden_factor))
                # Update the layer with the data
                _, _ = layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=CompressibleLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)


    def eval_memory(
            self,
            loss_func = F.mse_loss
    ) -> None:
        # Call to evict a percentage or any token under a threshold
        # This will probably be per model before moving to per layer?
        all_errors = []
        all_preds = []

        for layer in self.layers:
            errors, v_approx = layer.eval_layer(loss_func=loss_func)        
            all_errors.append(errors.flatten()) 
            all_preds.append(v_approx) # Keep shape [1, h, t, d]

        return all_errors, all_preds

    def train(
              self, 
              num_epochs, 
              optimizer=Adam, 
              loss_func=mse_loss, 
              lr=1.e-3,
            ):
        
        all_params = [p for layer in self.layers for p in layer.memory.parameters()]
        optimizer = optimizer(all_params, lr=lr)
        for epoch in range(num_epochs):
                optimizer.zero_grad()
                total_loss = 0
                
                for layer in self.layers:
                    keys = layer.keys
                    values = layer.values
                    # keys/values shape: [1, num_head, num_token, head_dim]                    
                    v_hat = layer.memory(keys)
                    loss = loss_func(v_hat, values)
                    loss.backward()
                    total_loss += loss.item()
                
                optimizer.step()

                print(f'Epoch {epoch} loss is {total_loss}')

    def compress(
            self,
            thresh,
    ) -> None:
        # runs train and then evict
        for layer in self.layers:
            layer.compress(thresh)

    def replace(
            self,
            threshold_value,
    ) -> None:
        count = 0
        for layer in self.layers:
            count += layer.replace(threshold_value)
        return count
    
    def decompress(
            self,
    ) -> None:  
        for layer in self.layers:
            layer.decompress()
        