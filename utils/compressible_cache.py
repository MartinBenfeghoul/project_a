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



