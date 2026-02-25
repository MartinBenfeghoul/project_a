import random
import torch

from .metrics import get_loss_func


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlps(layer_mlps, kv_cache, config, inner_lr_params=None):
    """
    Train MLPs to predict values from keys in the KV cache.

    Args:
        layer_mlps: List of MLP modules, one per layer
        kv_cache: KV cache containing keys and values
        config: Training config with num_epochs, lr, loss_func, optimizer
        inner_lr_params: Optional list of per-parameter LR tensors from meta-learning

    Returns:
        Trained layer_mlps
    """
    num_epochs = config.num_epochs
    lr = config.lr
    loss_func = get_loss_func(config.loss_func)
    optimizer_name = config.get("optimizer", "adam")

    all_params = [p for mlp in layer_mlps for p in mlp.parameters()]
    if inner_lr_params is not None:
        param_groups = [
            {"params": [p], "lr": float(lr_val)}
            for p, lr_val in zip(all_params, inner_lr_params)
        ]
        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(param_groups)
        else:
            optimizer = torch.optim.Adam(param_groups)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(all_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(all_params, lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0

        for layer_idx, layer in enumerate(kv_cache.layers):
            v_hat = layer_mlps[layer_idx](layer.keys.float())
            loss = loss_func(v_hat, layer.values.float())
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        print(f"Epoch {epoch} loss is {total_loss}")

    optimizer.zero_grad()
    del optimizer

    return layer_mlps
