import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader


from utils import (
    get_model_and_tokenizer,
    avg_nll,
    VectorizedIndependentHeadMLP,
    MetaLearningDataset,
    load_data,
    meta_collate,
    )

from dotenv import load_dotenv

load_dotenv()

def inner_loop_functional(layer_mlps, kv_cache, support_slice, inner_lr, inner_steps, loss_fn=F.mse_loss):
    params = [p for mlp in layer_mlps for p in mlp.parameters()]
    adapted_params = [p.clone() for p in params] # to return adapted params as new tensor (not in-place)

    for _ in range(inner_steps):
        total_loss = 0
        param_idx = 0

        for layer_idx,layer in enumerate(kv_cache.layers):
            k = layer.keys[:, :, support_slice, :].float()
            v = layer.values[:, :, support_slice, :].float()

            mlp = layer_mlps[layer_idx]
            v_hat = functional_mlp_forward(mlp, k.float(), adapted_params, param_idx)
            param_idx += sum(1 for _ in mlp.parameters())

            total_loss += loss_fn(v_hat, v)

        grads = torch.autograd.grad(total_loss, adapted_params, create_graph=True)

        # functional, keeps graph
        adapted_params = [p - inner_lr * g for p, g in zip(adapted_params, grads)]

    return adapted_params


def functional_mlp_forward(mlp, x, all_params, start_idx):
    b, h, t, d = x.shape
    x_reshaped = x.permute(0, 1, 3, 2).reshape(b, h * d, t)

    param_idx = start_idx
    out = x_reshaped

    for layer in mlp.net:
        if isinstance(layer, nn.Conv1d):
            weight = all_params[param_idx]
            bias = all_params[param_idx + 1]
            out = F.conv1d(out, weight, bias, groups=layer.groups)
            param_idx += 2
        elif isinstance(layer, nn.GELU):
            out = F.gelu(out)

    return out.view(b, h, d, t).permute(0, 1, 3, 2)


def compute_query_loss_functional(layer_mlps, kv_cache, query_slice, adapted_params, loss_fn=F.mse_loss):
    total_loss = 0
    param_idx = 0

    for layer_idx, layer in enumerate(kv_cache.layers):
        k = layer.keys[:, :, query_slice, :].float()
        v = layer.values[:, :, query_slice, :].float()

        mlp = layer_mlps[layer_idx]
        v_hat = functional_mlp_forward(mlp, k.float(), adapted_params, param_idx)
        param_idx += sum(1 for _ in mlp.parameters())

        total_loss += loss_fn(v_hat, v)

    return total_loss


def meta_train(
    model,
    tokenizer,
    layer_mlps,
    dataloader,
    device,
    config,
    loss_fn=F.mse_loss,
):
    meta_lr = config.meta_lr
    inner_lr = config.inner_lr
    inner_steps = config.inner_steps
    support_ratio = config.support_ratio
    num_meta_epochs = config.num_meta_epochs
    eval_batch_size = config.eval_batch_size

    all_params = [p for mlp in layer_mlps for p in mlp.parameters()]
    meta_optimizer = torch.optim.Adam(all_params, lr=meta_lr)

    for epoch in range(num_meta_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            meta_optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            _, seq_len = input_ids.shape

            split_idx = int(seq_len * support_ratio)
            support_slice = slice(0, split_idx)
            query_slice = slice(split_idx, seq_len)

            _, kv_cache = avg_nll(batch, model, eval_batch_size, tokenizer, device, already_tokenized=True)

            # functional to track gradients
            adapted_params = inner_loop_functional(
                layer_mlps, kv_cache, support_slice, inner_lr, inner_steps, loss_fn
            )

            query_loss = compute_query_loss_functional(
                layer_mlps, kv_cache, query_slice, adapted_params, loss_fn
            )

            # Backpropagate through both loops to get meta-gradient
            query_loss.backward()

            meta_optimizer.step()

            epoch_loss += query_loss.item()
            num_batches += 1

            del kv_cache
            torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Query Loss: {query_loss.item():.6f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch} complete. Average Query Loss: {avg_loss:.6f}")

    return layer_mlps



def load_config():
    """
    CLI args use dotlist notation, e.g.:
        python meta_learning.py training.batch_size=4 training.inner_lr=0.001
    """
    base_config = OmegaConf.load("config/meta_learning.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    return config

def main():
    config = load_config()
    training_config = config.training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model, tokenizer = get_model_and_tokenizer(config.model.name, device)

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    print(f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    layer_mlps = [
        VectorizedIndependentHeadMLP(num_heads=num_kv_heads, head_dim=head_dim).to(device)
        for _ in range(num_layers)
    ]

    print("Loading dataset...")
    hf_dataset = load_data()

    meta_dataset = MetaLearningDataset(
        hf_dataset,
        tokenizer,
        seq_len=training_config.seq_len,
        eos_id=tokenizer.eos_token_id,
        support_ratio=training_config.support_ratio,
    )

    dataloader = DataLoader(
        meta_dataset,
        batch_size=training_config.batch_size,
        collate_fn=meta_collate,
    )

    print("Starting meta-training...")
    layer_mlps = meta_train(
        model=model,
        tokenizer=tokenizer,
        layer_mlps=layer_mlps,
        dataloader=dataloader,
        device=device,
        config=training_config,
    )
    
    trained_params = {f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)}
    torch.save(trained_params, config.checkpoint_path)
    print(f"Meta-learned parameters saved to {training_config.checkpoint_path}")


if __name__ == "__main__":
    trained_paramss = main()