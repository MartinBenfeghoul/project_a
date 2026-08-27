import argparse
import os
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.mlp import MLP
from utils import (
    Dataset,
    collate,
    extract_kv_linear_init,
    load_data,
    compute_rope_cos_sin,inverse_rope
)



def get_layer_kv(past_key_values, layer_idx):
    layer = past_key_values.layers[layer_idx]
    key = getattr(layer, "keys", getattr(layer, "key_states", None))
    value = getattr(layer, "values", getattr(layer, "value_states", None))
    return key, value


def unrope(keys, rope_theta):
    cos, sin = compute_rope_cos_sin(
        keys.shape[-2],
        keys.shape[-1],
        rope_theta,
        keys.device,
        keys.dtype,
    )
    return inverse_rope(keys, cos, sin)


def build_mlps(model, device, dtype):
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    w_linear = extract_kv_linear_init(model)
    mlps = []
    for layer_idx in range(cfg.num_hidden_layers):
        mlp = MLP(
            head_dim=head_dim,
            num_heads=cfg.num_key_value_heads,
            use_residual=True,
        ).to(device=device, dtype=dtype)
        if w_linear is not None:
            with torch.no_grad():
                mlp.W_linear.copy_(
                    w_linear[layer_idx].to(device=device, dtype=dtype)
                )
        mlps.append(mlp)
    return mlps


def optimizer_params(mlp, args):
    base_params = [param for param in mlp.parameters()]
    return [{"params": base_params, "lr": args.lr}]


def train_layer(mlp, optimizer, keys, values, args):
    loss_func = torch.nn.functional.mse_loss
    last_loss = None
    for _ in range(args.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func(mlp(keys), values)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def clone_state_dict(mlps, layers):
    return {
        layer_idx: {
            name: tensor.detach().cpu().clone()
            for name, tensor in mlps[layer_idx].state_dict().items()
        }
        for layer_idx in layers
    }


def load_state_dict(mlps, state_dicts, layers, device, dtype):
    for layer_idx in layers:
        state_dict = {
            name: tensor.to(device=device, dtype=dtype)
            for name, tensor in state_dicts[layer_idx].items()
        }
        mlps[layer_idx].load_state_dict(state_dict)


def save_checkpoint(path, mlps, args, layers, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        f"layer_{layer_idx}": mlps[layer_idx].state_dict()
        for layer_idx in layers
    }
    checkpoint["train_args"] = vars(args)
    checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def prepare_kv_batch(batch, model, layers, device, rope_theta, dtype):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    kv_by_layer = {}
    for layer_idx in layers:
        keys, values = get_layer_kv(outputs.past_key_values, layer_idx)
        keys = unrope(keys.detach(), rope_theta)
        kv_by_layer[layer_idx] = (
            keys.to(dtype=dtype),
            values.detach().to(dtype=dtype),
        )
    return kv_by_layer


def evaluate_mlps(mlps, val_batches, layers):
    loss_func = torch.nn.functional.mse_loss
    losses = {layer_idx: 0.0 for layer_idx in layers}
    with torch.no_grad():
        for kv_by_layer in val_batches:
            for layer_idx in layers:
                keys, values = kv_by_layer[layer_idx]
                loss = loss_func(mlps[layer_idx](keys), values)
                losses[layer_idx] += float(loss.detach().cpu())

    num_batches = max(len(val_batches), 1)
    losses = {
        layer_idx: loss / num_batches for layer_idx, loss in losses.items()
    }
    mean_loss = sum(losses.values()) / max(len(losses), 1)
    return mean_loss, losses


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    rope_theta = getattr(model.config, "rope_theta", 500000.0)
    layers = list(range(model.config.num_hidden_layers))
    mlps = build_mlps(model, device, dtype)
    optimizers = {
        layer_idx: Adam(optimizer_params(mlps[layer_idx], args), lr=args.lr)
        for layer_idx in layers
    }

    hf_dataset = load_data(
        dataset_path="HuggingFaceFW/fineweb-edu",
        subset_name="sample-100BT",
        shuffle_buffer_size=10000,
    )
    token_dataset = Dataset(
        hf_dataset,
        tokenizer,
        seq_len=args.seq_len,
        eos_id=tokenizer.eos_token_id,
    )
    dataloader = DataLoader(
        token_dataset,
        batch_size=1,
        collate_fn=collate,
    )
    dataloader_iter = iter(dataloader)

    val_batches = []
    if args.val_batches > 0:
        for _ in tqdm(
            range(args.val_batches), desc="preparing validation batches"
        ):
            val_batches.append(
                prepare_kv_batch(
                    next(dataloader_iter),
                    model,
                    layers,
                    device,
                    rope_theta,
                    dtype,
                )
            )

    running_loss = {layer_idx: 0.0 for layer_idx in layers}
    num_updates = {layer_idx: 0 for layer_idx in layers}
    best_val_loss = float("inf")
    best_state_dict = None
    best_metrics = None
    progress = tqdm(
        islice(dataloader_iter, args.max_batches),
        total=args.max_batches,
        desc="training value MLPs",
    )

    for batch_idx, batch in enumerate(progress, start=1):
        kv_by_layer = prepare_kv_batch(
            batch, model, layers, device, rope_theta, dtype
        )

        batch_loss = 0.0
        layer_losses = {}
        for layer_idx in layers:
            keys, values = kv_by_layer[layer_idx]
            loss = train_layer(
                mlps[layer_idx],
                optimizers[layer_idx],
                keys,
                values,
                args,
            )
            running_loss[layer_idx] += loss
            num_updates[layer_idx] += 1
            batch_loss += loss / len(layers)
            layer_losses[f"train/layer_{layer_idx}_loss"] = loss

        log_payload = {
            "train/step": batch_idx,
            "train/loss": batch_loss,
            "train/lr": args.lr,
            **layer_losses,
        }

        if val_batches:
            val_loss, val_layer_losses = evaluate_mlps(
                mlps, val_batches, layers
            )
            val_metrics = {
                f"val/layer_{idx}_loss": loss
                for idx, loss in val_layer_losses.items()
            }
            log_payload.update({"val/loss": val_loss, **val_metrics})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = clone_state_dict(mlps, layers)
                best_metrics = dict(log_payload)

        postfix = {"loss": f"{batch_loss:.4f}"}
        if val_batches:
            postfix["val"] = f"{best_val_loss:.4f}"
        progress.set_postfix(postfix)

    metrics = {
        f"layer_{idx}_loss": running_loss[idx] / max(num_updates[idx], 1)
        for idx in layers
    }
    if val_batches and best_state_dict is None:
        val_loss, val_layer_losses = evaluate_mlps(mlps, val_batches, layers)
        best_val_loss = val_loss
        best_state_dict = clone_state_dict(mlps, layers)
        best_metrics = {
            "val/loss": val_loss,
            **{
                f"val/layer_{idx}_loss": loss
                for idx, loss in val_layer_losses.items()
            },
        }
    if best_state_dict is not None:
        load_state_dict(mlps, best_state_dict, layers, device, dtype)
        metrics.update(
            {
                "best_val_loss": best_val_loss,
                **{
                    key.replace("/", "_"): value
                    for key, value in best_metrics.items()
                    if key.startswith("val/")
                },
            }
        )
    model_name = args.model_name.replace("/", "--")
    output_path = f"checkpoints/value_mlps/{model_name}/value_mlps_{args.max_batches}batches_{args.num_epochs}epochs_{args.seq_len}seqlen.pt"
    save_checkpoint(output_path, mlps, args, layers, metrics)
    print(f"Saved value MLP checkpoint to {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train value-cache MLPs on FineWeb."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_batches", type=int, default=32)
    parser.add_argument(
        "--val_batches",
        type=int,
        default=2,
        help="Number of batches for validation.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
    )
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
