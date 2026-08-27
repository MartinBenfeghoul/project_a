import argparse
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.attention_predictor import (
    AttentionPredictor,
    _attention_backend_specs,
    _get_attention_backend,
    attention_targets_to_distribution,
    max_pool_attention,
    topk_block_mask,
)
from utils import (
    parse_layers,
    prepare_run_directory,
    save_attention_predictor_checkpoint,
)
from utils import Dataset, collate, load_data


def sample_positions(
    seq_len: int,
    history_step: int,
    block_size: int,
    topk_blocks: int,
    samples_per_sequence: int,
    device: torch.device,
) -> torch.Tensor:
    min_pos = max(history_step, topk_blocks * block_size)
    if min_pos >= seq_len:
        raise ValueError(
            "seq_len must be larger than both history_step and "
            "topk_blocks * block_size."
        )
    valid = torch.arange(min_pos, seq_len, device=device)
    if valid.numel() <= samples_per_sequence:
        return valid
    perm = torch.randperm(valid.numel(), device=device)[:samples_per_sequence]
    return valid[perm].sort().values


def build_training_examples(
    attn: torch.Tensor,
    positions: torch.Tensor,
    history_step: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert one layer's full teacher attention into CNN examples.

    Args:
        attn: [batch, heads, query_len, key_len] attention probabilities.
        positions: Query positions to predict.

    Returns:
        histories: [batch * heads * positions, history_step, num_blocks]
        targets: [batch * heads * positions, num_blocks]
    """
    pooled = max_pool_attention(attn.float(), block_size)
    histories = []
    targets = []

    for pos in positions.tolist():
        histories.append(pooled[:, :, pos - history_step : pos, :])
        targets.append(pooled[:, :, pos, :])

    histories = torch.stack(histories, dim=2).flatten(0, 2)
    targets = torch.stack(targets, dim=2).flatten(0, 2)
    return histories, targets


def predictor_loss(  # TODO: the original paper uses MSE only
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk_blocks: int,
    bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_dist = attention_targets_to_distribution(targets)
    kl_loss = F.kl_div(
        F.log_softmax(logits.float(), dim=-1),
        target_dist,
        reduction="batchmean",
    )

    if bce_weight > 0:
        target_mask = topk_block_mask(target_dist, topk_blocks)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.float(), target_mask.float()
        )
    else:
        bce_loss = logits.new_tensor(0.0)

    loss = kl_loss + bce_weight * bce_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "bce_loss": float(bce_loss.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def recovery_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk_blocks: int,
) -> dict[str, float]:
    target_dist = attention_targets_to_distribution(targets)
    k = min(topk_blocks, logits.shape[-1])

    pred_idx = logits.topk(k, dim=-1).indices
    true_idx = target_dist.topk(k, dim=-1).indices

    pred_mask = torch.zeros_like(target_dist)
    pred_mask.scatter_(-1, pred_idx, 1.0)
    true_mask = torch.zeros_like(target_dist)
    true_mask.scatter_(-1, true_idx, 1.0)

    recovered_mass = (target_dist * pred_mask).sum(dim=-1).mean()
    recall_at_k = (pred_mask * true_mask).sum(dim=-1).div(k).mean()

    return {
        "recovered_mass": float(recovered_mass.detach().cpu()),
        "recall_at_k": float(recall_at_k.detach().cpu()),
    }


def train_layer_examples(
    predictor: AttentionPredictor,
    histories: torch.Tensor,
    targets: torch.Tensor,
    topk_blocks: int,
    bce_weight: float,
    loss_scale: float,
    microbatch_size: int,
) -> dict[str, float]:
    if histories.shape[0] != targets.shape[0]:
        raise ValueError(
            "histories and targets must have the same first dimension"
        )

    n = histories.shape[0]
    metrics_sum = {
        "loss": 0.0,
        "kl_loss": 0.0,
        "bce_loss": 0.0,
        "recovered_mass": 0.0,
        "recall_at_k": 0.0,
    }

    for start in range(0, n, microbatch_size):
        end = min(start + microbatch_size, n)
        weight = (end - start) / n
        logits = predictor(histories[start:end])
        loss, metrics = predictor_loss(
            logits=logits,
            targets=targets[start:end],
            topk_blocks=topk_blocks,
            bce_weight=bce_weight,
        )
        (loss * loss_scale * weight).backward()

        metrics.update(
            recovery_metrics(
                logits=logits.detach(),
                targets=targets[start:end],
                topk_blocks=topk_blocks,
            )
        )
        for key, value in metrics.items():
            metrics_sum[key] += value * weight

    return metrics_sum


def train(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, "bfloat16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False

    device = next(model.parameters()).device
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    prepare_run_directory(args, layers)

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
        batch_size=args.batch_size,
        collate_fn=collate,
    )

    predictor = AttentionPredictor().to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    running = {
        "loss": 0.0,
        "kl_loss": 0.0,
        "bce_loss": 0.0,
        "recovered_mass": 0.0,
        "recall_at_k": 0.0,
    }
    num_updates = 0

    train_state = {"positions": None, "batch_metrics": None}

    def capture_attention_hook(module, module_args, output):
        attn_weights = output[1]
        histories, targets = build_training_examples(
            attn=attn_weights,
            positions=train_state["positions"],
            history_step=args.history_step,
            block_size=args.block_size,
        )
        with torch.enable_grad():
            metrics = train_layer_examples(
                predictor=predictor,
                histories=histories,
                targets=targets,
                topk_blocks=args.topk_blocks,
                bce_weight=args.bce_weight,
                loss_scale=1.0 / len(layers),
                microbatch_size=args.predictor_microbatch_size,
            )
        for key, value in metrics.items():
            train_state["batch_metrics"][key] += value / len(layers)

    attention_specs = _attention_backend_specs()
    hook_handles = [
        module.register_forward_hook(capture_attention_hook)
        for module in model.modules()
        if _get_attention_backend(module, attention_specs) is not None
        and getattr(module, "layer_idx", None) in layers
    ]
    if not hook_handles:
        raise RuntimeError(
            "No attention modules matched --layers for capture hooks."
        )

    progress = tqdm(
        islice(dataloader, args.max_batches),
        total=args.max_batches,
        desc="training attention predictor",
    )

    for batch_idx, batch in enumerate(progress, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        train_state["positions"] = sample_positions(
            seq_len=input_ids.shape[-1],
            history_step=args.history_step,
            block_size=args.block_size,
            topk_blocks=args.topk_blocks,
            samples_per_sequence=args.samples_per_sequence,
            device=device,
        )
        train_state["batch_metrics"] = {key: 0.0 for key in running}

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        batch_metrics = train_state["batch_metrics"]

        torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), args.max_grad_norm
        )
        optimizer.step()
        num_updates += 1

        for key in running:
            running[key] += batch_metrics[key]

        if batch_idx % args.log_every == 0:
            averaged = {
                key: value / num_updates for key, value in running.items()
            }
            progress.set_postfix(
                loss=f"{averaged['loss']:.4f}",
                mass=f"{averaged['recovered_mass']:.4f}",
                recall=f"{averaged['recall_at_k']:.4f}",
            )

    for handle in hook_handles:
        handle.remove()

    save_attention_predictor_checkpoint(
        args,
        predictor,
        optimizer,
        layers,
        args.max_batches,
        running,
        num_updates,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Attention Predictor.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/attention_predictor",
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--samples_per_sequence", type=int, default=128)
    parser.add_argument("--predictor_microbatch_size", type=int, default=256)
    parser.add_argument("--history_step", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--topk_blocks", type=int, default=16)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma/range layer spec, for example '0,4,8' or '8-15'.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bce_weight", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
