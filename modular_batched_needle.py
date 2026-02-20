import os
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.mlp import MLP
from utils.passkey_data import generate_passkey_sample
from utils import (
    get_model_and_tokenizer,
    generate_kv_batched,
    avg_nll,
    clean,
    load_pretrained_mlps,
    train_mlps,
    set_seed,
    save_results,
)

from dotenv import load_dotenv

load_dotenv()


def evaluate_sequence(
    seq,
    layer_mlps,
    model,
    tokenizer,
    device,
    config,
):
    target_percentages = config.evaluation.target_percentages
    num_generate_tokens = config.evaluation.num_generate_tokens

    results = {
        "accuracies": [],
        "num_changed_kv": [],
        "thresholds": [],
    }

    with torch.no_grad():
        past_key_values, original_input_id = generate_kv_batched(
            [seq], model, 1, tokenizer, device  # batch_size=1 since processing one seq at a time
        )

        for target_perc in target_percentages:
            all_errors = []
            all_preds = []
            input_id = original_input_id.clone()

            for layer_idx, layer in enumerate(past_key_values.layers):
                keys = layer.keys.float()
                values = layer.values.float()

                v_approx = layer_mlps[layer_idx](keys)
                errors = F.mse_loss(v_approx, values, reduction="none").mean(dim=-1)

                all_errors.append(errors.flatten())
                all_preds.append(v_approx)

            flat_errors = torch.cat(all_errors)
            k = int(len(flat_errors) * (target_perc / 100))
            if k == 0:
                results["accuracies"].append(0)
                results["num_changed_kv"].append(0)
                results["thresholds"].append(0)
                continue

            threshold_val = torch.topk(flat_errors, k, largest=False).values[-1]
            count = 0

            for layer_idx, layer in enumerate(past_key_values.layers):
                v_approx_layer = all_preds[layer_idx]
                v_orig = layer.values  # [1, num_head, num_token, head_dim]
                v_approx_3d = v_approx_layer[0].float()

                layer_errors = F.mse_loss(
                    v_approx_3d, v_orig[0].float(), reduction="none"
                ).mean(dim=-1)
                mask = layer_errors <= threshold_val

                # Modify the actual cache in-place
                v_orig[0, mask] = v_approx_3d[mask].to(v_orig.dtype)
                count += mask.sum().item()

            results["num_changed_kv"].append(count)
            results["thresholds"].append(threshold_val.item())

            prompt_len = past_key_values.get_seq_length()
            pred = ""

            for _ in range(num_generate_tokens):
                with torch.no_grad():
                    out = model(
                        input_ids=input_id,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = out.past_key_values
                next_token_id = out.logits[:, -1].argmax(dim=-1)
                new_token = tokenizer.decode(next_token_id)
                pred += new_token
                input_id = next_token_id.unsqueeze(0)

            print("Generated:", pred)

            accuracy = 1 if seq["answer"] == clean(pred) else 0
            results["accuracies"].append(accuracy)

            past_key_values.crop(prompt_len)

    del past_key_values
    torch.cuda.empty_cache()

    return results


def run_experiment(model, tokenizer, splits, config, device):
    target_percentages = config.evaluation.target_percentages
    num_samples_per_split = config.data.num_samples_per_split
    pretrained_mlps_path = config.model.get("pretrained_mlps", None)
    num_epochs = config.training.num_epochs
    model_name = config.model.name.split("/")[-1]
    output_file = f"results/{model_name}/{num_samples_per_split}_samples_{num_epochs}epoch.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lr = config.training.lr
    loss_func = config.training.get("loss_func", "mse")

    for split in splits:
        print(f"Currently running with sequence length {split}")

        old_total_nlls = 0
        total_num_changed_kv = [0 for _ in target_percentages]
        accuracies = [0 for _ in target_percentages]
        new_thresholds = [0 for _ in target_percentages]

        start_time = time.time()
        num_samples = 0

        for _ in range(num_samples_per_split):
            seq = generate_passkey_sample(tokenizer, seq_len=split)

            num_samples += 1

            avg_nll_, kv_cache = avg_nll(
                [seq], model, 1, tokenizer, device
            )
            _, num_head, num_token, head_dim = kv_cache.layers[0].keys.shape

            layer_mlps = [
                MLP(num_heads=num_head, head_dim=head_dim).to(
                    device
                )
                for _ in range(len(kv_cache))
            ]

            num_layer = len(kv_cache)
            old_total_nlls += avg_nll_

            inner_lr_params = None
            if pretrained_mlps_path:
                layer_mlps, inner_lr_params = load_pretrained_mlps(pretrained_mlps_path, layer_mlps, device)

            layer_mlps = train_mlps(layer_mlps, kv_cache, config.training, inner_lr_params=inner_lr_params)

            del kv_cache
            torch.cuda.empty_cache()

            results = evaluate_sequence(
                seq, layer_mlps, model, tokenizer, device, config
            )

            for perc_idx in range(len(target_percentages)):
                accuracies[perc_idx] += results["accuracies"][perc_idx]
                total_num_changed_kv[perc_idx] += results["num_changed_kv"][perc_idx]
                new_thresholds[perc_idx] += results["thresholds"][perc_idx]

            del layer_mlps
            torch.cuda.empty_cache()

        end_time = time.time()

        print(
            f"Overall nll for sequence {split} is {old_total_nlls / num_samples:.4f}"
        )
        print(f"It took {(end_time - start_time) / 60:.2f} minutes")

        for perc_idx, target_perc in enumerate(target_percentages):
            accuracy = accuracies[perc_idx] / num_samples
            avg_changed_kv = total_num_changed_kv[perc_idx] / num_samples
            percentage_changed_kv = (total_num_changed_kv[perc_idx] * 100) / (
                num_samples * num_token * num_layer * num_head
            )
            avg_threshold = new_thresholds[perc_idx] / num_samples

            print(
                f"Overall acc for modified cache with seq_len {split} "
                f"and percentage {target_perc} is {accuracy:.4f}"
            )
            print(f"Avg KV changed: {avg_changed_kv:.2f}, Percentage: {percentage_changed_kv:.2f}%")

            save_results(
                file_name=output_file,
                avg_nll=old_total_nlls / num_samples,
                avg_nll_modified_cache=0,  # Not tracked in this version
                new_avg_nll_change_perc=target_perc,
                accuracy=accuracy,
                threshold=avg_threshold,
                num_epoch=num_epochs,
                mem_func="MLPs",
                num_token=num_token,
                num_token_per_training=num_token,
                type_of_seq="needle",
                num_seq=num_samples,
                lr=lr,
                loss_func=loss_func,
                num_changed_kv=avg_changed_kv,
                percentage_changed_kv=percentage_changed_kv,
                seq_len=split,
            )
            print(f"Saved results for split {split}, compression {target_perc}%")


def load_config():
    base_config = OmegaConf.load("config/batched_needle.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    return config


def main():
    config = load_config()

    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Using seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model, tokenizer = get_model_and_tokenizer(config.model.name, device)

    splits = config.data.splits

    print("Starting experiment...")
    run_experiment(model, tokenizer, splits, config, device)


if __name__ == "__main__":
    main()
