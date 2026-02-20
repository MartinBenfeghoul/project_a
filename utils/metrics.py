import torch
import re
import torch.nn.functional as F


def cosine_loss(v_hat, v_true):
    """Cosine similarity loss between predicted and true values."""
    return 1 - F.cosine_similarity(v_hat, v_true, dim=-1).mean()


def get_loss_func(loss_func_name):
    """Get loss function by name."""
    if loss_func_name == "cosine":
        return cosine_loss
    return F.mse_loss


def clean(text):
    pre_dot = text.split(".")[0]
    digits = "".join(re.findall(r"\d", pre_dot))
    return digits


def accuracy(label, pred_label):
    return 1 if label == clean(pred_label) else 0


def make_predictions(
    batch_size, ds, model, tokenizer, device, past_key_values=None
):
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    predictions = []
    for std in range(0, len(prompts), batch_size):
        end = min(len(prompts), std + batch_size)
        batch_prompts = prompts[std:end]
        if len(batch_prompts) == 0:
            continue
        inputs = tokenizer(
            prompts[std:end],
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            output = (
                model(
                    input_ids=inputs["input_ids"],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if past_key_values
                else model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            )
        generated_tokens = output[:, inputs["input_ids"].shape[1] :]
        predictions.extend(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )
    return predictions


def eval_model(ds, model, batch_size, tokenizer, device, past_key_values=None):
    preds = make_predictions(
        batch_size, ds, model, tokenizer, device, past_key_values
    )
    acc = 0
    for pred, sample in zip(preds, ds):
        acc += accuracy(sample["answer"], pred)
    return (acc / len(ds)) * 100


def max_diff(a, b):
    return (a - b).abs().mean().item()


def update_cache(cache, updated_cache, num_head, token_idx, std=0, end=1):
    for layer, updated_layer in zip(cache.layers, updated_cache.layers):
        layer.keys[std:end, :, token_idx, :] = updated_layer.keys[
            std:end, :, token_idx, :
        ]
        layer.values[std:end, :, token_idx, :] = updated_layer.values[
            std:end, :, token_idx, :
        ]
    return cache

def avg_nll(ds, model, batch_size, tokenizer, device, num_head=0, updated_cache=None, already_tokenized=False):
    #texts = [ds["prompt"][i] + ds["answer"][i] for i in range(len(ds["prompt"]))]
    if not already_tokenized:
        texts = [ds[i]["prompt"] for i in range(len(ds))]
        ds = tokenizer(texts, return_tensors='pt', padding=True).to(device)

    input_ids = ds["input_ids"].to(device)
    attention_mask = ds["attention_mask"].to(device)
    num_seq, seq_len = input_ids.shape

    all_seq_means = []

    for std in range(0, num_seq, batch_size):
        end = min(std + batch_size, num_seq)
        batch_ids = input_ids[std:end]
        batch_mask = attention_mask[std:end]

        batch_loss = torch.zeros(batch_ids.size(0), device=device)
        valid_token_counts = batch_mask[:, 1:].sum(dim=1)

        with torch.no_grad():
            out = model(input_ids=batch_ids[:, :1], use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]

        if updated_cache:
            cache = update_cache(
                cache, updated_cache, num_head, token_idx=0, std=std, end=end
            )

        for token_idx in range(1, seq_len - 1):
            target_tokens = batch_ids[:, token_idx]

            step_loss = F.cross_entropy(logits, target_tokens, reduction="none")

            step_loss = step_loss * batch_mask[:, token_idx]
            batch_loss += step_loss

            if token_idx < seq_len:
                with torch.no_grad():
                    out = model(
                        input_ids=batch_ids[:, token_idx : token_idx + 1],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    logits = out.logits[:, -1, :]

            if updated_cache:
                cache = update_cache(
                    cache, updated_cache, num_head, token_idx, std=std, end=end
                )

        all_seq_means.append(batch_loss / valid_token_counts)

    final_mean = torch.cat(all_seq_means).mean().item()
    return final_mean, cache


def compute_kv_loss(layer_mlps, kv_cache, token_slice=None, loss_fn=F.mse_loss):
    total_loss = 0
    for layer_idx, (keys, values) in enumerate(kv_cache):
        if token_slice is not None:
            keys = keys[:, :, token_slice, :]
            values = values[:, :, token_slice, :]
        v_hat = layer_mlps[layer_idx](keys)
        total_loss += loss_fn(v_hat, values)
    return total_loss

