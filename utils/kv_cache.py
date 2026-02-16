import torch


def generate_outputs_single_pass(ds, model, tokenizer, device):
    texts = [ds[i]["prompt"] for i in range(len(ds))]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(
            **inputs,
            labels=inputs["input_ids"],
            use_cache=True,
        )
    return inputs, out


def generate_kv_batched(ds, model, batch_size, tokenizer, device):
    texts = [ds[i]["prompt"] for i in range(len(ds))]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    num_seq, seq_len = input_ids.shape

    for std in range(0, num_seq, batch_size):
        end = min(std + batch_size, num_seq)
        batch_ids = input_ids[std:end]

        with torch.no_grad():
            out = model(input_ids=batch_ids[:, :1], use_cache=True)
            cache = out.past_key_values

        for token_idx in range(1, seq_len):
            if token_idx < seq_len - 1:
                with torch.no_grad():
                    out = model(
                        input_ids=batch_ids[:, token_idx : token_idx + 1],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values

    return cache, batch_ids[:, token_idx : token_idx + 1]
