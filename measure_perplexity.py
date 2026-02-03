import itertools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import (
    SVDCache,
    PackedTokens,
    load_data,
    collate,
    get_model_and_tokenizer,
)

def measure_perplexity(logits, target):
    nll = torch.nn.functional.cross_entropy(
        logits, target, reduction="mean"
    )
    ppl = torch.exp(nll)
    return nll, ppl

def main(
    model_name,
    dataset,
    seq_len=1024,
    micro_bs=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    if not os.path.exists('model_inputs.pt'):
        ds = load_data(dataset)
        packed = PackedTokens(
            ds, tokenizer, seq_len=seq_len, eos_id=tokenizer.eos_token_id,
            buffer_tokens=2 * micro_bs * seq_len,
        )
        dl = DataLoader(
            packed,
            batch_size=micro_bs,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        batches = list(itertools.islice(dl, micro_bs))
        inputs = batches[0]  # Take first batch only for saving
    else:
        inputs = torch.load('model_inputs.pt')


    inputs = {k: v.to(device) for k, v in inputs.items()}

    cache_position = torch.arange(
        inputs["input_ids"].shape[1], device=model.device
    )

    past_key_values = SVDCache(
        config=model.config, comp_ratio=2, niter=3
    )

    pos = 512
    chunk_input_ids = inputs["input_ids"][..., :pos]
    chunk_position = cache_position[:pos]
    out = model(
        chunk_input_ids, 
        labels=chunk_input_ids,
        past_key_values=past_key_values, 
        cache_position=chunk_position, 
        use_cache=True
    )
    nll = out.loss
    ppl= torch.exp(out.loss)
    print(f"Prefill over {pos} positions, nll={nll.item():.1f}, ppl={ppl.item():.1f}")
    past_key_values.update_events()
    ppls = []
    for _ in range(16):
        out = model(
            inputs["input_ids"][..., pos:pos+1], 
            past_key_values=past_key_values, 
            cache_position=cache_position[pos:pos+1], 
            use_cache=True
        )
        target = inputs["input_ids"][:, pos+1]  # shape (B,)
        logits = out.logits[:, -1, :] # shape (B, V)
        nll, ppl = measure_perplexity(logits, target)
        ppls.append(ppl.item())
        cr = past_key_values.compression_ratio
        print(f"pos={pos}, CR={cr:.1f}, nll={nll.item():.1f}, ppl={ppl.item():.1f}")
        
        pos += 1
    print(f"Average ppl={np.mean(ppls):.1f}+-{np.std(ppls):.1f}")

if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "/home/ma-user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    dataset = "HuggingFaceFW/fineweb-edu"  # "example_dataset"
    save_path = "model_outputs.pt"

    main(model_name, dataset)
