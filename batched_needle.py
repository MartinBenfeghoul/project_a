# Option B
# 1. For each sequence:
#   a. Generate KVs:
#   b. Train one MLP per head
#   c. Update KV cache
#   d. Compute the loss + perf on that batch
#   e. Accumulate the loss + perf
import json
import time

import torch
import torch.nn.functional as F
from torch import nn

from datasets import load_dataset, load_from_disk

from utils import (
    get_model_and_tokenizer,
    generate_kv_batched,
    avg_nll,
    clean,
)

# load all environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

class VectorizedIndependentHeadMLP(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_factor=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Total channels across all heads
        total_in = num_heads * head_dim
        total_hidden = num_heads * (head_dim * hidden_factor)
        
        self.net = torch.nn.Sequential(
            # First layer: Expand dim per head
            torch.nn.Conv1d(total_in, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            # Middle layers: Keep hidden dim per head
            torch.nn.Conv1d(total_hidden, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            torch.nn.Conv1d(total_hidden, total_hidden, kernel_size=1, groups=num_heads),
            torch.nn.GELU(),
            # Final layer: Project back to head_dim per head
            torch.nn.Conv1d(total_hidden, total_in, kernel_size=1, groups=num_heads),
        )

    def forward(self, x):
        # x input: [Batch, NumHeads, NumTokens, HeadDim]
        b, h, t, d = x.shape
        
        # 1. Reshape for Conv1d: [Batch, Channels (H*D), Length (T)]
        # We move 'd' next to 'h' and flatten them into the channel dim
        x_reshaped = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
        
        # 2. Process all heads in parallel
        out = self.net(x_reshaped)
        
        # 3. Reshape back to original: [Batch, NumHeads, NumTokens, HeadDim]
        return out.view(b, h, d, t).permute(0, 1, 3, 2)
    
def cosine_loss(v_hat, v_true):
    return 1 - F.cosine_similarity(v_hat, v_true, dim=-1).mean()

def save_results(file_name, avg_nll, avg_nll_modified_cache, new_avg_nll_change_perc, accuracy, threshold, num_epoch, mem_func, num_token, num_token_per_training, type_of_seq, num_seq, lr, loss_func, num_changed_kv, percentage_changed_kv, layer_decomposition=None):
    with open(file_name, 'a') as f:
        f.write(
            json.dumps(
            {
                "avg_nll": avg_nll,
                "avg_nll_change_thresh": avg_nll_modified_cache,
                "avg_nll_change_perc": new_avg_nll_change_perc,
                "avg_accuracy_modified_cache": accuracy,
                "threshold": threshold,
                "num_epoch": num_epoch,
                "mem_func": mem_func,
                "num_token": num_token,
                "num_token_per_training": num_token_per_training,
                "type_of_seq": type_of_seq,
                "num_seq": num_seq,
                "lr": lr,
                "loss_func": loss_func,
                "num_changed_kv": num_changed_kv,
                "percentage_changed_kv": percentage_changed_kv,
                "layer_decomposition": layer_decomposition,
            } 
        ) + "\n"
        )

if __name__ == "__main__":
    num_epochs = 20
    eval_batch_size = 32
    #loss_func = cosine_loss
    loss_func = F.mse_loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = "meta-llama/Llama-3.1-8B-Instruct"
    local_dir = "./llama_3.1_8b_instruct_local"
    file_name = "./results_needle_mse_long.jsonl"
    load_ds_from_disk = True
    lr = 1.e-3
    min_len_seq = 128
    max_len_seq = 1024

    #splits = [i for i in range(min_len_seq * 2, max_len_seq + 1, min_len_seq)]
    splits = [768, 1024, 2048, 3072, 4096, 5120]
    #splits = [6144, 7168, 8192, 9216, 10240]
    #target_percentages = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 5]
    target_percentages = [1, 5, 10, 25, 50, 75, 100]
    thresholds = [0.9999, 0.9995, 0.999, 0.99, 0.9, 0.85, 0.8]    
    #thresholds = [0.999999, 0.99995, 0.9999, 0.9995, 0.999, 0.995] 

    if load_ds_from_disk:
        print("Loading datasets from local")
        datasets = [load_from_disk(f"./datasets/passkey_seq_len_{split}") for split in splits]
    else:
        print("Loading datasets from HuggingFace")
        datasets = []
        for split in splits:
            ds = load_dataset("HHazard/passkey", split=split)
            ds.save_to_disk(f"./datasets/passkey_seq_len_{split}")
            datasets.append(ds)

    model, tokenizer = get_model_and_tokenizer(model, device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.eval()

    batch_size = 1

    for ds_idx, ds in enumerate(datasets):
        print(f"Currently running with sequence length {splits[ds_idx]}")
        #if ds_idx < 5:
        #    continue
        old_total_nlls = 0
        new_total_nlls = [0 for _ in thresholds]
        total_num_changed_kv = [0 for _ in thresholds]
        layer_decompositions = [{} for _ in thresholds]
        accuracies = [0 for _ in thresholds]
        total_nlls_perc_change = [0 for _ in target_percentages]
        new_thresholds = [0 for _ in target_percentages]

        start_time = time.time()

        for seq_idx, seq in enumerate(datasets[ds_idx]):
        
            if seq_idx > 5:
                break
  
            mlps = []
            displacement = []
            epoch_idx = 0

            avg_nll_, kv_cache = avg_nll([seq], model, eval_batch_size, tokenizer, device)
            #print(f'NLL for seq {seq_idx} is {avg_nll_}')
            _, num_head, num_token, head_dim = kv_cache[0][0].shape
            
            layer_mlps = [
                VectorizedIndependentHeadMLP(num_heads=num_head, head_dim=head_dim).to(device)
                for _ in range(len(kv_cache))
            ]

            for l, layer in enumerate(kv_cache.layers):
                print(l, layer.keys.dtype, layer.values.dtype)

            for l, mlp in enumerate(layer_mlps):
                dtypes = {p.dtype for p in mlp.parameters()}
                print(l, dtypes)

            num_layer = len(kv_cache)
            num_token_per_training = num_token
            
            # Verify NLL (optional check)
            # avg_nll_, _ = avg_nll_forced_batched_2([seq], model, eval_batch_size, tokenizer, device, num_head, kv_cache)
            old_total_nlls += avg_nll_

            all_params = [p for mlp in layer_mlps for p in mlp.parameters()]
            optimizer = torch.optim.Adam(all_params, lr=lr)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                total_loss = 0
                
                for layer_idx, (keys, values) in enumerate(kv_cache):
                    # keys/values shape: [1, num_head, num_token, head_dim]                    
                    v_hat = layer_mlps[layer_idx](keys[:, :, :10, :])
                    loss = loss_func(v_hat, values[:, :, :10, :])
                    loss.backward()
                    total_loss += loss.item()
                
                optimizer.step()
                print(f'Epoch {epoch} loss is {total_loss}')
            
            del kv_cache
            optimizer.zero_grad()
            del optimizer

            all_errors = []
            all_preds = []

            with torch.no_grad():
                past_key_values, original_input_id = generate_kv_batched([seq], model, eval_batch_size, tokenizer, device)

                for perc_idx, target_perc in enumerate(target_percentages):
                    input_id = original_input_id.clone()
                    for layer_idx, layer in enumerate(past_key_values.layers):
                        keys = layer.keys     # [1, num_head, num_token, head_dim]
                        values = layer.values # [1, num_head, num_token, head_dim]

                        v_approx = layer_mlps[layer_idx](keys)
                        
                        # shape: [1, num_head, num_token]
                        errors = F.mse_loss(v_approx, values, reduction='none').mean(dim=-1)
                        
                        all_errors.append(errors.flatten()) 
                        all_preds.append(v_approx) # Keep shape [1, h, t, d]

                    flat_errors = torch.cat(all_errors)
                    count = 0
                    k = int(len(flat_errors) * (target_perc / 100))
                    if k == 0: continue
                
                    threshold_val = torch.topk(flat_errors, k, largest=False).values[-1]

                    for layer_idx, layer in enumerate(past_key_values.layers):
                        
                        v_approx_layer = all_preds[layer_idx]
    
                        # layer.values is [1, H, T, D]
                        v_orig_3d = layer.values[0] 
                        v_approx_3d = v_approx_layer[0]

                        layer_errors = F.mse_loss(v_approx_3d, v_orig_3d, reduction='none').mean(dim=-1)
                        
                        mask = layer_errors <= threshold_val

                        v_orig_3d[mask] = v_approx_3d[mask]
                        
                        count += mask.sum().item()
                
                    """new_avg_nll_, _ = avg_nll_forced_batched_2([seq], model, eval_batch_size, tokenizer, device, num_head, past_key_values)
                    total_nlls_perc_change[perc_idx] += new_avg_nll_"""
                    total_num_changed_kv[perc_idx] += count
                    new_thresholds[perc_idx] += threshold_val

                    layer_decomposition = {}

                    #old_cache = tuple(
                    #    (k.clone(), v.clone())
                    #    for (k, v) in past_key_values
                    #)

                    #for layer_idx, (layer, updated_layer) in enumerate(zip(past_key_values, old_cache)):
                    #    k, v  = layer
                    #    updated_k, updated_v = updated_layer
                    #    dk = max_diff(k, updated_k)
                    #    dv = max_diff(v, updated_v)
                    #    layer_decompositions[perc_idx][layer_idx] = layer_decompositions[perc_idx].get(layer_idx, 0) + dv                
                        #print(
                        #    f"Layer {layer_idx:02d} | "
                        #    f"ΔK mean: {dk:.6e} | "
                        #    f"ΔV mean: {dv:.6e}"
                        #)
                    pred = ""
                    #print(f"Sequence is: {seq}")

                    prompt_len = past_key_values.get_seq_length()

                    #print("-------------------------")

                    for t in range(16):
                        with torch.no_grad():
                            out = model(
                                input_ids=input_id,
                                past_key_values=past_key_values,
                                use_cache=True
                            )
                        past_key_values = out.past_key_values
                        next_token_id = out.logits[:, -1].argmax(dim=-1)
                        new_token = tokenizer.decode(next_token_id)
                        pred += new_token
                        input_id = next_token_id.unsqueeze(0)

                    print("Generated:", pred)
                
                    accuracies[perc_idx] += 1 if seq["answer"] == clean(pred) else 0

                    past_key_values.crop(prompt_len)
                    
                    
            #del old_cache
            del past_key_values
            del mlps
            del all_params
            torch.cuda.empty_cache()
                
        num_samples = seq_idx

        end_time = time.time()

        print(f'Overall nll for sequence {splits[ds_idx]} is {old_total_nlls / num_samples:.4f}')
        print(f'It took {(end_time - start_time) / 60:.2f} minutes')
        for perc_idx, total_nll in enumerate(new_total_nlls):
            avg_nll_ = old_total_nlls / num_samples
            new_avg_nll_ = total_nll / num_samples
            new_avg_nll_change_perc = total_nlls_perc_change[perc_idx] / num_samples
            num_changed_kv = total_num_changed_kv[perc_idx] / num_samples
            accuracy = accuracies[perc_idx] / num_samples
            new_thresh = (new_thresholds[perc_idx] / num_samples).item()
            percentage_changed_kv = (total_num_changed_kv[perc_idx] * 100) / ((num_samples * num_token * num_layer * num_head)) 
            normalized_layers = {
                layer_num: total_dv / num_samples 
                for layer_num, total_dv in layer_decompositions[perc_idx].items()
            }
            
            print(f'Overall acc for modified cache with threshold {splits[ds_idx]} is {accuracy:.4f}')
            #print(f'Overall nll for modified cache for sequence {seq_idx} with threshold {thresholds[idx]} is {new_avg_nll_:.4f}')
            print(f'Overall nll for modified cache for sequence {splits[ds_idx]} with percentage {target_percentages[perc_idx]} is {new_avg_nll_change_perc:.4f} official {percentage_changed_kv}')
            #save_results(file_name, avg_nll_, new_avg_nll_, new_avg_nll_change_perc, accuracy, thresholds[perc_idx], num_epochs, "MLPs", num_token, num_token_per_training, "needle", num_samples, lr, "cosine_similarity", num_changed_kv, percentage_changed_kv, normalized_layers)
            #save_results(file_name, avg_nll_, new_avg_nll_, new_avg_nll_change_perc, accuracy, new_thresh, num_epochs, "MLPs", num_token, num_token_per_training, "needle", num_samples, lr, "cosine_similarity", num_changed_kv, percentage_changed_kv, normalized_layers)
