import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv

load_dotenv()


def get_model_and_tokenizer(
    model_name, pad_token=None, pad_token_side="left", torch_dtype=None
):
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer.pad_token = (
        tokenizer.eos_token if pad_token is None else pad_token
    )
    tokenizer.padding_side = pad_token_side
    model.eval()
    return model, tokenizer


def extract_kv_linear_init(model) -> list[torch.Tensor]:
    """Pre-compute a per-KV-head W_linear for every transformer layer."""
    cfg = model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers

    # TODO: Batched pinv across all layers may need per-layer looping to cap RAM for larger models
    W_k_raw = torch.stack(
        [lay.self_attn.k_proj.weight.detach().cpu() for lay in layers]
    )
    W_v_raw = torch.stack(
        [lay.self_attn.v_proj.weight.detach().cpu() for lay in layers]
    )

    W_k = W_k_raw.transpose(-1, -2).float()
    W_v = W_v_raw.transpose(-1, -2).float()

    W_k = W_k.view(len(layers), -1, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    W_v = W_v.view(len(layers), -1, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    W_linear = (torch.linalg.pinv(W_k) @ W_v).to(
        W_k_raw.dtype
    )  # [layers, num_kv_heads, head_dim, head_dim]

    return [W_linear[i] for i in range(len(layers))]
