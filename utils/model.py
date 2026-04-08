import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from dotenv import load_dotenv

load_dotenv()


def download_model_to_hub(model_name, **kwargs):
    snapshot_download(model_name, **kwargs)


def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side="left", torch_dtype=None
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
    """
    Pre-compute W_linear for every transformer layer to initialise residual
    connection in MLPValueCache.
    """
    cfg = model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers

    W_k_raw = torch.stack([lay.self_attn.k_proj.weight.detach().cpu() for lay in layers])
    W_v_raw = torch.stack([lay.self_attn.v_proj.weight.detach().cpu() for lay in layers])

    W_k = W_k_raw.view(len(layers), num_kv_heads, head_dim, cfg.hidden_size).transpose(-1, -2)
    W_v = W_v_raw.view(len(layers), num_kv_heads, head_dim, cfg.hidden_size).transpose(-1, -2)

    # flatten L*H for a single batched pinv call
    LH = len(layers) * num_kv_heads
    W_k_flat = W_k.reshape(LH, cfg.hidden_size, head_dim).float()
    W_v_flat = W_v.reshape(LH, cfg.hidden_size, head_dim).float()

    W_linear = (torch.linalg.pinv(W_k_flat) @ W_v_flat).to(W_k_raw.dtype)

    W_linear = W_linear.view(len(layers), num_kv_heads, head_dim, head_dim)
    return [W_linear[i].unsqueeze(0) for i in range(len(layers))]


def clone_mlp_params(layer_mlps):
    return [[p.clone() for p in mlp.parameters()] for mlp in layer_mlps]


def load_mlp_params(layer_mlps, params_list):
    for mlp, params in zip(layer_mlps, params_list):
        for p, saved_p in zip(mlp.parameters(), params):
            p.data.copy_(saved_p.data)
