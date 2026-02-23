import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from dotenv import load_dotenv

load_dotenv()


def download_model_to_hub(model_name, **kwargs):
    snapshot_download(model_name, **kwargs)

def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side='left', torch_dtype=None
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

def clone_mlp_params(layer_mlps):
    return [[p.clone() for p in mlp.parameters()] for mlp in layer_mlps]


def load_mlp_params(layer_mlps, params_list):
    for mlp, params in zip(layer_mlps, params_list):
        for p, saved_p in zip(mlp.parameters(), params):
            p.data.copy_(saved_p.data)
