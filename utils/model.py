from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from dotenv import load_dotenv

load_dotenv()


def download_model_to_hub(model_name, **kwargs):
    snapshot_download(model_name, **kwargs)


def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side="left"
):
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    tokenizer.pad_token = (
        tokenizer.eos_token if pad_token is None else pad_token
    )
    tokenizer.padding_side = pad_token_side
    model.eval()
    return model, tokenizer
