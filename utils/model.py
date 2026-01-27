from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_and_tokenizer(
    model_name, device, pad_token=None, pad_token_side='left'
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.pad_token = tokenizer.eos_token if pad_token is None else pad_token
    tokenizer.padding_side = pad_token_side
    model.eval()
    return model, tokenizer