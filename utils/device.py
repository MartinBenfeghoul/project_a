import torch 

def get_device(model):
    try:
        device = model.device
    except AttributeError:
        try:
            device = model.model.device
        except AttributeError:
            device = "cuda:0"
    return device


def get_device_type():
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} CUDA chips")
        return "cuda"
    print("WARNING: CUDA not available. Continuing with CPU.")
    return "cpu"

