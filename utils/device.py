def get_device(model):
    try:
        device = model.device
    except AttributeError:
        try:
            device = model.model.device
        except AttributeError:
            device = "cuda:0"
    return device
