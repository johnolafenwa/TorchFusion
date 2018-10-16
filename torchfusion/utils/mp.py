import torch

def half(model):
    params_copy = [param.clone() for param in model.parameters()]
    for param in params_copy:
        param.requires_grad = True

    return model.half(), params_copy