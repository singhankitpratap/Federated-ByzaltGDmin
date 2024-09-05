import torch
torch.set_default_dtype(torch.float64)


def indicator_function(condition):
    return condition.double()
