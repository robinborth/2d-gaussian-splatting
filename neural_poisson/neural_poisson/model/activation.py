import torch
import torch.nn as nn

################################################################################
# Custom Activation Functions
################################################################################


class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


################################################################################
# Activation Functions Utils
################################################################################


def activation_fn(key: str):
    loockup = {
        "relu": nn.ReLU,
        "sin": SinActivation,
        "cos": CosActivation,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if key not in loockup:
        raise AttributeError(f"Unknown activation function {key=}")
    return loockup[key]()
