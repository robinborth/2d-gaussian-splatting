from collections import OrderedDict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

################################################################################
# Activation Functions
################################################################################


class BaseActivation(nn.Module):
    @property
    def name(self):
        return self.__class__.__name__.replace("Activation", "").lower()

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        return None

    @torch.no_grad()
    def first_layer_weight_init(self, m: nn.Module):
        return None


class CosineActivation(BaseActivation):
    def forward(self, x):
        return torch.cos(x)


class SinusActivation(BaseActivation):
    def forward(self, x):
        return torch.cos(x)


class SirenActivation(BaseActivation):
    def __init__(
        self,
        w: float = 30.0,
        weight_init: bool = True,
        first_layer_weight_init: bool = True,
    ):
        super().__init__()
        self.w = w
        self._weight_init = weight_init
        self._first_layer_weight_init = first_layer_weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        num_input = m.weight.size(-1)
        U = np.sqrt(6 / num_input) / self.w
        m.weight.uniform_(-U, U)

    @torch.no_grad()
    def first_layer_weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._first_layer_weight_init:
            return None
        num_input = m.weight.size(-1)
        U = 1 / num_input
        m.weight.uniform_(-U, U)

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w * x)


class ReLUActivation(BaseActivation, nn.ReLU):
    pass


class TanhActivation(BaseActivation, nn.Tanh):
    pass


class GELUActivation(BaseActivation, nn.GELU):
    pass


################################################################################
# Initalizations Functions
################################################################################


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_features: int = 256,
        num_hidden_layers: int = 5,
        activation: Any = ReLUActivation,
        out_activation: bool = False,
        out_bias: bool = False,
        weight_init: Callable | None = None,
        first_layer_weight_init: Callable | None = None,
    ):
        # compute the base activation for init and name
        activation_cls = activation()

        layers: list[Any] = []
        names: list[str] = []

        # input layers
        layers.append(nn.Linear(in_features, hidden_features))
        names.append("layer_0")
        layers.append(activation())
        names.append(f"{activation_cls.name}_0")

        # hidden layers
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            names.append(f"layer_{i+1}")
            layers.append(activation())
            names.append(f"{activation_cls.name}_{i+1}")

        # output layer
        layers.append(nn.Linear(hidden_features, out_features, bias=out_bias))
        names.append(f"layer_{i+2}")
        if out_activation:
            layers.append(activation())
            names.append(f"{activation_cls.name}_{i+2}")

        # initilize the mlp with the layers
        ordered_dict = OrderedDict(zip(names, layers))
        super().__init__(ordered_dict)

        # initilize the weights of the mlp based on the activation function
        if weight_init is None:
            weight_init = activation_cls.weight_init
        if first_layer_weight_init is None:
            first_layer_weight_init = activation_cls.first_layer_weight_init

        if weight_init is not None:
            self.apply(weight_init)
        if first_layer_weight_init is not None:
            self.layer_0.apply(first_layer_weight_init)

    def forward(self, x):
        return super().forward(x)
