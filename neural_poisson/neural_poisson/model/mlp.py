from collections import OrderedDict
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


class IdentityActivation(BaseActivation):
    domain: tuple[float, float] = (-torch.inf, torch.inf)

    def forward(self, x):
        return x


class CosineActivation(BaseActivation):
    domain: tuple[float, float] = (-1.0, 1.0)

    def forward(self, x):
        return torch.cos(x)


class SinusActivation(BaseActivation):
    domain: tuple[float, float] = (-1.0, 1.0)

    def forward(self, x):
        return torch.cos(x)


class SirenActivation(BaseActivation):
    domain: tuple[float, float] = (-1.0, 1.0)

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
    domain: tuple[float, float] = (0.0, torch.inf)

    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class SigmoidActivation(BaseActivation, nn.Sigmoid):
    domain: tuple[float, float] = (0.0, 1.0)

    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.xavier_normal_(m.weight)


class TanhActivation(BaseActivation, nn.Tanh):
    domain: tuple[float, float] = (-1.0, 1.0)

    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.xavier_normal_(m.weight)


class GELUActivation(BaseActivation, nn.GELU):
    domain: tuple[float, float] = (0.0, 1.0)

    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.xavier_normal_(m.weight)


################################################################################
# MultiLayerPerceptron (MLP)
################################################################################


class MultiLayerPerceptron(nn.Sequential):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_features: int = 256,
        num_hidden_layers: int = 5,
        activation: Any = ReLUActivation,
        out_activation: Any = SigmoidActivation,
        out_bias: bool = False,
        weight_init: Callable | None = None,
        first_layer_weight_init: Callable | None = None,
        last_layer_weight_init: bool = False,
    ):
        # compute the base activation for init and name
        activation_cls = activation()
        out_activation_cls = out_activation()

        layers: list[Any] = []
        names: list[str] = []

        # input layers
        layers.append(nn.Linear(in_features, hidden_features))
        names.append("layer_in")
        layers.append(activation())
        names.append("activation_in")

        # hidden layers
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            names.append(f"layer_{i+1}")
            layers.append(activation())
            names.append(f"activation_{i+1}")

        # output layer
        layers.append(nn.Linear(hidden_features, out_features, bias=out_bias))
        names.append("layer_out")
        layers.append(out_activation())
        names.append("activation_out")

        # initilize the mlp with the layers
        ordered_dict = OrderedDict(zip(names, layers))
        super().__init__(ordered_dict)

        # initilize the weights of the mlp based on the activation function
        if weight_init is None:
            weight_init = activation_cls.weight_init
        if first_layer_weight_init is None:
            first_layer_weight_init = activation_cls.first_layer_weight_init
        self.weight_init = weight_init
        self.first_layer_weight_init = first_layer_weight_init

        # use special init for last activation based on the final non-linearity
        self.last_layer_weight_init = out_activation_cls.weight_init
        if not last_layer_weight_init:
            self.last_layer_weight_init = None

        # perform weight initialization
        self.register_parameter()

    def register_parameter(self):
        if self.weight_init is not None:
            self.apply(self.weight_init)
        if self.first_layer_weight_init is not None:
            self.layer_in.apply(self.first_layer_weight_init)
        if self.last_layer_weight_init is not None:
            self.layer_out.apply(self.last_layer_weight_init)

    def forward(self, x: torch.Tensor):
        """Returns the output before the activation and afte the activation."""
        logit = None
        out = None
        for name, module in self._modules.items():
            x = module(x)
            if name.startswith("layer"):
                logit = x
            if name.startswith("activation"):
                out = x
        return out, logit
