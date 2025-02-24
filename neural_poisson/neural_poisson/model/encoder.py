from collections import OrderedDict
from typing import Any

import torch.nn as nn

from neural_poisson.model.activation import activation_fn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 5,
        activation: str = "relu",
    ):
        layers: list[Any] = []
        names: list[str] = []

        # input layers
        layers.append(nn.Linear(in_dim, hidden_dim))
        names.append("layer_0")
        layers.append(activation_fn(activation))
        names.append(f"{activation}_0")

        # hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            names.append(f"layer_{i+1}")
            layers.append(activation_fn(activation))
            names.append(f"{activation}_{i+1}")

        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        names.append(f"layer_{i+2}")

        ordered_dict = OrderedDict(zip(names, layers))
        super().__init__(ordered_dict)

    def forward(self, x):
        return super().forward(x)
