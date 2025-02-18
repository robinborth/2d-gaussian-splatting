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

        # input layers
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn(activation))

        # hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn(activation))

        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        # self.mlp = nn.Sequential(*layers)
        super().__init__(*layers)

    def forward(self, x):
        return super().forward(x)
