from typing import Callable

import torch
import torch.nn as nn

from neural_poisson.model.encoding import Encoding
from neural_poisson.model.mlp import MultiLayerPerceptron


class ImplicitField(nn.Module):
    """A scalar field that transforms points (P,3) into a scalar value (P,)."""

    def __init__(
        self,
        mlp: Callable[..., MultiLayerPerceptron],
        encoding: Callable[..., Encoding],
        device: str = "cuda",
    ):
        super().__init__()
        self.encoding = encoding()
        # prepare the mlp based on the encoding
        D = self.encoding.compute_output_dim()
        self.mlp = mlp(in_features=D)
        # re-initialize the first layer depending on the encoding
        if self.encoding.reinitalize_first_layer:
            self.mlp.first_layer_weight_init = self.mlp.weight_init
            self.mlp.register_parameter()

        # initilize to the correct device
        self.to(device)

    def forward(self, x: torch.Tensor):
        # handle arbitrary points shapes
        shape = x.shape[:-1]
        points = x.reshape(-1, 3)  # (P, 3)
        # positional encoding
        encoding = self.encoding(points)  # (P, D)
        # predicts
        out, logits = self.mlp(encoding)  # (P,1), (P,1)
        # transform into original shape
        out = out.reshape(*shape)  # (P,)
        logits = logits.reshape(*shape)  # (P,)
        return out, logits


class IndicatorFunction(ImplicitField):
    def __init__(
        self,
        mlp: Callable[..., MultiLayerPerceptron],
        encoding: Callable[..., Encoding],
        mode: str = "default",  # "default", "center"
        device: str = "cuda",
    ):
        """
        The indicator takes as input a point cloud of dim (P, 3) and produces the
        logits of the indicator function which are then encoded with a tanh/sin
        function to be in the range of (-0.5, 0.5).
        """
        # initilize the indicator function
        super().__init__(mlp=mlp, encoding=encoding)

        # for default: [0,1] - for center: [-0.5, 0.5]
        assert mode in ["default", "center"]
        self.X_offset = -0.5 if mode == "center" else 0.0
        self.isolevel = 0.0 if mode == "center" else 0.5
        self.mode = mode

        # initilize to the correct device
        self.to(device)

    def forward(self, points: torch.Tensor):
        """Evaluates the indicator function for the given points."""
        X, logits = super().forward(points)  # the logits of the encoder of dim (P,)
        # transforms into indicator function [0, 1.0]
        d_min, d_max = self.mlp.activation_out.domain
        X = (X - d_min) / (d_max - d_min)
        # transform into the required range : [0, 1] <-> [-0.5, 0.5]
        X = X + self.X_offset
        return X, logits  # (P,), (P,)
