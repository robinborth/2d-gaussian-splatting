from typing import Any, Callable

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

    def compute_gradient(
        self,
        points: torch.Tensor,
        field_values: torch.Tensor | None = None,
        mode: str = "analytical",  # "analytical", "numerical"
        eps: float = 1e-08,
    ):
        """Compute gradients w.r.t to the input point cloud.

        We want to compute dX/dp, which is the gradient of the estimated indicator function
        X w.r.t the input points. However we can only compute dL/dp which computes the
        gradients from a loss scalar. The chain rule is dL/dp = dL/dX * dX/dp. In order to
        compute dX/dp we need to define the loss function do get dL/dX = 1, which results in
        a simple summation of dX, e.g. L=X.sum(), where the derivatives are 1.

        Args:
            points (torch.Tensor): The input points cloud of dim (P, 3).
            field_values (torch.Tensor | None, optional): The scalar output of the field
                that was computed in an differentiable way, if set to None we compute the
                scalar field. Defaults to None.
            mode (str, optional): The gradient computation mode where we either use
                analytical gradients computed with autograd or numerical gradients based
                on finit differences. Defaults to "analytical".
            eps (float): The size of the finit differences, where a larger value results
                in a coarser approximation of the gradient field.

        Returns:
            (torch.Tensor): The gradients of the field w.r.t. to the input point cloud.
        """
        if mode == "analytical":
            if field_values is None:
                points.requires_grad_(True)
                field_values, _ = self.forward(points)
            gradients = torch.autograd.grad(
                outputs=field_values.sum(),
                inputs=points,
                retain_graph=True,
                create_graph=True,
            )[0]
        elif mode == "numerical":
            eps_points: Any = [
                torch.Tensor([eps, 0.0, 0.0]),
                torch.Tensor([-eps, 0.0, 0.0]),
                torch.Tensor([0.0, eps, 0.0]),
                torch.Tensor([0.0, -eps, 0.0]),
                torch.Tensor([0.0, 0.0, eps]),
                torch.Tensor([0.0, 0.0, -eps]),
            ]
            eps_points = torch.stack(eps_points, dim=0).to(points)
            x = points[None, :, :] + eps_points[:, None, :]  # (6, P, 3)
            x, _ = self.forward(x)
            grads: Any = [x[0] - x[1], x[2] - x[3], x[4] - x[5]]
            gradients = torch.stack(grads, dim=-1) / (2 * eps)  # (P, 3)
        else:
            raise ValueError(f"Wrong {mode=}!")
        return gradients

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
