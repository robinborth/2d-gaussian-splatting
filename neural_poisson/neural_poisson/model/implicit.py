import logging
from typing import Any, Callable

import torch
import torch.nn as nn
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.structures import Meshes

from neural_poisson.data.grid import coord_grid, coord_grid_along_axis
from neural_poisson.model.encoding import Encoding
from neural_poisson.model.mlp import MultiLayerPerceptron

log = logging.getLogger()


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

    @property
    def device(self):
        return self.mlp.device

    def compute_axis(self, axis: str = "x", voxel_size: int = 256) -> torch.Tensor:
        grid = coord_grid_along_axis(
            axis=axis,
            voxel_size=voxel_size,
            domain=self.encoding.domain,
            default_coord=0.0,
            device=self.device,
        )
        return self.forward(grid)[0]

    def compute_grid(
        self,
        voxel_size: int = 256,
        chunk_size: int = 10_000,
    ) -> torch.Tensor:
        # fetch the point on the grid lattice
        N = voxel_size
        grid = coord_grid(voxel_size=N, domain=self.encoding.domain).reshape(-1, 3)
        # evaluate the indicator function on the grid structure
        field = []
        for points in torch.split(grid, chunk_size):
            x, _ = self.forward(points.to(self.device))
            field.append(x.detach().cpu())
        return torch.cat(field).reshape(N, N, N)

    def compute_gradient(
        self,
        points: torch.Tensor,
        field_values: torch.Tensor | None = None,
        mode: str = "analytical",  # "analytical", "numerical"
        eps: float = 1e-08,
    ) -> torch.Tensor:
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
            x = points.unsqueeze(-2) + eps_points  # (6, P, 3)
            x, _ = self.forward(x)
            grads: Any = [
                x[..., 0] - x[..., 1],
                x[..., 2] - x[..., 3],
                x[..., 4] - x[..., 5],
            ]
            gradients = torch.stack(grads, dim=-1) / (2 * eps)  # (P, 3)
        else:
            raise ValueError(f"Wrong {mode=}!")
        return gradients

    def marching_cubes(
        self,
        voxel_size: int = 256,
        isolevel: float = 0.0,
    ) -> tuple[Meshes, torch.Tensor]:
        # evaluate the field on the grid nodes
        grid = self.compute_grid(voxel_size=voxel_size)  # (W, H, D)

        # ensures that we have a valid isolevel and can extract a mesh
        if isolevel > grid.max() or isolevel < grid.min():
            old_isolevel = isolevel
            isolevel = (grid.max().item() - grid.min().item()) / 2
            log.warning(
                f"Isolevel is set to: {old_isolevel} and field is in range "
                f"({grid.min()}, {grid.max()})! Change the isolevel to: {isolevel}"
            )

        # perform marching cubes with pytorch3d
        sdf_grid = grid.permute(2, 1, 0)[None]  # (W, H, D) -> (1, D, H, W)
        verts, faces = marching_cubes(sdf_grid, isolevel=isolevel)
        mesh = Meshes(verts=verts, faces=faces).to(self.device)

        # returns the extracted mesh and the grid
        return mesh, grid

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        device: str = "cuda",
    ):
        """
        The indicator takes as input a point cloud of dim (P, 3) and produces the
        logits of the indicator function which are then encoded with a tanh/sin
        function to be in the range of (0.0, 1.0).
        """
        # initilize the indicator function
        super().__init__(mlp=mlp, encoding=encoding)

        # initilize to the correct device
        self.to(device)

    @torch.no_grad()
    def otsu_threshold(self, x: torch.Tensor, L: int = 128) -> float:
        """Returns the otsu threshold that seperates gray-level histograms."""
        counts, bins = torch.histogram(x.flatten().detach().cpu(), bins=L)
        p = counts / counts.sum()

        # compute the centers of the bins
        bin_size = (bins[1:] - bins[:-1]).mean()
        bin_center = bins[:-1] + (bin_size / 2)  # (L,)

        # computes p[:k] * bin_center[:k]).sum() for each k
        mK = torch.cumsum(p * bin_center, dim=0)  # (L, )
        # computes p[:k].sum() for each k
        wK = torch.cumsum(p, dim=0)  # (L, )
        # computes the criterion form otsu paper (18)
        criterions = ((mK[-1] * wK - mK) ** 2) / (wK * (1 - wK) + 1e-10)

        # compute the otsu-threshold
        threshold = bin_center[criterions.argmax()]
        return threshold.item()

    def indicator(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        assert 0.0 <= threshold <= 1.0
        x = x.clone()
        x[x < threshold] = 0.0
        x[x >= threshold] = 1.0
        return x

    def forward(self, points: torch.Tensor, threshold: float | None = None):
        """Evaluates the indicator function for the given points."""
        X, logits = super().forward(points)  # the logits of the encoder of dim (P,)

        # transforms into indicator function [0, 1.0]
        d_min, d_max = self.mlp.activation_out.domain
        X = (X - d_min) / (d_max - d_min)
        assert X.min() >= 0.0 and X.max() <= 1.0

        # hard indicator threshold with {0, 1}
        if threshold is not None:
            self.indicator(X, threshold=threshold)

        return X, logits  # (P,), (P,)
