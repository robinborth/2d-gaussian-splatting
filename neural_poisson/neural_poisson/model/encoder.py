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


class TanhActivation(BaseActivation, nn.Tanh):
    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.xavier_normal_(m.weight)


class GELUActivation(BaseActivation, nn.GELU):
    def __init__(self, weight_init: bool = True):
        super().__init__()
        self._weight_init = weight_init

    @torch.no_grad()
    def weight_init(self, m: nn.Module):
        if not hasattr(m, "weight") or not self._weight_init:
            return None
        nn.init.xavier_normal_(m.weight)


################################################################################
# MLP
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


################################################################################
# Encodings of Position
################################################################################


class PositionalEncoding(nn.Module):
    def __init__(self, L: int = 10, domain: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.L = L  # frequency levels
        self.domain = domain

    def check_domain(self, x: torch.Tensor):
        """Ensures that the input points are in the domain of the positional encoding."""
        assert (x >= self.domain[0]).all()
        assert (x <= self.domain[1]).all()

    def forward(self, x: torch.Tensor):
        P, _ = x.shape
        self.check_domain(x)  # x is of dim (P, 3)

        # precompute the multiplier
        l = torch.arange(0, self.L, device=x.device)
        freq = (torch.pow(2, l) * torch.pi).reshape(1, 1, -1)  # (1, 1, L)
        sin = torch.sin(freq * x.unsqueeze(-1))  # (P, L)
        cos = torch.cos(freq * x.unsqueeze(-1))  # (P, L)

        # combine together different frequencies together where we have the following
        # (sin(2^0pix)_x, cos(2^0pix)_x, ..., sin(2^0l-1pix)_x, cos(2^l-1pix)_x, ...)
        # (s0_x, c0_x, ..., sl-1_x, cl-1_x, s0_y, ..., s0_z, ...)
        return torch.cat([sin, cos], dim=-1).reshape(P, self.L * 2 * 3)  # (P, L*2*3)


class DenseGridEncoding(nn.Module):
    def __init__(
        self,
        L: int = 10,
        D: int = 20,
        domain: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.L = L  # resolution levels
        self.D = D  # dimension of features per voxel node
        self.voxel_size = 2**self.L
        self.domain = domain

        num_embeddings = self.voxel_size**3  # (V, V, V) for each node a embedding
        self.grid = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.D)

    def check_domain(self, x: torch.Tensor):
        """Ensures that the input points are in the domain of the positional encoding."""
        assert (x >= self.domain[0]).all()
        assert (x <= self.domain[1]).all()

    @property
    def domain_length(self) -> float:
        return self.domain[1] - self.domain[0]

    @property
    def voxel_length(self) -> float:
        """The length of a voxel in world space."""
        return self.domain_length / self.voxel_size

    def points_to_grid_space(self, points: torch.Tensor):
        # points float(-1.0,1.0) -> int[0,voxel_size]
        normalized_points = (points - self.domain[0]) / self.domain_length  # (0.0, 1.0)
        # each voxel has size 1.0 after that operation
        return normalized_points * self.voxel_size  # float(0.0, self.voxel_size)

    def grid_to_points_space(self, grid_idx: torch.Tensor):
        # inverse function to points_to_grid_space
        return ((grid_idx / self.voxel_size) * self.domain_length) + self.domain[0]

    def fetch_embeddings(self, grid_idx: torch.Tensor):
        # for each point we have 8 embeddings ids
        grid_embs_idx = (
            grid_idx[:, :, 0] * self.voxel_size**0
            + grid_idx[:, :, 1] * self.voxel_size**1
            + grid_idx[:, :, 2] * self.voxel_size**2
        )  # (P, 8)
        return self.grid(grid_embs_idx)  # (P, 8, D)

    def points_to_voxel_grid(self, points: torch.Tensor):
        # the idx of the top left voxel grid
        grid_points = torch.floor(self.points_to_grid_space(points))  # (P, 3)
        # compute all the grids
        grid_idx = torch.stack(
            [
                grid_points + torch.tensor([0, 0, 0]),
                grid_points + torch.tensor([1, 0, 0]),
                grid_points + torch.tensor([0, 1, 0]),
                grid_points + torch.tensor([1, 1, 0]),
                grid_points + torch.tensor([0, 0, 1]),
                grid_points + torch.tensor([1, 0, 1]),
                grid_points + torch.tensor([0, 1, 1]),
                grid_points + torch.tensor([1, 1, 1]),
            ],
            dim=1,
        ).long()  # (P, 8, 3) -> (P, 8, (x,y,z))
        # convert into point space
        grid_pos = self.grid_to_points_space(grid_idx)
        return grid_pos, grid_idx

    def trilinear_interpolation(
        self,
        points: torch.Tensor,  # (P, 3)
        Q_pos: torch.Tensor,  # (P, 8, 3)
        Q_emb: torch.Tensor,  # (P, 8, D)
    ):
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        # fetch the value out of the points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # fetch the values out of the grid
        x1 = Q_pos[:, 0, 0]  # (P, 3)
        x2 = Q_pos[:, 1, 0]  # (P, 3)
        y1 = Q_pos[:, 0, 1]  # (P, 3)
        y2 = Q_pos[:, 2, 1]  # (P, 3)
        z1 = Q_pos[:, 0, 2]  # (P, 3)
        z2 = Q_pos[:, 4, 2]  # (P, 3)

        # fetches the embeddings
        q000 = Q_emb[:, 0]
        q100 = Q_emb[:, 1]
        q010 = Q_emb[:, 2]
        q110 = Q_emb[:, 3]
        q001 = Q_emb[:, 4]
        q101 = Q_emb[:, 5]
        q011 = Q_emb[:, 6]
        q111 = Q_emb[:, 7]

        # linear interpolation in the x-direction
        wx0 = ((x2 - x) / (x2 - x1)).unsqueeze(dim=-1)
        wx1 = ((x - x1) / (x2 - x1)).unsqueeze(dim=-1)
        fx0 = wx0 * q000 + wx1 * q100
        fx1 = wx0 * q010 + wx1 * q110
        fx2 = wx0 * q001 + wx1 * q101
        fx3 = wx0 * q011 + wx1 * q111

        # linear interpolation in the y-direction
        wy0 = ((y2 - y) / (y2 - y1)).unsqueeze(dim=-1)
        wy1 = ((y - y1) / (y2 - y1)).unsqueeze(dim=-1)
        fy0 = wy0 * fx0 + wy1 * fx1
        fy1 = wy0 * fx2 + wy1 * fx3

        # linear interpolation in the z-direction
        wz0 = ((z2 - z) / (z2 - z1)).unsqueeze(dim=-1)
        wz1 = ((z - z1) / (z2 - z1)).unsqueeze(dim=-1)
        fz0 = wz0 * fy0 + wz1 * fy1

        return fz0

    def forward(self, x: torch.Tensor):
        P, _ = x.shape
        self.check_domain(x)  # x is of dim (P, 3)
        # compute the nearby grid location
        grid_pos, grid_idx = self.points_to_voxel_grid(x)
        # compute the embeddings
        grid_emb = self.fetch_embeddings(grid_idx)
        # interplate along the cube grid of dim (P, D)
        emb = self.trilinear_interpolation(points=x, Q_pos=grid_pos, Q_emb=grid_emb)
        return emb  # (P, D)


class HashGridEncoding(nn.Module):
    def __init__(
        self,
        L: int = 10,
        T: int = 16384,  # 2^14 to 2^24
        F: int = 2,
        N_min: int = 16,
        N_max: int = 512,  # 512 to 524288
        domain: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.L = L  # resolution levels
        self.T = T  # max entries per level
        self.F = F  # dimension of features per entry
        self.domain = domain

        # compute the lookup table embeddings
        num_embeddings = self.L * self.T
        self.grid = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.F)

    def check_domain(self, x: torch.Tensor):
        """Ensures that the input points are in the domain of the positional encoding."""
        assert (x >= self.domain[0]).all()
        assert (x <= self.domain[1]).all()

    @property
    def domain_length(self) -> float:
        return self.domain[1] - self.domain[0]  # (1,)

    @property
    def voxel_size(self) -> torch.Tensor:
        return 2 ** torch.arange(self.L)  # (L,)

    @property
    def voxel_length(self) -> torch.Tensor:
        """The length of a voxel in world space."""
        return self.domain_length / self.voxel_size  # (L,)

    def points_to_grid_space(self, points: torch.Tensor):
        # points float(-1.0,1.0) -> int[0,voxel_size]
        normalized_points = (points - self.domain[0]) / self.domain_length  # (P, 3)
        # each voxel has size 1.0 after that operation float(0.0, self.voxel_size)
        voxel_size = self.voxel_size.reshape(-1, 1, 1)
        return normalized_points.unsqueeze(dim=0) * voxel_size  # (L, P, 3)

    def grid_to_points_space(self, grid_idx: torch.Tensor):
        # inverse function to points_to_grid_space
        grid_points = grid_idx / self.voxel_size.reshape(-1, 1, 1, 1)
        return (grid_points * self.domain_length) + self.domain[0]

    def fetch_embeddings(self, grid_idx: torch.Tensor):
        # for each point we have 8 embeddings ids
        grid_embs_idx = (
            grid_idx[:, :, :, 0] * self.voxel_size.reshape(-1, 1, 1) ** 0
            + grid_idx[:, :, :, 1] * self.voxel_size.reshape(-1, 1, 1) ** 1
            + grid_idx[:, :, :, 2] * self.voxel_size.reshape(-1, 1, 1) ** 2
        )  # (L, P, 8)
        return self.grid(grid_embs_idx)  # (L, P, 8, D)

    def points_to_voxel_grid(self, points: torch.Tensor):
        # the idx of the top left voxel grid
        grid_points = torch.floor(self.points_to_grid_space(points))  # (L, P, 3)
        # compute all the grids
        grid_idx = torch.stack(
            [
                grid_points + torch.tensor([0, 0, 0]),
                grid_points + torch.tensor([1, 0, 0]),
                grid_points + torch.tensor([0, 1, 0]),
                grid_points + torch.tensor([1, 1, 0]),
                grid_points + torch.tensor([0, 0, 1]),
                grid_points + torch.tensor([1, 0, 1]),
                grid_points + torch.tensor([0, 1, 1]),
                grid_points + torch.tensor([1, 1, 1]),
            ],
            dim=-2,
        ).long()  # (L, P, 8, 3) -> (L, P, 8, (x,y,z))
        # convert into point space
        grid_pos = self.grid_to_points_space(grid_idx)
        return grid_pos, grid_idx

    def trilinear_interpolation(
        self,
        points: torch.Tensor,  # (P, 3)
        Q_pos: torch.Tensor,  # (L, P, 8, 3)
        Q_emb: torch.Tensor,  # (L, P, 8, D)
    ):
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        # fetch the value out of the points
        x = points[:, 0]  # (P,)
        y = points[:, 1]  # (P,)
        z = points[:, 2]  # (P,)

        # fetch the values out of the grid
        x1 = Q_pos[:, :, 0, 0]  # (L, P)
        x2 = Q_pos[:, :, 1, 0]  # (L, P)
        y1 = Q_pos[:, :, 0, 1]  # (L, P)
        y2 = Q_pos[:, :, 2, 1]  # (L, P)
        z1 = Q_pos[:, :, 0, 2]  # (L, P)
        z2 = Q_pos[:, :, 4, 2]  # (L, P)

        # fetches the embeddings
        q000 = Q_emb[:, :, 0]  # (L, P, D)
        q100 = Q_emb[:, :, 1]  # (L, P, D)
        q010 = Q_emb[:, :, 2]  # (L, P, D)
        q110 = Q_emb[:, :, 3]  # (L, P, D)
        q001 = Q_emb[:, :, 4]  # (L, P, D)
        q101 = Q_emb[:, :, 5]  # (L, P, D)
        q011 = Q_emb[:, :, 6]  # (L, P, D)
        q111 = Q_emb[:, :, 7]  # (L, P, D)

        # linear interpolation in the x-direction
        wx0 = ((x2 - x) / (x2 - x1)).unsqueeze(dim=-1)  # (L, P, 1)
        wx1 = ((x - x1) / (x2 - x1)).unsqueeze(dim=-1)  # (L, P, 1)
        fx0 = wx0 * q000 + wx1 * q100  # (L, P, D)
        fx1 = wx0 * q010 + wx1 * q110  # (L, P, D)
        fx2 = wx0 * q001 + wx1 * q101  # (L, P, D)
        fx3 = wx0 * q011 + wx1 * q111  # (L, P, D)

        # linear interpolation in the y-direction
        wy0 = ((y2 - y) / (y2 - y1)).unsqueeze(dim=-1)  # (L, P, 1)
        wy1 = ((y - y1) / (y2 - y1)).unsqueeze(dim=-1)  # (L, P, 1)
        fy0 = wy0 * fx0 + wy1 * fx1  # (L, P, D)
        fy1 = wy0 * fx2 + wy1 * fx3  # (L, P, D)

        # linear interpolation in the z-direction
        wz0 = ((z2 - z) / (z2 - z1)).unsqueeze(dim=-1)  # (L, P, 1)
        wz1 = ((z - z1) / (z2 - z1)).unsqueeze(dim=-1)  # (L, P, 1)
        fz0 = wz0 * fy0 + wz1 * fy1  # (L, P, D)

        return fz0  # (L, P, D)

    def forward(self, x: torch.Tensor):
        P, _ = x.shape  # (P, 3)
        self.check_domain(x)  # x is of dim (P, 3)
        # compute the nearby grid location
        grid_pos, grid_idx = self.points_to_voxel_grid(x)
        # compute the embeddings
        grid_emb = self.fetch_embeddings(grid_idx)
        # interplate along the cube grid of dim (P, D)
        emb = self.trilinear_interpolation(points=x, Q_pos=grid_pos, Q_emb=grid_emb)
        return emb.permute(1, 0, 2).reshape(P, -1)  # (P, D)
