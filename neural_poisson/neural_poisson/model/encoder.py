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


################################################################################
# Grid Encodings
################################################################################


class GridEncoding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        domain: tuple[float, float] = (-1.0, 1.0),
        init_mode: str = "uniform",  # "uniform", "normal", "none"
        activation: str = "relu",
    ):
        # initialize the embedding table
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # input domain of the points
        self.domain = domain
        self.activation = activation
        self.init_mode = init_mode

        # initialize the weights of the embedding tables
        self.apply(self.weight_init)

    @property
    def device(self):
        return self.weight.device

    def compute_voxel_size(self):
        """Compute the voxel size for L levels."""
        return NotImplementedError("Define the voxel resolution.")  # (L, )

    def compute_embedding_idx(self, grid_idx: torch.Tensor):  # (L, P, 8, 3)
        """Convert grid_idx to embedding idxs of a flat Embedding Table."""
        return NotImplementedError("Define the voxel resolution.")  # (L, P, 8)

    def compute_output_dim(self):
        """Compute the final output dimension of the embedding."""
        return NotImplementedError("Define the output dim computation.")

    def weight_init(self, m: nn.Module):
        N = self.compute_output_dim()
        gain = torch.nn.init.calculate_gain(self.activation)
        if self.init_mode == "uniform":
            U = gain * np.sqrt(3 / N)
            torch.nn.init.uniform_(self.weight, -U, U)
        elif self.init_mode == "normal":
            std = gain / np.sqrt(N)
            torch.nn.init.normal_(self.weight, std=std)
        elif self.init_mode == "none":
            pass
        else:
            raise AttributeError(f"Not defined for {self.init_mode=}!")

    def fetch_embeddings(self, grid_idx: torch.Tensor):
        embedding_idx = self.compute_embedding_idx(grid_idx)  # (L, P, 8)
        return super().forward(embedding_idx)  # (L, P, 8, D)

    def convert_into_voxel_cube(self, points: torch.Tensor):
        # ensures that the input points are in the domain of the positional encoding
        assert (points >= self.domain[0]).all()
        assert (points <= self.domain[1]).all()
        # convert points into coord system of a unit cube (0.0, 1.0)
        domain_length = self.domain[1] - self.domain[0]  # (1,)
        normalized_points = (points - self.domain[0]) / domain_length
        # convert the points into the voxel coordinates
        voxel_size = self.compute_voxel_size().reshape(-1, 1, 1)  # (L, 1, 1)
        voxel_size = voxel_size.to(self.device)
        return normalized_points.unsqueeze(dim=0) * voxel_size  # (L, P, 3)

    def points_to_grid_idxs(self, points: torch.Tensor):
        """
        The convention of the hypercube is similar to the cartesian coordinate system.
        Where we start with the origin (0,0,0) at index0. The coordinate system is
        +x right, +y up, +z backward.


        The idxs of the cube can be described with:

            (6)───────(7)
            / |       / |
          (2)───────(3) |
          |  |      |  |
          | (4)────|-|(5)
          | /       | /
          (0)───────(1)

        Where the output of the operation transforms points that are between (-1.0,1.0)
        for each grid resolution N := 2^L (which desribes the number of voxels), to the
        values (0.0, N), together with the integer coordinates of the
        """
        # the idx of the top left voxel grid
        voxel_points = self.convert_into_voxel_cube(points)  # (L, P, 3)
        voxel_corner = torch.floor(voxel_points)  # (L, P, 3)

        # compute all the grids
        grid_idx = torch.stack(
            [
                voxel_corner + torch.tensor([0, 0, 0], device=self.device),
                voxel_corner + torch.tensor([1, 0, 0], device=self.device),
                voxel_corner + torch.tensor([0, 1, 0], device=self.device),
                voxel_corner + torch.tensor([1, 1, 0], device=self.device),
                voxel_corner + torch.tensor([0, 0, 1], device=self.device),
                voxel_corner + torch.tensor([1, 0, 1], device=self.device),
                voxel_corner + torch.tensor([0, 1, 1], device=self.device),
                voxel_corner + torch.tensor([1, 1, 1], device=self.device),
            ],
            dim=-2,
        ).long()  # (L, P, 8, (x,y,z))

        # the weights based on the index0 which can be direclty used
        grid_weights = 1.0 - (voxel_points - voxel_corner)  # (L, P, 3)
        return grid_idx, grid_weights

    def trilinear_interpolation(
        self,
        embeddings: torch.Tensor,  # (L, P, 8, D)
        weights: torch.Tensor,  # (L, P, 3)
    ) -> torch.Tensor:
        """Performs trilinear interpolation based on the weights."""
        # extract the inerpolation weights
        wx = weights[:, :, 0].unsqueeze(dim=-1)  # (L, P, 1)
        wy = weights[:, :, 1].unsqueeze(dim=-1)  # (L, P, 1)
        wz = weights[:, :, 2].unsqueeze(dim=-1)  # (L, P, 1)
        # fetches the embeddings
        e000 = embeddings[:, :, 0, :]  # (L, P, D)
        e100 = embeddings[:, :, 1, :]  # (L, P, D)
        e010 = embeddings[:, :, 2, :]  # (L, P, D)
        e110 = embeddings[:, :, 3, :]  # (L, P, D)
        e001 = embeddings[:, :, 4, :]  # (L, P, D)
        e101 = embeddings[:, :, 5, :]  # (L, P, D)
        e011 = embeddings[:, :, 6, :]  # (L, P, D)
        e111 = embeddings[:, :, 7, :]  # (L, P, D)

        # linear interpolation in the x-direction
        fx0 = wx * e000 + (1.0 - wx) * e100  # (L, P, D)
        fx1 = wx * e010 + (1.0 - wx) * e110  # (L, P, D)
        fx2 = wx * e001 + (1.0 - wx) * e101  # (L, P, D)
        fx3 = wx * e011 + (1.0 - wx) * e111  # (L, P, D)
        # linear interpolation in the y-direction
        fy0 = wy * fx0 + (1.0 - wy) * fx1  # (L, P, D)
        fy1 = wy * fx2 + (1.0 - wy) * fx3  # (L, P, D)
        # linear interpolation in the z-direction
        fz0 = wz * fy0 + (1.0 - wz) * fy1  # (L, P, D)
        # trilinear interpolation
        return fz0  # (L, P, D)

    def forward(self, x: torch.Tensor):
        P, _ = x.shape  # (P, 3)
        # compute the nearby grid location
        grid_idx, grid_weights = self.points_to_grid_idxs(x)  # (L, P, 8, 3), (L, P, 3)
        # compute the embeddings
        grid_emb = self.fetch_embeddings(grid_idx)  # (L, P, 8, D)
        # interplate along the cube grid
        points_emb = self.trilinear_interpolation(grid_emb, grid_weights)  # (L, P, D)
        # merge the embeddings at different levels together
        emb = points_emb.permute(1, 0, 2).reshape(P, -1)  # (P, L*D)
        assert self.compute_output_dim() == emb.shape[1]
        # return the final positional encoding for each point
        return emb


class DenseGridEncoding(GridEncoding):
    def __init__(
        self,
        L: int = 10,
        D: int = 20,
        domain=(-1.0, 1.0),
        init_mode: str = "uniform",  # "uniform", "normal", "none"
        activation: str = "relu",
    ):
        self.L = L  # resolution levels
        self.D = D  # features dimension
        V = self.compute_voxel_size().item()
        super().__init__(
            num_embeddings=V**3,  # (V, V, V) for each node a embedding
            embedding_dim=D,
            domain=domain,
            init_mode=init_mode,
            activation=activation,
        )

    def compute_voxel_size(self):
        """Compute the voxel size for L levels."""
        return torch.tensor([2**self.L])

    def compute_output_dim(self):
        """Compute the final output dimension of the embedding."""
        return self.D

    def compute_embedding_idx(self, grid_idx: torch.Tensor):
        """Convert grid_idx to embedding idxs of a flat Embedding Table."""
        voxel_size = self.compute_voxel_size().to(self.device)
        grid_embs_idx = grid_idx[:, :, :, 0] * voxel_size**0
        grid_embs_idx += grid_idx[:, :, :, 1] * voxel_size**1
        grid_embs_idx += grid_idx[:, :, :, 2] * voxel_size**2
        return grid_embs_idx.long()  # (L, P, 8)


class HashGridEncoding(GridEncoding):
    def __init__(
        self,
        L: int = 10,
        T: int = 2**19,  # 2^14 to 2^24
        F: int = 2,
        N_min: int = 16,
        N_max: int = 512,  # 512 to 524288
        domain: tuple[float, float] = (-1.0, 1.0),
        init_mode: str = "uniform",  # "uniform", "normal", "none"
        activation: str = "relu",
    ):
        self.L = L  # resolution levels
        self.T = T  # max entries per level
        self.F = F  # dimension of features per entry
        self.N_min = N_min
        self.N_max = N_max

        # prime numbers to compute the hashing
        self.pi0 = 1
        self.pi1 = 2_654_435_761
        self.pi2 = 805_459_861

        # compute the lookup table embeddings
        super().__init__(
            num_embeddings=T * L,
            embedding_dim=F,
            domain=domain,
            init_mode=init_mode,
            activation=activation,
        )

    def compute_voxel_size(self):
        """Compute the voxel size for L levels."""
        b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.L - 1))
        return torch.floor(self.N_min * (b ** torch.arange(self.L)))  # (L,)

    def compute_output_dim(self):
        """Compute the final output dimension of the embedding."""
        return self.L * self.F

    def compute_embedding_idx(self, grid_idx: torch.Tensor):
        """Convert grid_idx to embedding idxs of a flat Embedding Table."""
        voxel_size = self.compute_voxel_size().to(self.device)
        voxel_size = voxel_size.unsqueeze(dim=-1).unsqueeze(dim=-1)  # (L, 1, 1)

        # compute embeddings for the coarse resolution without collision
        grid_embs_idx = grid_idx[:, :, :, 0] * voxel_size**0
        grid_embs_idx += grid_idx[:, :, :, 1] * voxel_size**1
        grid_embs_idx += grid_idx[:, :, :, 2] * voxel_size**2
        grid_embs_idx = grid_embs_idx.long()

        # compute the embeddings with the hash value
        h0 = grid_idx[:, :, :, 0] * self.pi0
        h1 = grid_idx[:, :, :, 1] * self.pi1
        h2 = grid_idx[:, :, :, 2] * self.pi2
        h = torch.bitwise_xor(torch.bitwise_xor(h0, h1), h2) % self.T

        # for the fine resolution replace the grid_embeddings
        colision_mask = ((voxel_size + 1) ** 3) > self.T
        colision_mask = colision_mask.squeeze(dim=-1).squeeze(dim=-1)
        grid_embs_idx[colision_mask] = h[colision_mask]
        assert grid_embs_idx.max() < self.T

        # extract for each level the correct entry in the hash table
        levels = torch.arange(self.L).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grid_embs_idx *= levels
        assert grid_embs_idx.max() < self.num_embeddings

        return grid_embs_idx  # (L, P, 8)
