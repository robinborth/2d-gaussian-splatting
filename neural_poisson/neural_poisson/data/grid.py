from typing import Any

import torch


def coord_grid_along_axis(
    axis: str = "x",
    voxel_size: int = 256,
    domain: tuple[float, float] = (-1.0, 1.0),
    default_coord: float = 0.0,
    device: str | Any = "cpu",
):
    """Compute the points locations along the axis."""
    grid_vals = torch.linspace(domain[0], domain[1], voxel_size)
    xs, ys = torch.meshgrid(grid_vals, grid_vals, indexing="ij")
    zs = torch.full_like(xs, default_coord)
    if axis == "x":
        coords = (zs.ravel(), xs.ravel(), ys.ravel())
    if axis == "y":
        coords = (xs.ravel(), zs.ravel(), ys.ravel())
    if axis == "z":
        coords = (xs.ravel(), ys.ravel(), zs.ravel())
    grid = torch.stack(coords, dim=-1).to(device)  # (H, W, 3)
    return grid.reshape(voxel_size, voxel_size, 3)


def coord_grid(
    voxel_size: int = 256,
    domain: tuple[float, float] = (-1.0, 1.0),
    device: str | Any = "cpu",
):
    grid_vals = torch.linspace(domain[0], domain[1], voxel_size)
    xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    grid = torch.stack((xs.ravel(), ys.ravel(), zs.ravel()), dim=-1).to(device)
    return grid.reshape(voxel_size, voxel_size, voxel_size, 3)


def grid_to_frames(grid: torch.Tensor, axis: str = "x"):
    D = grid.shape[0]
    if axis == "x":
        return grid[None].permute(1, 0, 2, 3).expand(D, 3, D, D)
    if axis == "y":
        return grid[None].permute(2, 0, 1, 3).expand(D, 3, D, D)
    if axis == "z":
        return grid[None].permute(3, 0, 1, 2).expand(D, 3, D, D)
    raise AttributeError(f"Specify a valid axis and not {axis=}")
