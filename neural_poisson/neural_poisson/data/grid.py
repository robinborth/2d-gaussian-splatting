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


def point_rasterization(
    points: torch.Tensor,  # (P, 3)
    values: torch.Tensor,  # (P, D)
    voxel_size: int = 256,
    domain: tuple[float, float] = (-1.0, 1.0),
):
    P, _ = points.shape  # (P, 3)
    grid = coord_grid(voxel_size=voxel_size, domain=domain, device=points.device)
    X, Y, Z, _ = grid.shape

    # scale grid values from [-1, 1] to [0, X-1] for x and [0, Y-1] for y and z.
    domain_scale = domain[1] - domain[0]
    x = ((grid[..., 0] - domain[0]) * (X - 1)) / domain_scale
    y = ((grid[..., 1] - domain[0]) * (Y - 1)) / domain_scale
    z = ((grid[..., 2] - domain[0]) * (Z - 1)) / domain_scale

    # get the integer part of the coordinates
    x0 = torch.floor(x).long()
    x1 = torch.ceil(x).long()
    y0 = torch.floor(y).long()
    y1 = torch.ceil(y).long()
    z0 = torch.floor(z).long()
    z1 = torch.ceil(z).long()

    # create masks for values that need clamping
    x0_clamp_mask = (x0 < 0) | (x0 >= X)
    x1_clamp_mask = (x1 < 0) | (x1 >= X)
    y0_clamp_mask = (y0 < 0) | (y0 >= Y)
    y1_clamp_mask = (y1 < 0) | (y1 >= Y)
    z0_clamp_mask = (z0 < 0) | (z0 >= Z)
    z1_clamp_mask = (z1 < 0) | (z1 >= Z)
    clamp_mask = (
        x0_clamp_mask
        | x1_clamp_mask
        | y0_clamp_mask
        | y1_clamp_mask
        | z0_clamp_mask
        | z1_clamp_mask
    )

    # clamp the values
    x0 = torch.clamp(x0, 0, X - 1)  # (B, H, W)
    x1 = torch.clamp(x1, 0, X - 1)  # (B, H, W)
    y0 = torch.clamp(y0, 0, Y - 1)  # (B, H, W)
    y1 = torch.clamp(y1, 0, Y - 1)  # (B, H, W)
    z0 = torch.clamp(z0, 0, Z - 1)  # (B, H, W)
    z1 = torch.clamp(z1, 0, Z - 1)  # (B, H, W)

    # gather pixel values at the corners
    batch_indices = torch.arange(B, device=value.device).view(-1, 1, 1)
    Ia = value[batch_indices, y0, x0, :]  # top-left
    Ib = value[batch_indices, y1, x0, :]  # bottom-left
    Ic = value[batch_indices, y0, x1, :]  # top-right
    Id = value[batch_indices, y1, x1, :]  # bottom-right

    # get the fractional part of the coordinates
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # bilinear interpolate the cornes
    interpolated_values = (
        wa.unsqueeze(-1) * Ia
        + wb.unsqueeze(-1) * Ib
        + wc.unsqueeze(-1) * Ic
        + wd.unsqueeze(-1) * Id
    )

    # zero out the where we need to clamp
    zero_tensor = torch.zeros_like(interpolated_values)
    interpolated_values[clamp_mask] = zero_tensor[clamp_mask]

    return interpolated_values
