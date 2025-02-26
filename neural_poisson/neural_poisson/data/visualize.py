import math

import matplotlib.pyplot as plt
import open3d as o3d
import torch

from neural_poisson.data.prepare import to_pcd_o3d


def plot_camera_grid(images: list, figsize: int = 6):
    grid_size = int(math.sqrt(len(images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(figsize, figsize))
    axes = axes.flatten()
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_normal_maps(data: list, cameras: list, camera_space: bool = False):
    images = []
    for normal_map, camera in zip(data, cameras):
        if camera_space:
            P = camera.get_world_to_view_transform()
            normal_map = P.transform_normals(normal_map.to(camera))
        normal = (normal_map + 1) / 2
        normal = torch.clip(normal, 0.0, 1.0)
        images.append(normal.detach().cpu().numpy())
    plot_camera_grid(images)


def visualize_point_cloud(points: torch.Tensor, normals: torch.Tensor | None = None):
    pcd = to_pcd_o3d(points=points, normals=normals)
    o3d.visualization.draw_plotly([pcd])
