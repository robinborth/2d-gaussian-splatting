import random

import lightning as L
import torch
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import DataLoader, Dataset

from neural_poisson.data.prepare import (
    extract_surface_data,
    sample_empty_space_points,
    uniform_sphere_cameras,
)

################################################################################
# Different Point Types inside Point Cloud
# 1) P_s: surface points obtained from the depth maps
# 2) P_e: empty points sampled from the empty space
# 3) P_se: empty points sampled from the empty space around the surface
################################################################################


class ShapeNetCoreDatamodule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        dataset: Dataset | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        self.dataset = self.hparams["dataset"]()

    def collate_fn(self, batch):
        assert len(batch) == 1
        return batch[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
            collate_fn=self.collate_fn,
        )


class ShapeNetCoreDataset(Dataset):
    def __init__(
        self,
        path: str,
        device: str = "cuda",
        image_size: int = 256,
        segments: int = 10,
        batch_size: int = 100_000,
        empty_points_per_ray: int = 4,
        close_points_per_ray: int = 2,
        close_points_surface_threshold: float = 0.01,
    ):
        # load the mesh and the cameras
        mesh = load_objs_as_meshes([path], device=device)
        self.cameras = uniform_sphere_cameras(segments=segments, device=device)

        # settings
        self.batch_size = batch_size
        self.points_type = {"surface": 0, "close": 1, "empty": 2}

        # preprocess and load the data
        normals = []
        points_surface = []
        self.normal_maps = []
        self.point_maps = []
        for camera in self.cameras:
            data = extract_surface_data(camera=camera, mesh=mesh, image_size=image_size)
            # extract raw points
            points_surface = data["points"]
            points_close = sample_empty_space_points(
                points=data["points"],
                camera=camera,
                samples=close_points_per_ray,
                surface_threshold=close_points_surface_threshold,
            )
            points_empty = sample_empty_space_points(
                points=data["points"],
                camera=camera,
                samples=empty_points_per_ray,
                surface_threshold=1.0,
            )
            normals.append(data["normals"])
            points_surface.append(data["points"])
            self.normal_maps.append(data["normal_map"])
            self.point_maps.append(data["point_map"])

        self.normals = torch.cat(normals)  # (P, 3)
        self.points = torch.cat(points)  # (P, 3)

    def random_camera(self):
        idx = random.choice(range(len(self.cameras)))
        return self.cameras[idx], idx

    def random_idxs(self):
        return torch.randperm(len(self.points))[: self.batch_size]

    def __len__(self):
        return len(self.points) // self.batch_size

    def __getitem__(self, idx: int):
        idx = self.random_idxs()
        _, camera_idx = self.random_camera()
        return {
            "points": self.points[idx],
            "normals": self.normals[idx],
            "normal_map": self.normal_maps[camera_idx],
            "point_map": self.normal_maps[camera_idx],
            "camera_idx": camera_idx,
        }
