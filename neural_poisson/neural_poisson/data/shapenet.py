import random

import lightning as L
import torch
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import DataLoader, Dataset

from neural_poisson.data.prepare import (
    extract_surface_data,
    sample_empty_space_points,
    select_vector_field_function,
    subsample_points,
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
        # dataset settings
        path: str = "model_normalized.obj",
        image_size: int = 256,
        segments: int = 10,
        # training settings
        batch_size: int = 100_000,
        device: str = "cuda",
        # subsampling settings
        resolution: float = 0.01,
        # empty space sampling
        empty_points_per_ray: int = 4,
        close_points_per_ray: int = 2,
        close_points_surface_threshold: float = 0.01,
        empty_space_max_ratio: float = -1.0,  # (0.0, inf)
        # vector field settings: "nearest_neighbor", "k_nearest_neighbors", "cluster"
        vector_field_mode: str = "nearest_neighbor",
        normalize: bool = True,
        chunk_size: int = 1_000,
        k: int = 20,
        sigma: float = 1.0,
        chunk_threshold: float = 30,
    ):
        # load the mesh and the cameras
        mesh = load_objs_as_meshes([path], device=device)
        self.cameras = uniform_sphere_cameras(segments=segments, device=device)

        # settings
        self.batch_size = batch_size
        self.points_type_dict = {"surface": 0, "close": 1, "empty": 2}

        # prepare the vector field estimation function
        self.estimate_vector_field = select_vector_field_function(
            vector_field_mode=vector_field_mode,
            normalize=normalize,
            chunk_size=chunk_size,
            k=k,
            sigma=sigma,
            threshold=chunk_threshold,
        )

        # extract the surface data: point_maps, normal_maps, points, normals
        self.data = []
        for camera in self.cameras:
            data = extract_surface_data(camera=camera, mesh=mesh, image_size=image_size)
            self.data.append(data)

        # collect all the surface points and sample in the empty space
        _normals = []
        _points_surface = []
        _points_close = []
        _points_empty = []
        for data, camera in zip(self.data, self.cameras):
            # extract the surface points and normals
            _points_surface.append(data["points"])
            _normals.append(data["normals"])
            # extract the close surface points
            _points = sample_empty_space_points(
                points=data["points"],
                camera=camera,
                samples=close_points_per_ray,
                surface_threshold=close_points_surface_threshold,
            )
            _points_close.append(_points)
            # extract the empty space points
            points = sample_empty_space_points(
                points=data["points"],
                camera=camera,
                samples=empty_points_per_ray,
                surface_threshold=1.0,
            )
            _points_empty.append(points)
        normals = torch.cat(_normals)  # (P, 3)
        points_surface = torch.cat(_points_surface)  # (P, 3)
        points_empty = torch.cat(_points_empty)  # (P, 3)
        points_close = torch.cat(_points_close)  # (P, 3)

        # subsample points to the desired resolution
        points_surface, normals = subsample_points(
            points=points_surface,
            normals=normals,
            resolution=resolution,
        )
        points_close = subsample_points(points=points_close, resolution=resolution)

        # subsample empty points and ensure similar ratio
        points_empty = subsample_points(points=points_empty, resolution=resolution)
        if empty_space_max_ratio > 0:
            surface_count = points_surface.shape[0] + points_close.shape[0]
            max_empty_points = int(surface_count * empty_space_max_ratio)
            max_empty_points = min(max_empty_points, points_empty.shape[0])
            indices = torch.randperm(points_empty.shape[0])[:max_empty_points]
            points_empty = points_empty[indices]

        # compute the vectors for all the points
        vectors_surface = self.estimate_vector_field(
            points=points_surface,
            normals=normals,
            query=points_surface,
        )
        vectors_empty = self.estimate_vector_field(
            points=points_surface,
            normals=normals,
            query=points_empty,
        )
        vectors_close = self.estimate_vector_field(
            points=points_surface,
            normals=normals,
            query=points_close,
        )

        # safe the vector and points
        self.noramls_surface = normals
        self.points_surface = points_surface
        self.points_close = points_close
        self.points_empty = points_empty
        self.vectors_surface = vectors_surface
        self.vectors_close = vectors_close
        self.vectors_empty = vectors_empty

    def random_camera(self):
        idx = random.choice(range(len(self.cameras)))
        return self.cameras[idx], idx

    def random_selection(
        self,
        points: torch.Tensor,
        vectors: torch.Tensor,
        batch_size: int,
    ):
        assert points.shape == vectors.shape
        idx = torch.randperm(len(points))[:batch_size]
        return points[idx], vectors[idx]

    def __len__(self):
        return len(self.points) // self.batch_size

    def __getitem__(self, idx: int):
        points_surface, vectors_surface = self.random_selection(
            points=self.points_surface,
            vectors=self.vectors_surface,
            batch_size=self.batch_size,
        )
        points_close, vectors_close = self.random_selection(
            points=self.points_close,
            vectors=self.vectors_close,
            batch_size=self.batch_size // 2,
        )
        points_empty, vectors_empty = self.random_selection(
            points=self.points_empty,
            vectors=self.vectors_empty,
            batch_size=self.batch_size // 2,
        )
        _, camera_idx = self.random_camera()
        return {
            "points_surface": points_surface,
            "points_close": points_close,
            "points_empty": points_empty,
            "vectors_surface": vectors_surface,
            "vectors_close": vectors_close,
            "vectors_empty": vectors_empty,
            "normal_map": self.data[camera_idx]["normal_map"],
            "point_map": self.data[camera_idx]["point_map"],
            "camera_idx": camera_idx,
        }
