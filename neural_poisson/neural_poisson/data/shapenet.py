import logging

import lightning as L
from torch.utils.data import DataLoader, Dataset

from neural_poisson.data.prepare import (
    extract_points_data,
    load_mesh,
    select_random_camera,
    select_random_points,
    select_random_points_and_normals,
    select_vector_field_function,
    subsample_dataset_points,
    uniform_sphere_cameras,
)
from neural_poisson.data.visualize import plot_normal_maps, visualize_point_cloud

log = logging.getLogger()


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
    ################################################################################
    # Load the Dataset: Different Point Types inside Point Cloud
    # 1) surface points: obtained from the depth maps
    # 2) empty points: sampled from the empty space
    # 3) close points: sampled from the empty space around the surface
    ################################################################################
    def __init__(
        self,
        # dataset settings
        path: str = "model_normalized.obj",
        image_size: int = 256,
        segments: int = 10,
        fill_depth: str = "zfar",
        # training settings
        batch_size: int = 100_000,
        epoch_size: int = 100,
        device: str = "cuda",
        # subsampling settings
        resolution: float = 0.01,
        empty_space_max_ratio: float = -1.0,  # (0.0, inf)
        # empty space sampling
        empty_points_per_ray: int = 4,
        close_points_per_ray: int = 2,
        close_points_surface_threshold: float = 0.01,
        # vector field settings: "nearest_neighbor", "k_nearest_neighbors", "cluster"
        vector_field_mode: str = "nearest_neighbor",
        normalize: bool = True,
        chunk_size: int = 1_000,
        k: int = 20,
        sigma: float = 1.0,
        chunk_threshold: float = 30,
    ):
        log.info(f"==> initializing dataset <{self}> ...")
        # store settings
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.segments = segments
        self.image_size = image_size
        self.resolution = resolution

        # prepare the vector field estimation function
        log.info(f"\t==> prepare {vector_field_mode} vector field function ...")
        self.vector_fn = select_vector_field_function(
            vector_field_mode=vector_field_mode,
            normalize=normalize,
            chunk_size=chunk_size,
            k=k,
            sigma=sigma,
            threshold=chunk_threshold,
        )

        log.info(f"\t==> loading mesh from {path} ...")
        self.mesh = load_mesh(path, device=device)

        log.info(f"\t==> loading {segments**2} cameras ...")
        self.cameras = uniform_sphere_cameras(segments=segments, device=device)

        log.info("\t==> extract the surface data ...")
        data = extract_points_data(
            cameras=self.cameras,
            mesh=self.mesh,
            image_size=image_size,
            fill_depth=fill_depth,
        )
        self.indicator_maps = data["indicator_maps"]
        self.normal_maps = data["normal_maps"]
        self.point_maps = data["point_maps"]
        self.masks = data["masks"]

        log.info("\t==> subsample the data with a resolution ...")
        points_surface, points_close, points_empty, normals = subsample_dataset_points(
            points_surface=data["points_surface"],
            points_empty=data["points_empty"],
            points_close=data["points_close"],
            normals=data["normals"],
            resolution=resolution,
            empty_space_max_ratio=empty_space_max_ratio,
        )
        self.points_surface = points_surface
        self.points_close = points_close
        self.points_empty = points_empty
        self.normals_surface = normals

        log.info("\t==> evaluate the vector field ...")
        self.vectors_surface = self.vector_fn(points_surface, normals, points_surface)
        self.vectors_close = self.vector_fn(points_surface, normals, points_close)

    ################################################################################
    # Usefull Properties
    ################################################################################

    @property
    def num_surface_points(self):
        return self.points_surface.shape[0]

    @property
    def num_empty_points(self):
        return self.points_empty.shape[0]

    @property
    def num_close_points(self):
        return self.points_close.shape[0]

    ################################################################################
    # Predefined Visualizations
    ################################################################################

    def plot_point_cloud(
        self,
        mode: str = "surface_normals",
        max_samples: int = 100_000,
    ):
        points_mode, field_mode = mode.split("_")
        points = self.__getattribute__(f"points_{points_mode}")
        normals = self.__getattribute__(f"{field_mode}_{points_mode}")
        points, normals = select_random_points_and_normals(points, normals, max_samples)
        visualize_point_cloud(points=points, normals=normals)

    def plot_normal_maps(self, camera_space: bool = False):
        plot_normal_maps(self.normal_maps, self.cameras, camera_space)

    ################################################################################
    # Dataset API
    ################################################################################

    def __len__(self):
        return self.epoch_size  # TODO change this with a sampler

    def __getitem__(self, idx: int):
        # extract the current camera for debuging
        camera, camera_idx = select_random_camera(self.cameras)

        # sample the points for training
        points_surface, vectors_surface = select_random_points_and_normals(
            points=self.points_surface,
            normals=self.vectors_surface,
            max_samples=self.batch_size,
        )
        points_close, vectors_close = select_random_points_and_normals(
            points=self.points_close,
            normals=self.vectors_close,
            max_samples=self.batch_size // 2,
        )
        points_empty = select_random_points(
            points=self.points_empty,
            max_samples=self.batch_size // 2,
        )

        # prepare the batch information
        return {
            # point information (point+vector)
            "points_surface": points_surface,
            "points_close": points_close,
            "points_empty": points_empty,
            "vectors_surface": vectors_surface,
            "vectors_close": vectors_close,
            # camera information
            "camera_idx": camera_idx,
            "camera": camera,
            # images from the camera
            "mask": self.masks[camera_idx],
            "indicator_map": self.indicator_maps[camera_idx],
            "normal_map": self.normal_maps[camera_idx],
            "point_map": self.point_maps[camera_idx],
            # mesh information
            "mesh": self.mesh,
        }
