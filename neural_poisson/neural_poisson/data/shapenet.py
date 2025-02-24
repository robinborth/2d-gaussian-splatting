import logging
import random
import time

import lightning as L
from torch.utils.data import DataLoader, Dataset

from neural_poisson.data.prepare import (
    compute_chunks,
    extract_points_data,
    load_mesh,
    select_random_points,
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
        dist: float = 1.0,
        fov: float = 60.0,
        fill_depth: str = "zfar",
        # training settings
        device: str = "cuda",
        num_chunks: int = 100,
        chunk_size: int = 100_000,
        surface_chunk_factor: float = 1.0,
        close_chunk_factor: float = 0.5,
        empty_chunk_factor: float = 0.5,
        use_full_chunk: bool = False,  # overrides the chunk_size but not factor
        # subsampling settings
        resolution: float = 0.01,
        domain: tuple[float, float] = (-1.0, 1.0),
        max_surface_points: int = 100_000,
        max_close_points: int = 100_000,
        max_empty_points: int = 100_000,
        # empty space sampling
        empty_points_per_ray: int = 4,
        close_points_per_ray: int = 2,
        close_points_surface_threshold: float = 0.01,
        # vector field settings: "nearest_neighbor", "k_nearest_neighbors", "cluster"
        vector_field_mode: str = "nearest_neighbor",
        vector_field_chunk_size: int = 1_000,
        k: int = 20,
        sigma: float = 1.0,
        chunk_threshold: float = 30,
        # logging settings
        log_camera_idxs: list[int] = [0],
    ):
        self.start_log(f"==> initializing dataset <{self}> ...")
        self.device = device
        self.segments = segments
        self.image_size = image_size
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.surface_chunk_factor = surface_chunk_factor
        self.close_chunk_factor = close_chunk_factor
        self.empty_chunk_factor = empty_chunk_factor
        self.resolution = resolution
        self.log_camera_idxs = log_camera_idxs
        self.fill_depth = fill_depth

        self.start_log(f"\t-> loading mesh from {path} ...")
        self.mesh = load_mesh(path, device=device)
        self.finish_log()

        self.start_log(f"\t-> loading {segments**2} cameras ...")
        self.cameras = uniform_sphere_cameras(
            dist=dist,
            fov=fov,
            segments=segments,
            device=device,
        )
        self.finish_log()

        self.start_log("\t-> extract the surface data ...")
        data = extract_points_data(
            cameras=self.cameras,
            mesh=self.mesh,
            image_size=image_size,
            fill_depth=fill_depth,
            empty_points_per_ray=empty_points_per_ray,
            close_points_per_ray=close_points_per_ray,
            close_points_surface_threshold=close_points_surface_threshold,
        )
        # stores for visualization but no on the GPU
        self.indicator_maps = data["indicator_maps"]
        self.normal_maps = data["normal_maps"]
        self.point_maps = data["point_maps"]
        self.masks = data["masks"]
        self.finish_log()

        self.start_log("\t-> subsample the data with a resolution ...")
        points_surface, points_close, points_empty, normals = subsample_dataset_points(
            points_surface=data["points_surface"],
            points_close=data["points_close"],
            points_empty=data["points_empty"],
            normals=data["normals"],
            resolution=resolution,
            domain=domain,
            max_surface_points=max_surface_points,
            max_close_points=max_close_points,
            max_empty_points=max_empty_points,
        )
        self.points_surface = points_surface
        self.points_close = points_close
        self.points_empty = points_empty
        self.normals_surface = normals
        self.finish_log()
        self.start_log(f"\t-> extract {self.num_surface_points} surface points ...")
        self.start_log(f"\t-> extract {self.num_close_points} close points ...")
        self.start_log(f"\t-> extract {self.num_empty_points} empty points ...")

        if use_full_chunk:
            log.info(f"\t-> set batch size to {self.num_surface_points} ...")
            self.chunk_size = self.num_surface_points

        # prepare the vector field estimation function
        self.start_log(f"\t-> prepare {vector_field_mode} vector field function ...")
        self.vector_fn = select_vector_field_function(
            points=self.points_surface,
            normals=self.normals_surface,
            vector_field_mode=vector_field_mode,
            chunk_size=vector_field_chunk_size,
            k=k,
            sigma=sigma,
            threshold=chunk_threshold,
        )
        self.finish_log()

        self.start_log("\t-> evaluate the vector field ...")
        self.vectors_surface = self.vector_fn(query=points_surface)
        self.vectors_close = self.vector_fn(query=points_close)
        self.finish_log()

        self.start_log("\t-> evaluate the camera vector maps ...")
        self.vector_maps = {}
        for idx in self.log_camera_idxs:
            point_map = self.point_maps[idx]
            vectors = self.vector_fn(query=point_map.reshape(-1, 3))
            self.vector_maps[idx] = vectors.reshape(point_map.shape)
        self.finish_log()

        self.start_log("\t-> prepare the chunks ...")
        self.chunks = {}
        points_surface_chunk, vectors_surface_chunk = compute_chunks(
            num_chunks=self.num_chunks,
            chunk_size=int(self.chunk_size * self.surface_chunk_factor),
            values=[self.points_surface, self.vectors_surface],
        )
        points_close_chunk, vectors_close_chunk = compute_chunks(
            num_chunks=self.num_chunks,
            chunk_size=int(self.chunk_size * self.close_chunk_factor),
            values=[self.points_close, self.vectors_close],
        )
        points_empty_chunk = compute_chunks(
            num_chunks=self.num_chunks,
            chunk_size=int(self.chunk_size * self.empty_chunk_factor),
            values=[self.points_empty],
        )
        self.chunks["points_surface"] = points_surface_chunk
        self.chunks["vectors_surface"] = vectors_surface_chunk
        self.chunks["points_close"] = points_close_chunk
        self.chunks["vectors_close"] = vectors_close_chunk
        self.chunks["points_empty"] = points_empty_chunk
        self.finish_log()

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

    def start_log(self, msg: str):
        self.start_time = time.time()
        log.info(msg)

    def finish_log(self):
        duration = (time.time() - self.start_time) * 1000
        log.info(f"\t-> time taken {duration:.4f} ms")

    ################################################################################
    # Predefined Visualizations
    ################################################################################

    def plot_point_cloud(
        self,
        mode: str = "normals_surface",
        max_samples: int = 100_000,
    ):
        field_mode, points_mode = mode.split("_")
        points = self.__getattribute__(f"points_{points_mode}")
        if field_mode == "points":
            points = select_random_points(points, max_samples=max_samples)
            visualize_point_cloud(points=points)
        else:
            normals = self.__getattribute__(f"{field_mode}_{points_mode}")
            points, normals = select_random_points(points, normals, max_samples)
            visualize_point_cloud(points=points, normals=normals)

    def plot_normal_maps(self, camera_space: bool = False):
        plot_normal_maps(self.normal_maps, self.cameras, camera_space)

    ################################################################################
    # Dataset API
    ################################################################################

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        # extract the current camera for debuging
        camera_idx = random.choice(self.log_camera_idxs)
        # prepare the batch information
        return {
            # point information (point+vector)
            "points_surface": self.chunks["points_surface"][idx].detach().clone(),
            "points_close": self.chunks["points_close"][idx].detach().clone(),
            "points_empty": self.chunks["points_empty"][idx].detach().clone(),
            "vectors_surface": self.chunks["vectors_surface"][idx].detach().clone(),
            "vectors_close": self.chunks["vectors_close"][idx].detach().clone(),
            # camera information
            "camera_idx": camera_idx,
            "camera": self.cameras[camera_idx],
            # images from the camera
            "mask": self.masks[camera_idx].detach().clone(),
            "indicator_map": self.indicator_maps[camera_idx].detach().clone(),
            "normal_map": self.normal_maps[camera_idx].detach().clone(),
            "point_map": self.point_maps[camera_idx].detach().clone(),
            "vector_map": self.vector_maps[camera_idx].detach().clone(),
            # mesh information
            "mesh": self.mesh,
        }
