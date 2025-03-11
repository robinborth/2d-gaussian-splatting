import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import wandb
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

from neural_poisson.data.prepare import extract_surface_data, save_mesh_pytorch3d
from neural_poisson.model.implicit import IndicatorFunction


class NeuralPoisson(L.LightningModule):
    def __init__(
        self,
        # encoder module either MLP, DenseGrid, etc.
        indicator_function: IndicatorFunction,
        mode: str = "default",  # "default", "center"
        gradient_compute_mode: str = "analytical",  # "analytical", "numerical"
        gradient_eps: float = 1e-08,
        # loss settings
        lambda_gradient: float = 1.0,
        lambda_surface: float = 1.0,
        lambda_empty_space: float = 1.0,
        # warmup scheduler
        gradient_mode: str = "one",  # "increase", "one"
        close_mode: str = "one",  # "increase", "one"
        indicator_mode: str = "zero",  # "decrease",  "zero"
        gradient_steps: int = 100,
        close_steps: int = 100,
        indicator_steps: int = 100,
        # logging
        log_camera_idxs: list[int] = [0],
        log_metrics: bool = True,
        log_images: bool = True,
        log_optimizer: bool = True,
        log_mesh: bool = True,
        log_metrics_every_n_steps: int = 10,
        log_images_every_n_steps: int = 10,
        log_optimizer_every_n_steps: int = 10,
        log_mesh_every_n_epochs: int = 10,
        # metrics
        num_points_chamfer: int = 100_000,
        # marching cubes settings
        resolution: int = 256,
        domain: tuple[float, float] = (-1.0, 1.0),
        chunk_size: int = 10_000,
        otsu_bins: int = 128,
        # training settings
        optimizer=None,
        scheduler=None,
        monitor: str = "train/loss",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.indicator_function = indicator_function()

        # for default: [0,1] - for center: [-0.5, 0.5]
        assert mode in ["default", "center"]
        self.X_offset = -0.5 if mode == "center" else 0.0
        self.mode = mode

    ################################################################################
    # Optimizer Utils
    ################################################################################

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers()._optimizer  # type: ignore

    @property
    def optimizer_state(self) -> dict[Any, Any]:
        state = self.optimizer.state_dict()["state"]
        if state:
            return state[0]
        return {}

    @property
    def optimizer_param_group(self) -> dict[str, Any]:
        return self.optimizer.param_groups[0]

    def configure_optimizers(self):
        """Default lightning optimizer setup."""
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "monitor": self.hparams["monitor"]}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    ################################################################################
    # Data Utils
    ################################################################################

    @property
    def datamodule(self) -> Any:
        return self.trainer.datamodule  # type: ignore

    ################################################################################
    # Scheduling Utils
    ################################################################################

    def scheduler_step(self, key: str):
        # extract and return default values
        mode = self.hparams[f"{key}_mode"]
        if mode == "zero":
            return 0.0
        if mode == "one":
            return 1.0
        # linear interpolation
        steps = self.hparams[f"{key}_steps"]
        t = max(min(self.trainer.global_step / steps, 1.0), 0.0)  # [0.0, 1.0]
        if mode == "decrease":
            return 1 - t
        if mode == "increase":
            return t
        raise AttributeError(f"There is a wrong {mode=}!")

    def check_logging(self, mode: str = "metrics", batch_idx: int = 0):
        if f"log_{mode}_every_n_steps" in self.hparams:
            return (
                batch_idx % self.hparams[f"log_{mode}_every_n_steps"] == 0
                and self.hparams[f"log_{mode}"]
            )
        return (
            self.trainer.current_epoch % self.hparams[f"log_{mode}_every_n_epochs"] == 0
            and batch_idx == (self.trainer.num_training_batches - 1)
            and self.hparams[f"log_{mode}"]
        )

    ################################################################################
    # Logging Utils
    ################################################################################

    def log_video(self, name: str, frames: torch.Tensor, fps: int = 60):
        frames = (frames * 255).to(torch.uint8)  # (F, C, H, W)
        video = wandb.Video(frames, fps=60)  # type: ignore
        self.logger.experiment.log({name: video})  # type: ignore

    def log_image(self, name: str, image: torch.Tensor):
        img = wandb.Image(image.detach().cpu().numpy())
        self.logger.log_image(name, [img])  # type: ignore

    def log_histogram(self, name: str, x: torch.Tensor, bins: int = 10):
        hist = np.histogram(x.flatten().detach().cpu().numpy(), bins=bins)
        histogram = wandb.Histogram(np_histogram=hist)
        self.logger.experiment.log({name: histogram})  # type: ignore

    def log_histograms(self, histograms: dict[str, torch.Tensor], bins: int = 10):
        histogram = {}
        for name, x in histograms.items():
            hist = np.histogram(x.flatten().detach().cpu().numpy(), bins=bins)
            histogram[name] = wandb.Histogram(np_histogram=hist)
        self.logger.experiment.log(histogram)  # type: ignore

    def compute_basic_stats(self, x: torch.Tensor, name: str):
        stats = {}
        points_norm = torch.linalg.vector_norm(x, dim=-1)
        if points_norm.numel():
            stats[f"{name}_mean"] = points_norm.mean()
            stats[f"{name}_min"] = points_norm.min()
            stats[f"{name}_max"] = points_norm.max()
        return stats

    ################################################################################
    # Logging Scripts
    ################################################################################

    def logging_metrics(self, batch: dict, output: dict, mode: str = "train"):
        self.log(f"{mode}/loss", output["total_loss"], prog_bar=True, logger=False)

        # combine all the metrics together to only send one request to WandB
        unified_output = {}

        # log the different loss information in different sections
        for key, value in output["loss"].items():
            if key in ["surface", "total", "gradient", "empty_space"]:
                name = f"Loss Overview ({mode})"
                unified_output[f"{name}/{key}"] = value
        for key, value in output["loss"].items():
            if key.startswith("empty_space"):
                name = f"Empty Space Loss Overview ({mode})"
                unified_output[f"{name}/{key}"] = value
        for key, value in output["loss"].items():
            if key.startswith("gradient"):
                name = f"Gradient Loss Overview ({mode})"
                unified_output[f"{name}/{key}"] = value
        for key, value in output["loss"].items():
            if key.startswith("logit"):
                name = f"Logit Overview ({mode})"
                unified_output[f"{name}/{key}"] = value

        # log the warmup scheduler for stable training
        for key, value in output["scheduler"].items():
            name = f"Warmup Scheduler Overview ({mode})"
            unified_output[f"{name}/{key}"] = value

        # log the timings of the indicator function and gradient computation
        for key, value in output["time"].items():
            name = f"Time Overview ({mode})"
            unified_output[f"{name}/{key}"] = value

        # log the stats of the gradients of the indicator function
        for key, value in output["stats"].items():
            name = f"Stats Overview ({mode})"
            if key.startswith("dX") and key.endswith("mean"):
                unified_output[f"{name}/{key}"] = value

        # perform the logging
        self.log_dict(unified_output, prog_bar=False)

    def logging_images(self, batch: dict, output: dict, mode: str = "train"):
        # compute the gradient of the indicator function on the point map
        point_map = batch["point_map"].requires_grad_(True)
        x_point_map, _ = self.forward(points=point_map)

        # log the images
        name = f"Image-{batch['camera_idx']:03} ({mode})"
        self.log_image(f"{name}/indicator", x_point_map)
        self.log_image(f"{name}/indicator_gt", batch["indicator_map"])

        # compute the normal and vector maps
        dX_point_map = self.compute_gradient(point_map, x_point_map)
        self.log_image(f"{name}/vector", dX_point_map)
        self.log_image(f"{name}/vector_gt", batch["vector_map"])
        self.log_image(f"{name}/normal_gt", batch["normal_map"])

        # log the axis with the raw values
        bins = self.hparams["otsu_bins"]
        for axis in ["x", "y", "z"]:
            # continuos indicator function
            x = self.indicator_function.compute_axis(axis)
            self.log_image(f"Image-Axis ({mode})/{axis}", x)
            # hard threshold indicator function
            threshold = self.indicator_function.otsu_threshold(x, L=bins)
            X = self.indicator_function.indicator(x, threshold=threshold)
            self.log_image(f"Image Otsu ({mode})/{axis}", X)
            self.log_histogram(f"Histogram Otsu ({mode})/{axis}", x, bins=bins)

    def logging_optimizer(self):
        # log the histogram of the optimizer
        histograms = {}
        for name, param in self.indicator_function.mlp.named_parameters():
            # log the wandb gradients as histograms
            if param.grad is not None:
                histograms[f"Gradients Histogram/{name}"] = param.grad.data
            # log the weights distribution of the layers
            histograms[f"Weights Histogram/{name}"] = param.data
        self.log_histograms(histograms)

        # specifi logging for adam optimizer
        if self.optimizer_state:
            state = self.optimizer_state
            params = self.optimizer_param_group
            m_hat_t = state["exp_avg"] / (1 - params["betas"][0] ** state["step"])
            v_hat_t = state["exp_avg_sq"] / (1 - params["betas"][1] ** state["step"])
            lr_modifier = m_hat_t / (torch.sqrt(v_hat_t) + params["eps"])
            self.log_histogram("Learning Rate Modifier", lr_modifier)

    def log_axis_video(self, grid: torch.Tensor, dim: str = "x"):
        D = grid.shape[0]
        if dim == "x":
            frames = grid[None].permute(1, 0, 2, 3).expand(D, 3, D, D)
        if dim == "y":
            frames = grid[None].permute(2, 0, 1, 3).expand(D, 3, D, D)
        if dim == "z":
            frames = grid[None].permute(3, 0, 1, 2).expand(D, 3, D, D)
        self.log_video(f"Mesh Slicing Video/{dim}", frames)

    def logging_mesh(self, batch: dict, mode: str = "train"):
        # compute the mesh (slow)
        mesh, grid = self.indicator_function.marching_cubes(
            voxel_size=self.hparams["voxel_size"],
            isolevel=0.5,  # 0.0 -> 0.5 -> 1.0
        )

        # logging axis videos
        for axis in ["x", "y", "z"]:
            self.log_axis_video(grid=grid, dim=axis)

        # compute chamfer distance
        chamfer_samples = self.hparams["num_points_chamfer"]
        p1 = sample_points_from_meshes(mesh, chamfer_samples)
        p2 = sample_points_from_meshes(batch["mesh"], chamfer_samples)
        loss, _ = chamfer_distance(p1, p2)
        self.log(f"Metrics ({mode})/chamfer", loss, prog_bar=False)

        # save the mesh to disk
        file_name = f"epoch_{self.trainer.current_epoch:05}.obj"
        path = Path(self.trainer.default_root_dir) / f"mesh/{file_name}"
        save_mesh_pytorch3d(path=path, mesh=mesh)

        # log the mesh for the entire camera logs
        dataset = self.datamodule.dataset(mode=mode)
        for camera_idx in dataset.log_camera_idxs:
            data = extract_surface_data(
                camera=dataset.cameras[camera_idx],
                mesh=mesh,
                image_size=dataset.image_size,
                fill_depth=dataset.fill_depth,
            )
            name = f"Mesh-{camera_idx:03} ({mode})"
            self.log_image(f"{name}/normal", data["normal_map"])
            self.log_image(f"{name}/normal_gt", dataset.normal_maps[camera_idx])
            self.log_image(f"{name}/indicator", data["indicator_map"])
            self.log_image(f"{name}/indicator_gt", dataset.indicator_maps[camera_idx])

    ################################################################################
    # Loss Computation
    ################################################################################

    def l2_loss(self, x: torch.Tensor):
        """Simple L2-Loss."""
        if x.numel() == 0:
            return 0.0
        return (x**2).mean()

    ################################################################################
    # Training Methods
    ################################################################################

    def compute_gradient(self, points: torch.Tensor, X: torch.Tensor):
        """Compute the gradient w.r.t. to the points."""
        return self.indicator_function.compute_gradient(
            points=points,
            field_values=X,
            mode=self.hparams["gradient_compute_mode"],
            eps=self.hparams["gradient_eps"],
        )

    def forward(self, points: torch.Tensor):
        """Evaluates the indicator function for the given points."""
        return self.indicator_function(points)  # X, logits

    def model_step(self, batch: dict):
        # extract the batch information
        p_surface = batch["points_surface"].requires_grad_(True)
        p_close = batch["points_close"].requires_grad_(True)
        p_empty = batch["points_empty"].requires_grad_(True)
        v_surface = batch["vectors_surface"]
        v_close = batch["vectors_close"]
        v_empty = batch["vectors_empty"]

        # evaluate the indicator function
        time_X = time.time()
        x_surface = torch.tensor([])
        x_close = torch.tensor([])
        x_empty = torch.tensor([])
        logit_surface = torch.tensor([])
        logit_close = torch.tensor([])
        logit_empty = torch.tensor([])
        if self.hparams["lambda_surface"] or self.hparams["lambda_gradient"]:
            x_surface, logit_surface = self.forward(points=p_surface)
        if self.hparams["lambda_empty_space"] or self.hparams["lambda_gradient"]:
            x_close, logit_close = self.forward(points=p_close)
            x_empty, logit_empty = self.forward(points=p_empty)
        logit_surface = torch.nan_to_num(logit_surface.mean(), 0.0)
        logit_close = torch.nan_to_num(logit_close.mean(), 0.0)
        logit_empty = torch.nan_to_num(logit_empty.mean(), 0.0)
        time_X = time.time() - time_X

        time_dX = time.time()
        dX_surface = torch.tensor([])
        dX_close = torch.tensor([])
        dX_empty = torch.tensor([])
        if self.hparams["lambda_gradient"]:
            dX_surface = self.compute_gradient(p_surface, x_surface)
            dX_close = self.compute_gradient(p_close, x_close)
            dX_empty = self.compute_gradient(p_empty, x_empty)
        time_dX = time.time() - time_dX

        # surface constraint
        L_surface = 0.0
        if self.hparams["lambda_surface"]:
            L_surface = self.l2_loss(x_surface - 0.5)

        # empty space constraint
        L_empty_space = 0.0
        L_empty_space_close = 0.0
        L_empty_space_empty = 0.0
        if self.hparams["lambda_empty_space"]:
            L_empty_space_close = self.l2_loss(x_close)
            L_empty_space_empty = self.l2_loss(x_empty)
            empty_input = torch.cat([x_close * self.scheduler_step("close"), x_empty])
            L_empty_space = self.l2_loss(empty_input)

        # gradient constraint
        L_gradient = 0.0
        L_gradient_surface = 0.0
        L_gradient_close = 0.0
        L_gradient_empty = 0.0
        if self.hparams["lambda_gradient"]:
            step = self.scheduler_step("gradient")
            L_gradient_surface = self.l2_loss(dX_surface - v_surface)
            L_gradient_close = self.l2_loss(dX_close - v_close)
            L_gradient_empty = self.l2_loss(dX_empty - v_empty)
            gradient_input = [
                dX_surface - v_surface,
                dX_close - v_close,
                dX_empty - v_empty,
            ]
            L_gradient = self.l2_loss(torch.cat(gradient_input)) * step

        # total loss computation
        loss = (
            self.hparams["lambda_surface"] * L_surface
            + self.hparams["lambda_empty_space"] * L_empty_space
            + self.hparams["lambda_gradient"] * L_gradient
        )

        # pre-compute usefull stats for logging
        stats = {}
        stats.update(self.compute_basic_stats(dX_surface, "dX_surface"))
        stats.update(self.compute_basic_stats(dX_close, "dX_close"))
        stats.update(self.compute_basic_stats(dX_empty, "dX_empty"))
        stats.update(self.compute_basic_stats(v_surface, "v_surface"))
        stats.update(self.compute_basic_stats(v_close, "v_close"))
        stats.update(self.compute_basic_stats(v_empty, "v_empty"))

        # prepare output dict
        output = {
            "total_loss": loss,
            "loss": {
                "surface": L_surface,
                "empty_space": L_empty_space,
                "empty_space_close": L_empty_space_close,
                "empty_space_empty": L_empty_space_empty,
                "gradient": L_gradient,
                "gradient_surface": L_gradient_surface,
                "gradient_close": L_gradient_close,
                "gradient_empty": L_gradient_empty,
                "logit_surface": logit_surface,
                "logit_close": logit_close,
                "logit_empty": logit_empty,
                "total": loss,
            },
            "time": {
                "indicator": time_X * 1000,  # in ms
                "gradient": time_dX * 1000,  # in ms
            },
            "scheduler": {
                "gradient": self.scheduler_step("gradient"),
                "close": self.scheduler_step("close"),
                "indicator": self.scheduler_step("indicator"),
            },
            "stats": stats,
        }

        return output

    def training_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "train")
        if self.check_logging("images", batch_idx):
            self.logging_images(batch, output, "train")
        if self.check_logging("mesh", batch_idx):
            self.logging_mesh(batch, "train")
        return output["total_loss"]

    def on_before_optimizer_step(self, optimizer):
        log_steps = self.hparams["log_optimizer_every_n_steps"]
        batch_idx = self.trainer.global_step % log_steps
        if self.check_logging("optimizer", batch_idx):
            self.logging_optimizer()

    @torch.enable_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "val")
        return output["total_loss"]
