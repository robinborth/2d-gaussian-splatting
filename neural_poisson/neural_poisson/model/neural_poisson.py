import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import wandb
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from neural_poisson.data.grid import grid_to_frames
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
        isolevel_mode: str = "default",  # "default", "otsu"
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
        # combine all the metrics together to only send one request to WandB
        metrics = {}

        # vector field basic stats from the dataset
        vectors = {}
        vectors.update(self.compute_basic_stats(batch["vectors_surface"], "v_surface"))
        vectors.update(self.compute_basic_stats(batch["vectors_close"], "v_close"))
        vectors.update(self.compute_basic_stats(batch["vectors_empty"], "v_empty"))
        for key, value in vectors.items():
            name = f"Stats Overview ({mode})"
            if key.startswith("vectors") and key.endswith("mean"):
                metrics[f"{name}/{key}"] = value

        # gradient field basic stats from the indicator function
        gradients = {}
        gradients.update(self.compute_basic_stats(output["dX_surface"], "dX_surface"))
        gradients.update(self.compute_basic_stats(output["dX_close"], "dX_close"))
        gradients.update(self.compute_basic_stats(output["dX_empty"], "dX_empty"))
        for key, value in gradients.items():
            name = f"Stats Overview ({mode})"
            if key.startswith("dX") and key.endswith("mean"):
                metrics[f"{name}/{key}"] = value

        # raw logits before final activation from the indicator function
        logits = {}
        logits["surface"] = torch.nan_to_num(output["logit_surface"].mean(), 0.0)
        logits["close"] = torch.nan_to_num(output["logit_close"].mean(), 0.0)
        logits["empty"] = torch.nan_to_num(output["logit_empty"].mean(), 0.0)
        for key, value in logits.items():
            name = f"Logit Overview ({mode})"
            metrics[f"{name}/{key}"] = value

        # log the warmup scheduler for stable training
        scheduler = {}
        scheduler["gradient"] = self.scheduler_step("gradient")
        scheduler["close"] = self.scheduler_step("close")
        scheduler["indicator"] = self.scheduler_step("indicator")
        for key, value in scheduler.items():
            name = f"Warmup Scheduler Overview ({mode})"
            metrics[f"{name}/{key}"] = value

        # log the timings of the indicator function and gradient computation
        for key, value in output.items():
            if key.startswith("time"):
                name = f"Time Overview ({mode})"
                metrics[f"{name}/{key}"] = value

        # perform the logging
        self.log_dict(metrics, prog_bar=False)

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

    def logging_mesh(self, batch: dict, output: dict, mode: str = "train"):
        # logging axis videos
        for axis in ["x", "y", "z"]:
            frames = grid_to_frames(grid=output["grid"], axis=axis)
            self.log_video(f"Mesh Slicing Video/{axis}", frames)

        # log the mesh for the entire camera logs
        dataset = self.datamodule.dataset(mode=mode)
        for camera_idx in dataset.log_camera_idxs:
            data = extract_surface_data(
                camera=dataset.cameras[camera_idx],
                mesh=output["mesh"],
                image_size=dataset.image_size,
                fill_depth=dataset.fill_depth,
            )
            name = f"Mesh-{camera_idx:03} ({mode})"
            self.log_image(f"{name}/normal", data["normal_map"])
            self.log_image(f"{name}/normal_gt", dataset.normal_maps[camera_idx])
            self.log_image(f"{name}/indicator", data["indicator_map"])
            self.log_image(f"{name}/indicator_gt", dataset.indicator_maps[camera_idx])

        # save the mesh to disk
        file_name = f"epoch_{self.trainer.current_epoch:05}.obj"
        path = Path(self.trainer.default_root_dir) / f"mesh/{file_name}"
        save_mesh_pytorch3d(path=path, mesh=output["mesh"])

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

    def marching_cubes(self) -> tuple[Meshes, torch.Tensor]:
        return self.indicator_function.marching_cubes(
            voxel_size=self.hparams["voxel_size"],
            isolevel=self.indicator_function.isolevel,  # 0.0 -> 0.5 -> 1.0
            L=self.hparams["otsu_bins"],
            mode=self.hparams["isolevel_mode"],
        )

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

    ################################################################################
    # Training Methods
    ################################################################################

    def model_step(self, batch: dict, batch_idx: int, mode: str = "train"):
        # extract the batch information
        p_surface = batch["points_surface"].requires_grad_(True)
        p_close = batch["points_close"].requires_grad_(True)
        p_empty = batch["points_empty"].requires_grad_(True)

        # prepare the output information
        output: dict[str, Any] = {}
        output["x_surface"] = torch.tensor([])
        output["x_close"] = torch.tensor([])
        output["x_empty"] = torch.tensor([])
        output["logit_surface"] = torch.tensor([])
        output["logit_close"] = torch.tensor([])
        output["logit_empty"] = torch.tensor([])
        output["dX_surface"] = torch.tensor([])
        output["dX_close"] = torch.tensor([])
        output["dX_empty"] = torch.tensor([])
        output["mesh"] = None
        output["grid"] = torch.tensor([])

        # evaluate the indicator function
        output["time_X"] = time.time()
        if self.hparams["lambda_surface"] or self.hparams["lambda_gradient"]:
            output["x_surface"], output["logit_surface"] = self.forward(p_surface)
        if self.hparams["lambda_empty_space"] or self.hparams["lambda_gradient"]:
            output["x_close"], output["logit_close"] = self.forward(p_close)
            output["x_empty"], output["logit_empty"] = self.forward(p_empty)
        output["time_X"] = (time.time() - output["time_X"]) * 1000

        # compute the gradient of the indicator function
        output["time_dX"] = time.time()
        if self.hparams["lambda_gradient"]:
            output["dX_surface"] = self.compute_gradient(p_surface, output["x_surface"])
            output["dX_close"] = self.compute_gradient(p_close, output["x_close"])
            output["dX_empty"] = self.compute_gradient(p_empty, output["x_empty"])
        output["time_dX"] = (time.time() - output["time_dX"]) * 1000

        # extract the mesh from the indicator function
        output["time_mesh"] = time.time()
        if self.check_logging("mesh", batch_idx):
            output["mesh"], output["grid"] = self.marching_cubes()
        output["time_mesh"] = (time.time() - output["time_mesh"]) * 1000

        return output

    def loss_step(self, batch: dict, output: dict, batch_idx: int, mode: str = "train"):
        # extracts all the information from batch and output
        x_surface = output["x_surface"]
        x_close = output["x_close"]
        x_empty = output["x_empty"]
        dX_surface = output["dX_surface"]
        dX_close = output["dX_close"]
        dX_empty = output["dX_empty"]
        v_surface = batch["vectors_surface"]
        v_close = batch["vectors_close"]
        v_empty = batch["vectors_empty"]

        # prepare the output of the loss
        loss = {}
        loss["surface"] = 0.0
        loss["empty_space"] = 0.0
        loss["empty_space_close"] = 0.0
        loss["empty_space_empty"] = 0.0
        loss["gradient"] = 0.0
        loss["gradient_surface"] = 0.0
        loss["gradient_close"] = 0.0
        loss["gradient_empty"] = 0.0
        loss["total"] = 0.0

        # surface constraint
        if self.hparams["lambda_surface"]:
            loss["surface"] = self.l2_loss(x_surface - 0.5)

        # empty space constraint
        if self.hparams["lambda_empty_space"]:
            loss["empty_space_close"] = self.l2_loss(x_close)
            loss["empty_space_empty"] = self.l2_loss(x_empty)
            empty_input = torch.cat([x_close * self.scheduler_step("close"), x_empty])
            loss["empty_space"] = self.l2_loss(empty_input)

        # gradient constraint
        if self.hparams["lambda_gradient"]:
            step = self.scheduler_step("gradient")
            loss["gradient_surface"] = self.l2_loss(dX_surface - v_surface)
            loss["gradient_close"] = self.l2_loss(dX_close - v_close)
            loss["gradient_empty"] = self.l2_loss(dX_empty - v_empty)
            gradient_input = [
                dX_surface - v_surface,
                dX_close - v_close,
                dX_empty - v_empty,
            ]
            loss["gradient"] = self.l2_loss(torch.cat(gradient_input)) * step

        # total loss computation
        loss["total"] = (
            self.hparams["lambda_surface"] * loss["surface"]
            + self.hparams["lambda_empty_space"] * loss["empty_space"]
            + self.hparams["lambda_gradient"] * loss["gradient"]
        )

        # combine all the metrics together to only send one request to WandB
        metrics = {}
        for key, value in loss.items():
            if key in ["surface", "total", "gradient", "empty_space"]:
                name = f"Loss Overview ({mode})"
                metrics[f"{name}/{key}"] = value
        for key, value in loss.items():
            if key.startswith("empty_space"):
                name = f"Empty Space Loss Overview ({mode})"
                metrics[f"{name}/{key}"] = value
        for key, value in loss.items():
            if key.startswith("gradient"):
                name = f"Gradient Loss Overview ({mode})"
                metrics[f"{name}/{key}"] = value

        # log the loss information only at the correct intervalls
        if self.check_logging("metrics", batch_idx):
            self.log(f"{mode}/loss", loss["total"], prog_bar=True, logger=False)
            self.log_dict(metrics, prog_bar=False)

        return loss

    def evaluation_step(self, batch: dict, output: dict, mode: str = "train"):
        # extracts all the information from batch and output
        mesh = output["mesh"]
        gt_mesh = batch["mesh"]

        # Chamfer Distance (CD)
        chamfer_samples = self.hparams["num_points_chamfer"]
        p1 = sample_points_from_meshes(mesh, chamfer_samples)
        p2 = sample_points_from_meshes(gt_mesh, chamfer_samples)
        loss, _ = chamfer_distance(p1, p2)
        self.log(f"Metrics ({mode})/chamfer", loss, prog_bar=False)

        # Normal Alignment (Normal)
        # TODO

        # F-Score
        # TODO

        # Total Reconstruction Time
        # TODO

    def training_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch, batch_idx, "train")
        loss = self.loss_step(batch, output, batch_idx, "train")
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "train")
        if self.check_logging("images", batch_idx):
            self.logging_images(batch, output, "train")
        if self.check_logging("mesh", batch_idx):
            self.evaluation_step(batch, output, "train")
            self.logging_mesh(batch, output, "train")
        return loss["total"]

    @torch.enable_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch, batch_idx)
        loss = self.loss_step(batch, output, batch_idx, "val")
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "val")
        return loss["total"]

    def on_before_optimizer_step(self, optimizer):
        log_steps = self.hparams["log_optimizer_every_n_steps"]
        batch_idx = self.trainer.global_step % log_steps
        if self.check_logging("optimizer", batch_idx):
            self.logging_optimizer()
