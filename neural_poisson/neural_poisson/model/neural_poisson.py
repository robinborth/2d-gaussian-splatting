import time
from pathlib import Path

import lightning as L
import torch
import wandb
from pytorch3d.io import save_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.structures import Meshes

from neural_poisson.data.grid import coord_grid, coord_grid_along_axis
from neural_poisson.data.prepare import extract_surface_data
from neural_poisson.model.implicit import IndicatorFunction


class NeuralPoisson(L.LightningModule):
    def __init__(
        self,
        # encoder module either MLP, DenseGrid, etc.
        indicator_function: IndicatorFunction,
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
        # training settings
        optimizer=None,
        scheduler=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.indicator_function = indicator_function()
        self.X_offset = self.indicator_function.X_offset
        self.isolevel = self.indicator_function.isolevel

    ################################################################################
    # Optimizer Utils
    ################################################################################

    def configure_optimizers(self):
        """Default lightning optimizer setup."""
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "monitor": self.hparams["monitor"]}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def scheduler_step(self, key: str):
        mode = self.hparams[f"{key}_mode"]
        if mode == "zero":
            return 0.0
        if mode == "one":
            return 1.0

        steps = self.hparams[f"{key}_steps"]
        t = max(min(self.trainer.global_step / steps, 1.0), 0.0)  # [0.0, 1.0]
        if mode == "decrease":
            return 1 - t
        if mode == "increase":
            return t
        raise AttributeError(f"There is a wrong {mode=}!")

    ################################################################################
    # Loss Computation
    ################################################################################

    def l2_loss(self, x: torch.Tensor):
        """Simple L2-Loss."""
        if x.numel() == 0:
            return 0.0
        return (x**2).mean()

    ################################################################################
    # Logging
    ################################################################################

    def compute_axis(self, axis: str = "x", voxel_size: int = 256):
        N = self.hparams["voxel_size"] if voxel_size is None else voxel_size
        grid = coord_grid_along_axis(
            axis=axis,
            voxel_size=N,
            domain=self.hparams["domain"],
            default_coord=0.0,
            device=self.device,
        )
        # evaluate the indicator function
        x, _ = self.forward(grid)
        return x

    def compute_basic_stats(self, points: torch.Tensor, name: str):
        stats = {}
        points_norm = torch.linalg.vector_norm(points, dim=-1)
        if points_norm.numel():
            stats[f"{name}_mean"] = points_norm.mean()
            stats[f"{name}_min"] = points_norm.min()
            stats[f"{name}_max"] = points_norm.max()
        return stats

    def check_logging(self, mode: str = "metrics", batch_idx: int = 0):
        if mode == "mesh":
            log_epochs = self.hparams[f"log_{mode}_every_n_epochs"]
            return (
                self.trainer.current_epoch % log_epochs == 0
                and batch_idx == (self.trainer.num_training_batches - 1)
                and self.hparams[f"log_{mode}"]
            )
        return (
            batch_idx % self.hparams[f"log_{mode}_every_n_steps"] == 0
            and self.hparams[f"log_{mode}"]
        )

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

    def logging_images(self, batch: dict, mode: str = "train"):
        # compute the gradient of the indicator function on the point map
        point_map = batch["point_map"].requires_grad_(True)
        x_point_map, _ = self.forward(points=point_map)

        # log the axis
        name = f"Image-Axis ({mode})"
        imgX = self.compute_axis("x").detach().cpu().numpy() - self.X_offset
        imgY = self.compute_axis("y").detach().cpu().numpy() - self.X_offset
        imgZ = self.compute_axis("z").detach().cpu().numpy() - self.X_offset
        self.logger.log_image(f"{name}/x", [imgX])  # type: ignore
        self.logger.log_image(f"{name}/y", [imgY])  # type: ignore
        self.logger.log_image(f"{name}/z", [imgZ])  # type: ignore

        # log the images
        name = f"Image-{batch['camera_idx']:03} ({mode})"
        img_X = wandb.Image(x_point_map.detach().cpu().numpy() - self.X_offset)
        img_X_gt = wandb.Image(batch["indicator_map"].detach().cpu().numpy())
        self.logger.log_image(f"{name}/indicator", [img_X])  # type: ignore
        self.logger.log_image(f"{name}/indicator_gt", [img_X_gt])  # type: ignore

        # compute the normal and vector maps
        dX_point_map = self.compute_gradient(point_map, x_point_map)
        img_dX = wandb.Image(dX_point_map.detach().cpu().numpy())
        img_dX_gt = wandb.Image(batch["vector_map"].detach().cpu().numpy())
        img_N_gt = wandb.Image(batch["normal_map"].detach().cpu().numpy())
        self.logger.log_image(f"{name}/vector", [img_dX])  # type: ignore
        self.logger.log_image(f"{name}/vector_gt", [img_dX_gt])  # type: ignore
        self.logger.log_image(f"{name}/normal_gt", [img_N_gt])  # type: ignore

    def logging_optimizer(self, mode: str = "train"):
        # extreact the information from the training
        optimizer = self.optimizers()._optimizer  # type: ignore
        if self.global_step == 0 or not optimizer.state_dict()["state"]:
            return

        # extract the state dict and param groups
        layer = 0
        state = optimizer.state_dict()["state"][layer]
        params = optimizer.param_groups[0]

        m_hat_t = state["exp_avg"] / (1 - params["betas"][0] ** state["step"])
        v_hat_t = state["exp_avg_sq"] / (1 - params["betas"][1] ** state["step"])
        lr_modifier = m_hat_t / (torch.sqrt(v_hat_t) + params["eps"])
        histogram = wandb.Histogram(lr_modifier.detach().cpu())
        self.logger.experiment.log({"Learning Rate Modifier": histogram})  # type: ignore

    def logging_mesh(self, batch: dict, mode: str = "train"):
        # compute the mesh (slow)
        mesh = self.to_mesh()
        if mesh is None:
            return

        # compute chamfer distance
        chamfer_samples = self.hparams["num_points_chamfer"]
        p1 = sample_points_from_meshes(mesh, chamfer_samples)
        p2 = sample_points_from_meshes(batch["mesh"], chamfer_samples)
        loss, _ = chamfer_distance(p1, p2)
        self.log(f"Metrics ({mode})/chamfer", loss, prog_bar=False)

        # save the mesh
        file_name = f"epoch_{self.trainer.current_epoch:05}.obj"
        path = Path(self.trainer.default_root_dir) / f"mesh/{file_name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_obj(path, mesh.verts_packed(), mesh.faces_packed())

        # log the mesh for the entire camera logs
        dataset = self.trainer.datamodule.dataset(mode=mode)  # type: ignore
        for camera_idx in dataset.log_camera_idxs:
            normal_map = dataset.normal_maps[camera_idx].detach().cpu().numpy()
            indicator_map = dataset.indicator_maps[camera_idx].detach().cpu().numpy()
            data = extract_surface_data(
                camera=dataset.cameras[camera_idx],
                mesh=mesh,
                image_size=dataset.image_size,
                fill_depth=dataset.fill_depth,
            )
            # log mesh images
            name = f"Mesh-{camera_idx:03} ({mode})"
            img_N = wandb.Image(data["normal_map"].detach().cpu().numpy())
            img_X = wandb.Image(data["indicator_map"].detach().cpu().numpy())
            img_N_gt = wandb.Image(normal_map)
            img_X_gt = wandb.Image(indicator_map)
            self.logger.log_image(f"{name}/normal", [img_N])  # type: ignore
            self.logger.log_image(f"{name}/normal_gt", [img_N_gt])  # type: ignore
            self.logger.log_image(f"{name}/indicator", [img_X])  # type: ignore
            self.logger.log_image(f"{name}/indicator_gt", [img_X_gt])  # type: ignore

    def on_before_optimizer_step(self, optimizer):
        log_steps = self.hparams["log_optimizer_every_n_steps"]
        batch_idx = self.trainer.global_step % log_steps
        if not self.check_logging("optimizer", batch_idx):
            return

        # log the wandb gradients as histograms
        histograms = {}
        for name, p in self.indicator_function.mlp.named_parameters():
            if p.grad is None:
                continue
            h = wandb.Histogram(p.grad.data.detach().cpu())
            histograms[f"Gradients Histogram/{name}"] = h
        self.logger.experiment.log(histograms)

        # log the weights distribution of the layers
        histograms = {}
        for name, p in self.indicator_function.mlp.named_parameters():
            h = wandb.Histogram(p.data.detach().cpu())
            histograms[f"Weights Histogram/{name}"] = h
        self.logger.experiment.log(histograms)

    def log_video(self, sdf_grid: torch.Tensor, dim: str = "x"):
        D = sdf_grid.shape[0]
        volume = sdf_grid - self.X_offset
        if dim == "x":
            volume = sdf_grid[None].permute(1, 0, 2, 3).expand(D, 3, D, D)
        if dim == "y":
            volume = sdf_grid[None].permute(2, 0, 1, 3).expand(D, 3, D, D)
        if dim == "z":
            volume = sdf_grid[None].permute(3, 0, 1, 2).expand(D, 3, D, D)
        video = wandb.Video((volume * 255).to(torch.uint8), fps=60)  # type: ignore
        self.logger.experiment.log({f"Mesh Slicing Video/{dim}": video})  # type: ignore

    ################################################################################
    # Mesh Extraction
    ################################################################################

    def to_mesh(self, voxel_size: int | None = None):
        # prepare the evaluation
        self.eval()

        # fetch the point on the grid lattice
        N = self.hparams["voxel_size"] if voxel_size is None else voxel_size
        grid = coord_grid(voxel_size=N, domain=self.hparams["domain"]).reshape(-1, 3)

        # evaluate the indicator function on the grid structure
        sdfs = []
        for points in torch.split(grid, self.hparams["chunk_size"]):
            x, _ = self.forward(points.to(self.device))
            # convert indicator to "sdf" value, where negative is inside
            sdfs.append(-x.detach().cpu())
        sdf_grid = torch.cat(sdfs).reshape(N, N, N)

        # log the slice of the mesh
        # for dim in ["x", "y", "z"]:
        #     self.log_video(sdf_grid=sdf_grid, dim=dim)

        # ensures that we have a valid isolevel and can extract a mesh
        isolevel = self.isolevel
        if isolevel > sdf_grid.max() or isolevel < sdf_grid.min():
            isolevel = (sdf_grid.max().item() - sdf_grid.min().item()) / 2

        # perform marching cubes
        sdf_grid = sdf_grid.permute(2, 1, 0)[None]  # (W,H,D) -> (1,D,H,W)
        verts, faces = marching_cubes(sdf_grid, isolevel=isolevel)

        # wrap into a pytorch3d mesh
        if not len(verts[0]):
            return None
        return Meshes(verts=verts, faces=faces).to(self.device)

    ################################################################################
    # Training Methods
    ################################################################################

    def compute_gradient(self, points: torch.Tensor, X: torch.Tensor):
        return self.indicator_function.compute_gradient(
            points=points,
            field_values=X,
            mode=self.hparams["gradient_compute_mode"],
            eps=self.hparams["gradient_eps"],
        )

    def forward(self, points: torch.Tensor):
        """Evaluates the indicator function for the given points."""
        X, logits = self.indicator_function(points)
        return X, logits

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
            L_surface = self.l2_loss(x_surface - self.X_offset - 0.5)

        # empty space constraint
        L_empty_space = 0.0
        L_empty_space_close = 0.0
        L_empty_space_empty = 0.0
        if self.hparams["lambda_empty_space"]:
            i_close = x_close - self.X_offset
            i_empty = x_empty - self.X_offset
            L_empty_space_close = self.l2_loss(i_close)
            L_empty_space_empty = self.l2_loss(i_empty)
            empty_input = torch.cat([i_close * self.scheduler_step("close"), i_empty])
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
            self.logging_images(batch, "train")
        if self.check_logging("optimizer", batch_idx):
            self.logging_optimizer("train")
        if self.check_logging("mesh", batch_idx):
            self.logging_mesh(batch, "train")
        return output["total_loss"]

    @torch.enable_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "val")
        return output["total_loss"]
