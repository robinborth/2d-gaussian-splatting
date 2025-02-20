import time

import lightning as L
import torch
import torch.nn as nn
import wandb
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.structures import Meshes

from neural_poisson.data.prepare import extract_surface_data


class NeuralPoisson(L.LightningModule):
    def __init__(
        self,
        # encoder module either MLP, DenseGrid, etc.
        encoder: nn.Module,
        # loss settings
        lambda_gradient: float = 1.0,
        lambda_surface: float = 1.0,
        lambda_empty_space: float = 1.0,
        # indicator settings
        indicator_function: str = "default",  # "default", "center"
        activation: str = "sin",  # sin, sigmoid
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

        # check for valid values
        assert activation in ["sin", "sigmoid"]
        assert indicator_function in ["default", "center"]

        # for default: [0,1]; for center: [-0.5, 0.5]
        self.X_offset = 0.0
        self.isolevel = 0.5
        if indicator_function == "center":
            self.X_offset = -0.5
            self.isolevel = 0.0

        # the encoder takes as input a point cloud of dim (B, P, 3) and produces the
        # logits of the indicator function which are then encoded with a tanh/sin
        # function to be in the range of (-0.5, 0.5)
        self.encoder = encoder()

    def configure_optimizers(self):
        """Default lightning optimizer setup."""
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "monitor": self.hparams["monitor"]}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def compute_basic_stats(self, points: torch.Tensor, name: str):
        stats = {}
        points_norm = torch.linalg.vector_norm(points, dim=-1)
        if points_norm.numel():
            stats[f"{name}_mean"] = points_norm.mean()
            stats[f"{name}_min"] = points_norm.min()
            stats[f"{name}_max"] = points_norm.max()
        return stats

    def l2_loss(self, x: torch.Tensor):
        """Simple L2-Loss."""
        if x.numel() == 0:
            return 0.0
        return (x**2).mean()

    def compute_gradient(self, X: torch.Tensor, points: torch.Tensor):
        # we want to compute dX/dp, which is the gradient of the estimated indicator function
        # X w.r.t the input points. However we can only compute dL/dp which computes the
        # gradients from a loss scalar. The chain rule is dL/dp = dL/dX * dX/dp. In order to
        # compute dX/dp we need to define the loss function do get dL/dX = 1, which results in
        # a simple summation of dX, e.g. L=X.sum(), where the derivatives are 1.
        return torch.autograd.grad(
            outputs=X.sum(),
            inputs=points,
            retain_graph=True,
            create_graph=True,
        )[0]

    def forward(self, points: torch.Tensor):
        """Evaluates the indicator function for the given points."""
        x = self.encoder(points)  # the logits of the encoder of dim (P, 1)
        x = x.squeeze(-1)  # (P,)

        # transforms into indicator function
        if self.hparams["activation"] == "sin":
            x = (torch.sin(x) + 1) / 2  # [0, 1]
        elif self.hparams["activation"] == "sigmoid":
            x = torch.sigmoid(x)  # [0, 1]

        # # transform into the required range : [0, 1] <-> [-0.5, 0.5]
        return x + self.X_offset

    def model_step(self, batch: dict):
        # extract the batch information
        p_surface = batch["points_surface"].requires_grad_(True)
        p_close = batch["points_close"].requires_grad_(True)
        p_empty = batch["points_empty"]
        v_surface = batch["vectors_surface"]
        v_close = batch["vectors_close"]

        # evaluate the indicator function
        time_X = time.time()
        x_surface = torch.tensor([])
        x_close = torch.tensor([])
        x_empty = torch.tensor([])
        if self.hparams["lambda_surface"] or self.hparams["lambda_gradient"]:
            x_surface = self.forward(points=p_surface)
        if self.hparams["lambda_empty_space"] or self.hparams["lambda_gradient"]:
            x_close = self.forward(points=p_close)
            x_empty = self.forward(points=p_empty)
        time_X = time.time() - time_X

        time_dX = time.time()
        dX_surface = torch.tensor([])
        dX_close = torch.tensor([])
        if self.hparams["lambda_gradient"]:
            dX_surface = self.compute_gradient(x_surface, p_surface)
            dX_close = self.compute_gradient(x_close, p_close)
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
            L_empty_space_close = self.l2_loss(x_close - self.X_offset)
            L_empty_space_empty = self.l2_loss(x_empty - self.X_offset)
            empty_input = torch.cat([x_close - self.X_offset, x_empty - self.X_offset])
            L_empty_space = self.l2_loss(empty_input)

        # gradient constraint
        L_gradient = 0.0
        L_gradient_surface = 0.0
        L_gradient_close = 0.0
        if self.hparams["lambda_gradient"]:
            L_gradient_surface = self.l2_loss(dX_surface - v_surface)
            L_gradient_close = self.l2_loss(dX_close - v_close)
            gradient_input = torch.cat([dX_surface - v_surface, dX_close - v_close])
            L_gradient = self.l2_loss(gradient_input)

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
        stats.update(self.compute_basic_stats(v_surface, "v_surface"))
        stats.update(self.compute_basic_stats(v_close, "v_close"))

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
                "total": loss,
            },
            "time": {
                "indicator": time_X * 1000,  # in ms
                "gradient": time_dX * 1000,  # in ms
            },
            "stats": stats,
        }

        return output

    def check_logging(self, mode: str = "metrics", batch_idx: int = 0):
        if mode == "mesh":
            log_epochs = self.hparams[f"log_{mode}_every_n_epochs"]
            return (
                self.trainer.current_epoch % log_epochs == 0
                and batch_idx == 0
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
        x_point_map = self.forward(points=point_map)

        # log the images
        name = f"Image-{batch['camera_idx']:03} ({mode})"
        img_X = wandb.Image(x_point_map.detach().cpu().numpy() - self.X_offset)
        img_X_gt = wandb.Image(batch["indicator_map"].detach().cpu().numpy())
        self.logger.log_image(f"{name}/indicator", [img_X])  # type: ignore
        self.logger.log_image(f"{name}/indicator_gt", [img_X_gt])  # type: ignore

        # compute the normal and vector maps
        dX_point_map = self.compute_gradient(x_point_map, point_map)
        img_dX = wandb.Image(dX_point_map.detach().cpu().numpy())
        img_dX_gt = wandb.Image(batch["vector_map"].detach().cpu().numpy())
        img_N_gt = wandb.Image(batch["normal_map"].detach().cpu().numpy())
        self.logger.log_image(f"{name}/vector", [img_dX])  # type: ignore
        self.logger.log_image(f"{name}/vector_gt", [img_dX_gt])  # type: ignore
        self.logger.log_image(f"{name}/normal_gt", [img_N_gt])  # type: ignore

    def logging_optimizer(self, mode: str = "train"):
        if self.global_step == 0:
            return

        # extreact the information from the training
        optimizer = self.optimizers()._optimizer  # type: ignore

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
        self.mesh = self.to_mesh()
        if self.mesh is None:
            return

        # compute chamfer distance
        chamfer_samples = self.hparams["num_points_chamfer"]
        p1 = sample_points_from_meshes(self.mesh, chamfer_samples)
        p2 = sample_points_from_meshes(batch["mesh"], chamfer_samples)
        loss, _ = chamfer_distance(p1, p2)
        self.log(f"Metrics ({mode})/chamfer", loss, prog_bar=False)

        # log the mesh for the entire camera logs
        dataset = self.trainer.datamodule.dataset  # type: ignore
        for camera_idx in dataset.log_camera_idxs:
            data = extract_surface_data(
                camera=dataset.cameras[camera_idx],
                mesh=self.mesh,
                image_size=dataset.image_size,
                fill_depth=dataset.fill_depth,
            )
            # log mesh images
            name = f"Mesh-{batch['camera_idx']:03} ({mode})"
            img_N = wandb.Image(data["normal_map"].detach().cpu().numpy())
            img_N_gt = wandb.Image(batch["normal_map"].detach().cpu().numpy())
            img_X = wandb.Image(data["indicator_map"].detach().cpu().numpy())
            img_X_gt = wandb.Image(batch["indicator_map"].detach().cpu().numpy())
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
        for name, p in self.encoder.named_parameters():
            if p.grad is None:
                continue
            h = wandb.Histogram(p.grad.data.detach().cpu())
            histograms[f"Gradients Histogram/{name}"] = h
        self.logger.experiment.log(histograms)

        # log the weights distribution of the layers
        histograms = {}
        for name, p in self.encoder.named_parameters():
            h = wandb.Histogram(p.data.detach().cpu())
            histograms[f"Weights Histogram/{name}"] = h
        self.logger.experiment.log(histograms)

    def training_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        if self.check_logging("metrics", batch_idx):
            self.logging_metrics(batch, output, "train")
        if self.check_logging("images", batch_idx):
            self.logging_images(batch, output, "train")
        if self.check_logging("optimizer", batch_idx):
            self.logging_optimizer("train")
        if self.check_logging("mesh", batch_idx):
            self.logging_mesh(batch, "train")
        return output["total_loss"]

    def to_mesh(self, voxel_size: int | None = None):
        # prepare the evaluation
        self.eval()
        N = self.hparams["voxel_size"] if voxel_size is None else voxel_size
        min_val, max_val = self.hparams["domain"]

        # fetch the point on the grid lattice
        grid_vals = torch.linspace(min_val, max_val, N)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        grid = torch.stack((xs.ravel(), ys.ravel(), zs.ravel())).reshape(-1, 3)

        # evaluate the indicator function on the grid structure
        sdfs = []
        for points in torch.split(grid, self.hparams["chunk_size"]):
            x = self.forward(points.to(self.device))
            # convert indicator to "sdf" value, where negative is inside
            sdfs.append(-x.detach().cpu())
        sdf_grid = torch.cat(sdfs).reshape(N, N, N)

        # ensures that we have a valid isolevel and can extract a mesh
        isolevel = self.isolevel
        if isolevel > sdf_grid.max() or isolevel < sdf_grid.min():
            isolevel = (sdf_grid.max().item() - sdf_grid.min().item()) / 2

        # perform marching cubes
        verts, faces = marching_cubes(sdf_grid[None], isolevel=isolevel)
        if not len(verts[0]):
            return None
        return Meshes(verts=verts, faces=faces).to(self.device)
