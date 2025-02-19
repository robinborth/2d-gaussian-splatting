import time

import lightning as L
import open3d as o3d
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.utilities import grad_norm
from pytorch3d.ops import marching_cubes


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

    def l2_loss(self, x: torch.Tensor):
        """Simple L2-Loss."""
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

        # transform into the required range : [0, 1] <-> [-0.5, 0.5]
        x += self.X_offset
        return x

    def model_step(self, batch: dict):
        # extract the batch information
        p_surface = batch["points_surface"].requires_grad_(True)
        p_close = batch["points_close"].requires_grad_(True)
        p_empty = batch["points_empty"]
        v_surface = batch["vectors_surface"]
        v_close = batch["vectors_close"]

        # evaluate the indicator function
        time_X = time.time()
        x_surface = self.forward(points=p_surface)
        x_close = self.forward(points=p_close)
        x_empty = self.forward(points=p_empty)
        time_X = time.time() - time_X

        time_dX = time.time()
        dX_surface = self.compute_gradient(x_surface, p_surface)
        dX_close = self.compute_gradient(x_close, p_close)
        time_dX = time.time() - time_dX

        # surface constraint
        L_surface = self.l2_loss(x_surface - self.X_offset - 0.5)

        # empty space constraint
        L_empty_space_close = self.l2_loss(x_close - self.X_offset)
        L_empty_space_empty = self.l2_loss(x_empty - self.X_offset)
        empty_input = torch.cat([x_close - self.X_offset, x_empty - self.X_offset])
        L_empty_space = self.l2_loss(empty_input)

        # gradient constraint
        L_gradient_surface = self.l2_loss(dX_surface - v_surface)
        L_gradient_close = self.l2_loss(dX_close - v_close)
        gradient_input = torch.cat([dX_surface - v_surface, dX_close - v_close])
        L_gradient = self.l2_loss(gradient_input)

        # total loss computation
        loss = (
            self.hparams["lambda_surface"] * L_surface
            + self.hparams["lambda_gradient"] * L_gradient
            + self.hparams["lambda_empty_space"] * L_empty_space
        )

        # pre-compute usefull stats for logging
        dX_surface_norm = torch.linalg.vector_norm(dX_surface, dim=-1)
        dX_close_norm = torch.linalg.vector_norm(dX_close, dim=-1)
        v_surface_norm = torch.linalg.vector_norm(v_surface, dim=-1)
        v_close_norm = torch.linalg.vector_norm(v_close, dim=-1)

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
            "stats": {
                "dX_surface_mean": dX_surface_norm.mean(),
                "dX_surface_min": dX_surface_norm.min(),
                "dX_surface_max": dX_surface_norm.max(),
                "dX_close_mean": dX_close_norm.mean(),
                "dX_close_min": dX_close_norm.min(),
                "dX_close_max": dX_close_norm.max(),
                "v_surface_mean": v_surface_norm.mean(),
                "v_surface_min": v_surface_norm.min(),
                "v_surface_max": v_surface_norm.max(),
                "v_close_mean": v_close_norm.mean(),
                "v_close_min": v_close_norm.min(),
                "v_close_max": v_close_norm.max(),
            },
        }

        return output

    def logging_metrics(self, batch: dict, output: dict, mode: str = "train"):
        # log the total loss to the progress bar
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
        # only log if we want to monitor the camera
        if batch["camera_idx"] not in self.hparams["log_camera_idxs"]:
            return

        # compute the gradient of the indicator function on the point map
        point_map = batch["point_map"].requires_grad_(True)
        x_point_map = self.forward(points=point_map)
        dX_point_map = self.compute_gradient(x_point_map, point_map)
        # log the images
        name = f"Image-{batch['camera_idx']:03} ({mode})"
        img_X = wandb.Image(x_point_map.detach().cpu().numpy() - self.X_offset)
        img_dX = wandb.Image(dX_point_map.detach().cpu().numpy())
        img_N = wandb.Image(batch["normal_map"].detach().cpu().numpy())
        self.logger.log_image(f"{name}/indicator", [img_X])  # type: ignore
        self.logger.log_image(f"{name}/gradient", [img_dX])  # type: ignore
        self.logger.log_image(f"{name}/normal", [img_N])  # type: ignore

    def logging_optimizer(self, mode: str = "train"):
        # extreact the information from the training
        optimizer = self.optimizers()._optimizer  # type: ignore

        # skipts the logging for the first optimizer call
        if not optimizer.state_dict()["state"]:
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

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms)

        # log the wandb gradients as histograms
        histograms = {}
        for name, p in self.encoder.named_parameters():
            if p.grad is None:
                continue
            h = wandb.Histogram(p.grad.data.detach().cpu())
            histograms[f"grad_histogram/{name}"] = h
        self.logger.experiment.log(histograms)

        # log the weights distribution of the layers
        histograms = {}
        for name, p in self.encoder.named_parameters():
            h = wandb.Histogram(p.data.detach().cpu())
            histograms[f"weights_histogram/{name}"] = h
        self.logger.experiment.log(histograms)

    def training_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        self.logging_images(batch, output, "train")
        self.logging_optimizer("train")
        self.logging_metrics(batch, output, "train")
        return output["total_loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        """Perform validation step."""
        output = self.model_step(batch)
        self.logging_images(batch, output, "val")
        self.logging_metrics(batch, output, "val")
        return output["total_loss"]

    def to_mesh(self) -> o3d.geometry.TriangleMesh:
        # prepare the evaluation
        self.eval()
        N = self.hparams["voxel_size"]
        min_val, max_val = self.hparams["domain"]

        # fetch the point on the grid lattice
        grid_vals = torch.linspace(min_val, max_val, N)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        grid = torch.stack((xs.ravel(), ys.ravel(), zs.ravel())).reshape(-1, 3)

        # evaluate the indicator function on the grid structure
        X = []
        for points in torch.split(grid, self.hparams["chunk_size"]):
            points = points.to(self.device)
            x = self.forward(points)
            X.append(x)
        X_grid = torch.cat(X).reshape(N, N, N).detach().cpu().numpy()

        # perform marching cubes (slow)
        verts, faces, _, _ = marching_cubes(X_grid, level=self.isolevel)
        # verts = verts * ((max_val - min_val) / resolution) + min_val

        # merge into open3d mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        return mesh
