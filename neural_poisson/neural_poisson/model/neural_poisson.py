import time

import lightning as L
import open3d as o3d
import torch
import torch.nn as nn
import wandb
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
        self.save_hyperparameters(logger=False, ignore=["encoder"])

        # for default: [0,1]; for center: [-0.5, 0.5]
        self.X_offset = 0.0
        self.isolevel = 0.5
        if indicator_function == "center":
            self.X_offset = -0.5
            self.isolevel = 0.0

        # the encoder takes as input a point cloud of dim (B, P, 3) and produces the
        # logits of the indicator function which are then encoded with a tanh/sin
        # function to be in the range of (-0.5, 0.5)
        self.encoder = encoder

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
        return torch.autograd.grad(X.sum(), points, retain_graph=True)[0]

    def forward(self, points: torch.Tensor):
        """Evaluates the indicator function for the given points."""
        x = self.encoder(points)  # the logits of the encoder of dim (P, 1)
        x = x.squeeze(-1)  # (P,)
        return torch.sigmoid(x) + self.X_offset  # transforms into indicator function

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

    def logging_step(self, batch: dict, output: dict, mode: str = "train"):
        # log the total loss to the progress bar
        self.log("loss", output["total_loss"], prog_bar=True, logger=False)

        # log the stats to WandB
        unified_output = {}
        for key, value in output["loss"].items():
            unified_output[f"{mode}/loss/{key}"] = value
        for key, value in output["time"].items():
            unified_output[f"{mode}/time/{key}"] = value
        for key, value in output["stats"].items():
            unified_output[f"{mode}/stats/{key}"] = value
        self.log_dict(unified_output, prog_bar=False)

        # compute the gradient of the indicator function on the point map
        point_map = batch["point_map"].requires_grad_(True)
        x_point_map = self.forward(points=point_map)
        dX_point_map = self.compute_gradient(x_point_map, point_map)
        # log the images
        img_X = wandb.Image(x_point_map.detach().cpu().numpy())
        img_dX = wandb.Image(dX_point_map.detach().cpu().numpy())
        img_N = wandb.Image(batch["normal_map"].detach().cpu().numpy())
        self.logger.log_image(f"{mode}/images", [img_X, img_dX, img_N])  # type: ignore

    def training_step(self, batch: dict, batch_idx: int):
        """Perform training step."""
        output = self.model_step(batch)
        if batch_idx == 0:
            self.logging_step(batch, output, "train")
        return output["total_loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        """Perform validation step."""
        output = self.model_step(batch)
        self.logging_step(batch, output, "val")
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
