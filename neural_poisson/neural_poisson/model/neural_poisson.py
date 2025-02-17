import lightning as L
import torch
import wandb

from neural_poisson.model.layers import MLP


class NeuralPoisson(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.network = MLP(
            in_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=1,
        )

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "monitor": self.hparams["monitor"]}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def forward(self, points: torch.Tensor):
        x = self.network(points)
        return torch.tanh(x) / 2  # (-0.5, 0.5)

    def training_step(self, batch, batch_idx):
        # evaluate the data
        x = self.forward(points=batch["points_surface"])

        # compute the loss
        loss = ((x - 0.5) ** 2).mean()  # l2-loss
        self.log("train/loss", loss, prog_bar=True)

        # log the images
        normal_map = batch["normal_map"].detach().cpu().numpy()
        img = wandb.Image(normal_map)
        self.logger.log_image("train/normal_map", [img])

        return loss
