import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class DummyModel(L.LightningModule):
    def __init__(self, input_size=10, hidden_size=20, output_size=1, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "train/loss",
            loss,
            # on_epoch=True,
            # on_step=True,
            # logger=True,
            # prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
