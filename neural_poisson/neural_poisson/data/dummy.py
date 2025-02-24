import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class DummyDatamodule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 100,
        num_samples: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader:
        x = torch.randn(self.hparams["num_samples"], 10)
        y = torch.randn(self.hparams["num_samples"], 1)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.hparams["batch_size"], shuffle=True)
