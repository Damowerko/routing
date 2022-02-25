from idna import check_nfc
import pytorch_lightning as pl
from routing.utils import auto_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data


@auto_args
class DeepGNN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0,
        n_layers: int = 3,
        K: int = 4,
        F: int = 32,
        nonlinearity: str = "relu",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.nonlinearity = {
            "relu": nn.ReLU(True),
            "leaky_relu": nn.LeakyReLU(True),
        }[nonlinearity]

        channels = [1] + [F] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList(
            [
                gnn.TAGConv(
                    channels[i],
                    channels[i + 1],
                    K,
                    normalize=False,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, data: Data):
        x = data.x.to(self.dtype).view(-1, 1)
        edge_attr = data.edge_attr.to(self.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x.view(-1, data.num_features)

    def training_step(self, data):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, data, *args, **kwargs):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, data, *args, **kwargs):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
