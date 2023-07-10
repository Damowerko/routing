import pytorch_lightning as pl
from routing.utils import auto_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data


@auto_args
class MLPGNN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_mlp: int = 2,
        n_gnn: int = 3,
        K: int = 4,
        F: int = 256,
        nonlinearity: str = "leaky_relu",
        batch_norm: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_mlp = n_mlp
        self.n_gnn = n_gnn
        self.K = K
        self.F = F
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.normalize = normalize

        super().__init__()
        self.save_hyperparameters()

        self.nonlinearity = {
            "relu": nn.ReLU(True),
            "leaky_relu": nn.LeakyReLU(True),
        }[nonlinearity]

        encoder_layers = []
        for i in range(n_mlp):
            encoder_layers.append(nn.Linear(1, F) if i == 0 else nn.Linear(F, F))
            encoder_layers.append(self.nonlinearity)

        decoder_layers = []
        for i in range(n_mlp):
            decoder_layers.append(
                nn.Linear(F, 1) if i == n_mlp - 1 else nn.Linear(F, F)
            )
            decoder_layers.append(self.nonlinearity)

        gnn_layers = []
        for i in range(n_gnn):
            gnn_layers += [
                (
                    gnn.BatchNorm(F) if self.batch_norm else nn.Identity(),
                    "x -> x",
                ),
                (
                    gnn.TAGConv(
                        F,
                        F,
                        K,
                        normalize=self.normalize,
                    ),
                    "x, edge_index, edge_attr -> x",
                ),
                self.nonlinearity if i < n_gnn - 1 else nn.Identity(),
            ]

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.gnn = gnn.Sequential("x, edge_index, edge_attr", gnn_layers)

    def forward(self, data: Data):
        x = data.x.view(-1, 1)
        x = self.encoder(x)
        x = self.gnn(x, data.edge_index, data.edge_attr)
        x = self.decoder(x)
        return x.view(-1, data.num_features)

    def training_step(self, data):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("train/loss", loss)
        return loss

    @staticmethod
    def relative_error(x, target, eps=1e-4):
        error = (x - target).abs()
        return torch.where(target.abs() < eps, error, error / target.abs())

    def validation_step(self, data, *args, **kwargs):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("val/loss", loss, prog_bar=True)
        rel_error = self.relative_error(yhat, data.y).mean()
        self.log("val/rel_error", rel_error, prog_bar=True)

    def test_step(self, data, *args, **kwargs):
        yhat = self(data)
        loss = F.mse_loss(yhat, data.y)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
