import pytorch_lightning as pl
from routing.utils import auto_args


@auto_args
class Model(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
