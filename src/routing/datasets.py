import pytorch_lightning as pl
from routing.utils import auto_args


@auto_args
class Dataset(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

    def setup(self, stage=None):
        if stage == "fit":
            self.data, self.label = self.load_data()

    def train_dataloader(self):
        return self.data, self.label

    def val_dataloader(self):
        return self.data, self.label

    def test_dataloader(self):
        return self.data, self.label

    def load_data(self):
        data = None
        label = None
        return data, label
