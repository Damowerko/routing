import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from torch_geometric.data import LightningDataset
from torch.utils.data import random_split

from routing.datasets import DijkstraDataset
from routing.models import DeepGNN


def train(params):
    dataset = DijkstraDataset("data/", **vars(params))
    datamodule = LightningDataset(
        *random_split(dataset, [int(len(dataset) * frac) for frac in [0.8, 0.2]]),
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True
    )

    model = DeepGNN(**vars(params))

    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if params.log
        else None
    )
    callbacks = [
        ModelCheckpoint(
            monitor="train/loss",
            dirpath="./checkpoints",
            filename="epoch={epoch}-loss={train/loss:0.4f}",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=64,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
    )

    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--log", type=int, default=1)

    # data arguments
    group = parser.add_argument_group("Data")
    DijkstraDataset.add_args(group)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--num_workers", type=int, default=0)

    # model arguments
    group = parser.add_argument_group("Model")
    DeepGNN.add_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()
    train(params)
