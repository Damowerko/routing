import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os

from projectname.datasets import Dataset
from projectname.models import Model


def train(params):
    dataset = Dataset(**vars(params))
    model = Model(**vars(params))

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
        precision=32,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
    )

    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer.fit(model, datamodule=dataset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--log", type=int, default=1)

    # data arguments
    group = parser.add_argument_group("Data")
    Dataset.add_args(group)

    # model arguments
    group = parser.add_argument_group("Model")
    Model.add_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()
    train(params)
