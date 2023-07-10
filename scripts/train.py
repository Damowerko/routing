import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import os
from torch_geometric.data import LightningDataset
from torch.utils.data import random_split

from routing.datasets import DijkstraDataset
from routing.models import MLPGNN


def train(params):
    dataset = DijkstraDataset("data/", **vars(params))
    datamodule = LightningDataset(
        *random_split(dataset, [int(len(dataset) * frac) for frac in [0.8, 0.2]]),  # type: ignore
        batch_size=params.batch_size,
        num_workers=params.num_workers if params.num_workers >= 0 else os.cpu_count(),  # type: ignore
        pin_memory=params.gpus >= 0,
        shuffle=True,
    )

    model = MLPGNN(**vars(params))

    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if params.log
        else None
    )
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="./checkpoints",
            filename="epoch={epoch}-loss={train/loss:0.4f}",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=params.patience,
        ),
    ]

    trainer = pl.Trainer(
        logger=logger,  # type: ignore
        callbacks=callbacks,
        precision=32,
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="-1 uses all cpus. 0 uses the main process.",
    )

    # data arguments
    group = parser.add_argument_group("Data")
    DijkstraDataset.add_args(group)  # type: ignore

    # model arguments
    group = parser.add_argument_group("Model")
    MLPGNN.add_args(group)  # type: ignore

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=100)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()
    train(params)
