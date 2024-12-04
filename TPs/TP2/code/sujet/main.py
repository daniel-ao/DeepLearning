from pathlib import Path

import torch
import torch.optim as optim
import wandb
from torch.utils.data.dataset import random_split

from src.dataset import VAEDataset
from src.model import VAE
from src.trainer import Trainer


def config() -> dict():
    return {
        "batch_size": ...,
        "data_dir": Path("./images"),
        "device": "auto",
        "eval_freq": ...,
        "eval_iters": ...,
        "image_size": ...,
        "kl_weight": ...,
        "layers": [(..., ...), (..., ...), ...],
        "learning_rate": ...,
        "mode": "offline",  # either online or offline.
        "total_iters": ...,
    }


def main(conf: dict):
    if conf["device"] == "auto":
        conf["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    conf["device"] = torch.device(conf["device"])

    dataset = VAEDataset.from_dir(conf["data_dir"], conf["image_size"])
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    image_channels = dataset[0].shape[0]
    model = VAE(image_channels, conf["layers"])

    optimizer = optim.AdamW(model.parameters(), conf["learning_rate"])

    trainer = Trainer(
        conf["batch_size"],
        conf["device"],
        conf["eval_freq"],
        conf["eval_iters"],
        conf["kl_weight"],
        conf["total_iters"],
    )

    with wandb.init(
        project="l3-ia_vae",
        config=conf,
        mode=conf["mode"],
    ) as run:
        trainer.train(
            model,
            optimizer,
            train_dataset,
            val_dataset,
            run,
        )


if __name__ == "__main__":
    conf = config()
    main(conf)
