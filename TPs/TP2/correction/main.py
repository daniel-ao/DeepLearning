import hydra
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataset import random_split

from configs.template import MainConfig
from src.dataset import VAEDataset
from src.model import VAE
from src.trainer import Trainer


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    config = MainConfig.from_dict(dict_config)

    dataset = VAEDataset.from_dir(config.dataset.data_dir, config.dataset.image_size)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    image_channels = dataset[0].shape[0]
    model = VAE(image_channels, config.model.layers)

    optimizer = optim.AdamW(model.parameters(), config.trainer.learning_rate)

    trainer = Trainer(
        config.trainer.batch_size,
        config.trainer.device,
        config.trainer.draw_freq,
        config.trainer.eval_freq,
        config.trainer.eval_iters,
        config.trainer.kl_weight,
        config.trainer.total_iters,
    )

    with wandb.init(
        project="l3-ia_vae",
        config=OmegaConf.to_container(dict_config),
        entity=config.wandb.entity,
        mode=config.wandb.mode,
    ) as run:
        trainer.train(
            model,
            optimizer,
            train_dataset,
            val_dataset,
            run,
        )


if __name__ == "__main__":
    main()
