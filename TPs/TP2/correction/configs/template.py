from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.utils import to_absolute_path


@dataclass
class DatasetConfig:
    data_dir: Path
    image_size: int

    def __post_init__(self):
        self.data_dir = to_absolute_path(self.data_dir)
        self.data_dir = Path(self.data_dir)


@dataclass
class ModelConfig:
    layers: list[tuple[int, int]]

    def __post_init__(self):
        self.layers = [(n_channels, n_blocks) for n_channels, n_blocks in self.layers]


@dataclass
class TrainerConfig:
    batch_size: int
    device: torch.device
    draw_freq: int
    eval_freq: int
    eval_iters: int
    kl_weight: float
    learning_rate: float
    total_iters: int

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(self.device)


@dataclass
class WandBConfig:
    entity: str
    mode: str


@dataclass
class MainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    wandb: WandBConfig

    @classmethod
    def from_dict(cls, config: dict) -> "MainConfig":
        return cls(
            DatasetConfig(**config["dataset"]),
            ModelConfig(**config["model"]),
            TrainerConfig(**config["trainer"]),
            WandBConfig(**config["wandb"]),
        )
