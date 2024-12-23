import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torchinfo
import wandb
from einops import rearrange, reduce
from PIL import Image
from torch.functional import F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, RandomSampler
from tqdm import tqdm
from wandb.data_types import WBValue
from wandb.wandb_run import Run

from .dataset import VAEDataset
from .model import VAE


@dataclass
class Trainer:
    batch_size: int
    device: torch.device
    draw_freq: int
    eval_freq: int
    eval_iters: int
    kl_weight: float
    total_iters: int

    def train(
        self,
        model: VAE,
        optimizer: Optimizer,
        train_dataset: VAEDataset,
        val_dataset: VAEDataset,
        logger: Run,
    ):
        """Launch the training for the given model.

        ---
        Args:
            model: VAE model to train.
            optimizer: Initialized optimizer.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            logger: Associated WandB run.
        """
        sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=self.total_iters * self.batch_size,
        )
        dataloader = DataLoader(train_dataset, self.batch_size, sampler=sampler)
        model.to(self.device)

        torchinfo.summary(
            model, input_size=train_dataset[0].shape, batch_dim=0, device=self.device
        )

        batch = next(iter(dataloader)).to(self.device)
        _, (mu, logvar) = model(batch)
        original_dim = batch.reshape(self.batch_size, -1).shape[1]
        latent_dim = mu.reshape(self.batch_size, -1).shape[1]
        print(f"\nOriginal dimension: {original_dim:,}")
        print(f"Latent dimension: {latent_dim:,}")
        print(f"Compression rate: {1 - latent_dim/original_dim:.2%}")
        print(f"Training on {self.device}")

        for iter_id, batch in tqdm(
            enumerate(dataloader), "Training", total=self.total_iters
        ):
            self.batch_update(model, optimizer, batch)

            if iter_id % self.eval_freq == 0:
                metrics = {
                    "train": self.eval(model, train_dataset),
                    "val": self.eval(model, val_dataset),
                }
                logger.log(metrics, step=iter_id)

            if iter_id % self.draw_freq == 0:
                self.interpolate(
                    model, val_dataset, gif_path=f"interpolation-{iter_id}.gif"
                )

        # Evaluate the model one last time, and generate some interpolation gif.
        metrics = {
            "train": self.eval(model, train_dataset),
            "val": self.eval(model, val_dataset),
        }
        logger.log(metrics, step=iter_id)
        self.interpolate(model, val_dataset, gif_path="interpolation-final.gif")

    @torch.inference_mode()
    def eval(self, model: VAE, dataset: VAEDataset) -> dict[str, WBValue]:
        """Evaluate the model on the given dataset."""
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=self.eval_iters * self.batch_size,
        )
        dataloader = DataLoader(dataset, self.batch_size, sampler=sampler)

        metrics = defaultdict(list)
        for batch in tqdm(dataloader, "Evaluating", total=self.eval_iters, leave=False):
            for name, values in self.batch_metrics(model, batch).items():
                metrics[name].append(values)

        for name, values in metrics.items():
            values = torch.concatenate(values)
            metrics[name] = values.mean().item()

        batch = next(iter(dataloader)).to(self.device)
        recons, (mu, logvar) = model(batch)
        generated = model.generate(mu.shape)

        metrics["mu"] = wandb.Histogram(mu.flatten().cpu())
        metrics["logvar"] = wandb.Histogram(logvar.flatten().cpu())
        metrics["images.original"] = wandb.Image(batch)
        metrics["images.reconstructed"] = wandb.Image(recons)
        metrics["images.generated"] = wandb.Image(generated)

        return metrics

    # @torch.compile
    def batch_update(self, model: VAE, optimizer: Optimizer, batch: torch.Tensor):
        """Do the full forward-backward-update step."""
        model.train()

        batch = batch.to(self.device)
        recons, (mu, logvar) = model(batch)

        mse = F.mse_loss(batch, recons, reduction="none")
        mse = reduce(mse, "b c h w -> b", "mean")
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = reduce(kl, "b c h w -> b", "mean")
        loss = mse + self.kl_weight * kl

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    # @torch.compile
    def batch_metrics(self, model: VAE, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate the model on the given batch of samples.

        ---
        Args:
            model: Model to evaluate.
            batch: Batch of samples.
                Shape of [batch_size, n_channels, height, width].

        ---
        Returns:
            The metrics.
                Each entry is of shape [batch_size,].
        """
        model.eval()

        batch = batch.to(self.device)
        recons, (mu, logvar) = model(batch)

        mse = F.mse_loss(batch, recons, reduction="none")
        mse = reduce(mse, "b c h w -> b", "mean")
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = reduce(kl, "b c h w -> b", "mean")
        loss = mse + self.kl_weight * kl

        return {
            "loss": loss,
            "mse": mse,
            "kl": kl,
        }

    def interpolate(
        self,
        model: VAE,
        dataset: VAEDataset,
        n_points: int = 50,
        gif_path: str = "interpolation.gif",
    ):
        """Generate and save an animation moving from multiple random images of the
        dataset.
        """
        all_frames = []
        x = torch.randint(low=0, high=len(dataset), size=(20,))
        x = [dataset[i].to(self.device) for i in x]

        for x1, x2 in itertools.pairwise(x + [x[0]]):
            frames = model.interpolate(x1, x2, n_points)
            frames = rearrange(frames, "b c h w -> b h w c")
            frames = frames.cpu().numpy()
            frames = (frames * 255).astype(np.uint8)
            frames = [Image.fromarray(f) for f in frames]

            all_frames.extend(frames)

        all_frames[0].save(
            gif_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=10_000 / len(all_frames),  # Such that it lasts 10 seconds.
            loop=0,
        )
