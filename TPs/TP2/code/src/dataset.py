from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data.dataset import Dataset


class VAEDataset(Dataset):
    paths: list[Path]
    image_size: int

    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Fetch one sample of data.

        ---
        Args:
            index: Sample id.

        ---
        Returns:
            The normalized image.
                Shape of [n_channels, image_size, image_size].
        """
        with Image.open(self.paths[index]) as image:
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image)
        image = torch.FloatTensor(image)
        image = image / 255
        image = rearrange(image, "h w c -> c h w")
        return image

    @classmethod
    def from_dir(cls, path_dir: Path, image_size: int) -> "VAEDataset":
        """Initialize a dataset by gathering all images in a directory."""
        paths = [p for p in path_dir.glob("*.jpg")]
        return cls(paths, image_size)
