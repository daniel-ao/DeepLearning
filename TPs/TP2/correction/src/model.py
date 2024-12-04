import torch
from einops import rearrange
import torch.nn as nn


class EncoderBlock(nn.Module):
    project: nn.Conv2d
    blocks: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()

        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding="same"
                    ),
                    nn.GroupNorm(num_groups=1, num_channels=out_channels),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the dimension of the input batch and then apply a sequence of
        convolutions.

        ---
        Args:
            x: The input batch of images.
                Shape of [batch_size, in_channels, height, width].

        ---
        Returns:
            The processed batch.
                Shape of [batch_size, out_channels, height/2, width/2].
        """
        x = self.project(x)
        for block in self.blocks:
            x = block(x) + x
        return x


class DecoderBlock(nn.Module):
    project: nn.Conv2d
    blocks: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()

        self.project = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
                    nn.GroupNorm(num_groups=1, num_channels=in_channels),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand the dimension of the input batch and then apply a sequence of
        convolutions.

        ---
        Args:
            x: The input batch of images.
                Shape of [batch_size, in_channels, height, width].

        ---
        Returns:
            The processed batch.
                Shape of [batch_size, out_channels, height*2, width*2].
        """
        for block in self.blocks:
            x = block(x) + x
        x = self.project(x)
        return x


class VAE(nn.Module):
    encoders: nn.ModuleList
    decoders: nn.ModuleList
    project_latent: nn.Conv2d

    def __init__(self, image_channels: int, layers: list[tuple[int, int]]):
        super().__init__()

        encoder, decoder = [], []
        in_channels = image_channels
        for out_channels, n_layers in layers:
            encoder.append(EncoderBlock(in_channels, out_channels, n_layers))
            decoder.append(DecoderBlock(out_channels, in_channels, n_layers))
            in_channels = out_channels

        self.encoders = nn.ModuleList(encoder)
        self.decoders = nn.ModuleList(reversed(decoder))
        self.project_latent = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=1, stride=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Do a full encoder-decoder step.

        ---
        Args:
            x: The batch of images.
                Shape of [batch_size, image_channels, height, width].

        ---
        Returns:
            The decoder images.
                Shape of [batch_size, image_channels, height, width].
        """
        for encoder in self.encoders:
            x = encoder(x)

        x = self.project_latent(x)
        mu, logvar = torch.chunk(x, chunks=2, dim=1)
        x = VAE.reparameterize(mu, logvar)

        for decoder in self.decoders:
            x = decoder(x)

        x = torch.sigmoid(x)
        return x, (mu, logvar)

    @torch.inference_mode()
    def generate(
        self, latent_shape: torch.Size, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Generate random images.

        ---
        Args:
            latent_shape: Dimensions of the latent space.
            generator: Optional generator to fix randomness.

        ---
        Returns:
            The generated images.
                Shape of [batch_size, n_channels, height, width].
        """
        self.eval()
        device = next(self.parameters()).device

        mu = torch.zeros(latent_shape, device=device)
        logvar = torch.zeros(latent_shape, device=device)
        x = VAE.reparameterize(mu, logvar, generator)

        for decoder in self.decoders:
            x = decoder(x)

        x = torch.sigmoid(x)
        return x

    @torch.inference_mode()
    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, n_points: int
    ) -> torch.Tensor:
        """Generate the sequence of images along the latent line between the two input
        images.

        ---
        Args:
            x1: Image source.
                Shape of [n_channels, height, width].
            x2: Image target.
                Shape of [n_channels, height, width].
            n_points: The number of latent points.

        ---
        Returns:
            The interpolated latent images.
                Shape of [n_points, n_channels, height, witdh].
        """
        self.eval()
        x = torch.stack((x1, x2))

        # Encode both images.
        for encoder in self.encoders:
            x = encoder(x)
        x = self.project_latent(x)

        # Only keep the mean of their latent points.
        (x1, x2), _ = torch.chunk(x, chunks=2, dim=1)

        # Generate the sequence of points along the line joining both latent points.
        t = torch.linspace(0, 1, steps=n_points, device=x.device)
        t = rearrange(t, "n -> n 1 1 1")
        x1 = rearrange(x1, "c h w -> 1 c h w")
        x2 = rearrange(x2, "c h w -> 1 c h w")
        x = (1 - t) * x1 + t * x2

        # Finally decode all latent points at once.
        for decoder in self.decoders:
            x = decoder(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from the given gaussian distribution using the reparameterization
        trick.

        A deeper explanation can be found here: https://gregorygundersen.com/blog/2018/04/29/reparameterization/

        ---
        Args:
            mu: Mean of the gaussian distribution.
                Shape of [batch_size, n_channels, height, width].
            logvar: Log-var of the gaussian distribution.
                Shape of [batch_size, n_channels, height, width].
            generator: Used to fix the randomness, if not None.

        ---
        Returns:
            The gaussian samples.
                Shape of [batch_size, n_channels, height, width].
        """
        std = torch.exp(logvar / 2)
        eps = torch.randn(std.shape, generator=generator, device=std.device)
        sample = mu + (eps * std)
        return sample
