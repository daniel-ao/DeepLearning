import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """A single encoder block.

    The `project` convolutional layer compress the input images from
    [batch_size, in_channels, height, with] to
    [batch_size, out_channels, height/2, width/2].

    Then the blocks are applied with residual connections. A single block is a sequence
    of Conv2d -> Norm -> Activation.
    """

    project: nn.Conv2d
    blocks: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()

        self.project = ...
        self.blocks = ...

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
        ...


class DecoderBlock(nn.Module):
    """A single decoder block.

    First the blocks are applied in the same fashion as what the encoder is doing,
    using Conv -> Norm -> Activation, and residual connections.

    Then `project` layer decompress the images from
    [batch_size, in_channels, height, with] to
    [batch_size, out_channels, height*2, width*2], using a ConvTranspose2d.
    """

    project: nn.Conv2d
    blocks: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()

        self.project = ...
        self.blocks = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a sequence of convolutions and then expand the dimension of the input
        batch.

        ---
        Args:
            x: The input batch of images.
                Shape of [batch_size, in_channels, height, width].

        ---
        Returns:
            The processed batch.
                Shape of [batch_size, out_channels, height*2, width*2].
        """
        ...


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
            The projected latent informations as a tuple:
                mu: The mean of the gaussians.
                    For the AE, replace it with the predicted latent point.
                    Shape of [batch_size, latent_dim, height_latent, width_latent].
                logvar: The log-variance of the gaussians.
                    Shape of [batch_size, latent_dim, height_latent, width_latent].
                    For the AE, replace it with 0s.
        """
        ...

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
        ...

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
