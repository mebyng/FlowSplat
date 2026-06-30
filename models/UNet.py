"""
Lightweight UNet model inspired by OpenAI's improved-diffusion implementation.
Simplified for better efficiency while maintaining the core U-Net architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """Base class for modules that take timestep embeddings."""

    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    """A sequential module that passes timestep embeddings to children that support it."""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """A residual block that accepts timestep embeddings."""

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )

        self.out_norm = nn.GroupNorm(8, self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Upsample(nn.Module):
    """An upsampling layer with optional convolution."""

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with optional convolution."""

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class UNet(nn.Module):
    """
    A lightweight UNet model with timestep embeddings.

    Args:
        in_channels: Number of input channels
        model_channels: Base channel count
        out_channels: Number of output channels
        num_res_blocks: Number of residual blocks per resolution level
        channel_mult: Channel multiplier for each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks=1,
        channel_mult=(2, 4),
        dropout=0.1,
    ):
        super().__init__()

        self.model_channels = model_channels

        # timestep embeddings
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input blocks (encoder)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        input_block_chans = [model_channels]
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                if i == 0:
                    layers = [Downsample(ch, use_conv=True)]
                else:
                    layers = []

                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                    )
                )
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

        # Middle block (bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout=dropout,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout=dropout,
            ),
        )

        # Output blocks (decoder)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                    )
                ]
                ch = model_channels * mult
                if i == num_res_blocks - 1:
                    layers.append(Upsample(ch, use_conv=True))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output layer
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def device(self):
        return next(self.parameters()).device

    def forward(self, x, timesteps=None):
        """
        Apply the model to an input batch.

        Args:
            x: An [N x C x ...] tensor of inputs
            timesteps: A 1-D batch of timesteps

        Returns:
            An [N x C x ...] tensor of outputs
        """

        if timesteps is not None:
            time_emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels)
            ).squeeze(1)
        else:
            time_emb = self.time_embed(
                timestep_embedding(
                    torch.zeros(x.size(0), device=x.device), self.model_channels
                )
            ).squeeze(1)

        # Encoder
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, time_emb)
            hs.append(h)

        # Bottleneck
        h = self.middle_block(h, time_emb)

        # Decoder
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, time_emb)

        # Output
        return self.out(h)
