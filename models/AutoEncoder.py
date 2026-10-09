"""Residual autoencoder with four spatial downsampling and upsampling stages."""

import torch
import torch.nn as nn

from .modules import Down, ResBlock, Up


class SimpleAutoEncoder(nn.Module):
    """Compress images to (B, latent_dim, H//16, W//16).

    Shared residual Down/Up blocks provide two convolutions per stage;
    Input dimensions should be divisible by 16.
    """

    def __init__(self, latent_dim=4, use_depth=False, noise_sigma=0.001):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.channels = 4 if use_depth else 3
        self.register_buffer("latent_dim", torch.tensor(latent_dim, dtype=torch.int64))
        self.register_buffer("use_depth", torch.tensor(use_depth, dtype=torch.bool))
        self.encoder = nn.Sequential(
            ResBlock(self.channels, 16),
            Down(16, 32),
            ResBlock(32, 32),
            Down(32, 64),
            ResBlock(64, 64),
            Down(64, 64),
            ResBlock(64, 64),
            Down(64, latent_dim),
            ResBlock(latent_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            Up(latent_dim, 64),
            ResBlock(64, 64),
            Up(64, 64),
            ResBlock(64, 64),
            Up(64, 32),
            ResBlock(32, 32),
            Up(32, 16),
            ResBlock(16, 16),
        )

        self.rgb_head = ResBlock(16, 3)
        if use_depth:
            self.depth_head = ResBlock(16, 1)

    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            torch.load(path, map_location=self.device(), weights_only=True)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder(z)
        out = self.rgb_head(h)
        if self.use_depth:
            depth = self.depth_head(h)
            out = torch.cat([out, depth], dim=1)
        return out

    def forward(self, x):
        z = self.encode(x)
        if self.training:
            z = z + self.noise_sigma * torch.randn_like(z)
        return self.decode(z), z
