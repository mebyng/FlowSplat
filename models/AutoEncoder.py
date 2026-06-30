"""Simple autoencoder that reduces spatial dimensions by a factor of 4."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class ResidualBlock(nn.Module):
    """Residual block used before and after downsampling operations."""
    
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class SimpleAutoEncoder(nn.Module):
    """
    A simple autoencoder that compresses input by reducing spatial dimensions by 4x.
    
    Architecture:
    - Encoder: conv + residual block -> downsample -> residual block -> downsample -> residual block
    - Decoder: 2 stride-2 transposed convolutions (upsamples back to original size)
    
    The latent representation has shape: (batch, latent_dim, height//4, width//4)
    """
    
    def __init__(self, in_channels=3, latent_dim=64):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            latent_dim: Number of channels in the bottleneck/latent space
        """
        super().__init__()
        
        # Initial convolution, then residual block before first downsample
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True),
        )
        self.res_before_down1 = ResidualBlock(16)

        # First downsample and residual block after it
        self.down1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.res_after_down1 = ResidualBlock(32)

        # Second downsample and residual block after it
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.res_after_down2 = ResidualBlock(latent_dim)

        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.res_after_down3 = ResidualBlock(latent_dim)

        self.down4 = nn.Sequential(
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(inplace=True),
        )
        self.res_after_down4 = ResidualBlock(latent_dim)
        
        # Decoder: upsamples by 4x using interpolation and conv blocks, with residual blocks
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.res_after_up1 = ResidualBlock(64)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.res_after_up2 = ResidualBlock(64)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.res_after_up3 = ResidualBlock(32)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True),
        )
        self.res_after_up4 = ResidualBlock(16)

        self.reconstruction = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)

    def device(self):
        """Return the device of the model parameters."""
        return next(self.parameters()).device
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def encode(self, x):
        """Encode input to latent representation."""
        x = self.initial_conv(x)
        x = self.res_before_down1(x)
        x = self.down1(x)
        x = self.res_after_down1(x)
        x = self.down2(x)
        x = self.res_after_down2(x)
        x = self.down3(x)
        x = self.res_after_down3(x)
        x = self.down4(x)
        x = self.res_after_down4(x)
        return x
    
    def decode(self, z):
        """Decode latent representation back to input space."""
        x = self.up1(z)
        x = self.res_after_up1(x)
        x = self.up2(x)
        x = self.res_after_up2(x)
        x = self.up3(x)
        x = self.res_after_up3(x)
        x = self.up4(x)
        x = self.res_after_up4(x)
        x = self.reconstruction(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
