"""UNet architecture for view synthesis, conditioned on a 3x3 rotation matrix.
Dummy version AI generated to test training loop. Not expected to produce good results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.cond1 = nn.Linear(cond_dim, out_channels) if cond_dim is not None else None
        self.cond2 = nn.Linear(cond_dim, out_channels) if cond_dim is not None else None

    def forward(self, x, cond=None):
        x = self.conv1(x)
        x = self.norm1(x)
        if cond is not None:
            x = x + self.cond1(cond)[:, :, None, None]
        x = F.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if cond is not None:
            x = x + self.cond2(cond)[:, :, None, None]
        x = F.silu(x)
        return x


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = _ConvBlock(in_channels, out_channels, cond_dim=cond_dim)

    def forward(self, x, cond=None):
        x = self.pool(x)
        return self.block(x, cond=cond)


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=None):
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels, cond_dim=cond_dim)

    def forward(self, x, skip, cond=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, cond=cond)


class RotationConditionedUNet(nn.Module):
    """Lightweight UNet that conditions on a 3x3 rotation matrix."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=32, cond_dim=64):
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        self.enc = _ConvBlock(in_channels, base_channels, cond_dim=cond_dim)
        self.down1 = _Down(base_channels, base_channels * 2, cond_dim=cond_dim)
        self.down2 = _Down(base_channels * 2, base_channels * 4, cond_dim=cond_dim)

        self.bottleneck1 = _ConvBlock(base_channels * 4, base_channels * 8, cond_dim=cond_dim)
        self.bottleneck2 = _ConvBlock(base_channels * 8, base_channels * 8, cond_dim=cond_dim)
        self.bottleneck3 = _ConvBlock(base_channels * 8, base_channels * 8, cond_dim=cond_dim)
        self.bottleneck4 = _ConvBlock(base_channels * 8, base_channels * 8, cond_dim=cond_dim)

        self.up1 = _Up(base_channels * 8 + base_channels * 2, base_channels * 4, cond_dim=cond_dim)
        self.up2 = _Up(base_channels * 4 + base_channels, base_channels * 2, cond_dim=cond_dim)
        self.dec = _ConvBlock(base_channels * 2, base_channels, cond_dim=cond_dim)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def device(self):
        return next(self.parameters()).device

    def forward(self, image, timestep=None):
        """Forward pass.

        Args:
            image: tensor of shape (B, C, H, W)
            timestep: tensor of shape (B, 1)
        Returns:
            tensor of shape (B, out_channels, H, W)
        """
        # cond = torch.cat([plucker.view(batch_size, 6), timestep], dim=1)
        if timestep is not None:
            cond_tmp = self.cond_embed(timestep)
        else:
            cond_tmp = torch.zeros(image.size(0), self.cond_embed[0].out_features, device=image.device)

        x1 = self.enc(image, cond=cond_tmp) # shape (B, base_channels, H, W)
        x2 = self.down1(x1, cond=cond_tmp) # shape (B, base_channels*2, H/2, W/2)
        x3 = self.down2(x2, cond=cond_tmp) # shape (B, base_channels*4, H/4, W/4)

        x = self.bottleneck1(x3, cond=cond_tmp) # shape (B, base_channels*8, H/4, W/4)
        x = self.bottleneck2(x, cond=cond_tmp) # shape (B, base_channels*8, H/4, W/4)
        x = self.bottleneck3(x, cond=cond_tmp) # shape (B, base_channels*8, H/4, W/4)
        x = self.bottleneck4(x, cond=cond_tmp) # shape (B, base_channels*8, H/4, W/4)
        x = self.up1(x, x2, cond=cond_tmp) # shape (B, base_channels*4, H/2, W/2)
        x = self.up2(x, x1, cond=cond_tmp) # shape (B, base_channels*2, H, W)
        x = self.dec(x, cond=cond_tmp) # shape (B, base_channels, H, W)
        return self.out_conv(x)