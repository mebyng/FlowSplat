"""UNet architecture for view synthesis, conditioned on a 3x3 rotation matrix.
Dummy version AI generated to test training loop. Not expected to produce good results.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.cond1 = (
            nn.Linear(emb_channels, out_channels) if emb_channels is not None else None
        )
        self.cond2 = (
            nn.Linear(emb_channels, out_channels) if emb_channels is not None else None
        )

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


class ResBlock(nn.Module):
    """A residual block that accepts timestep embeddings."""

    def __init__(
        self,
        channels,
        out_channels,
        emb_channels=None,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, 2 * self.out_channels),
            )

        self.out_norm = nn.GroupNorm(8, self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        h = self.in_layers(x)
        h = self.out_norm(h)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            scale, shift = emb_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        x = self.norm(x)

        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(attn_out)
        return residual + out


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        # self.block = _ConvBlock(in_channels, out_channels, emb_channels=emb_channels)
        self.block = ResBlock(in_channels, out_channels, emb_channels=emb_channels)

    def forward(self, x, cond=None):
        x = self.pool(x)
        return self.block(x, cond)


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        # self.block = _ConvBlock(in_channels, out_channels, emb_channels=emb_channels)
        self.block = ResBlock(in_channels, out_channels, emb_channels=emb_channels)

    def forward(self, x, skip, cond=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, cond)


class RotationConditionedUNetRes(nn.Module):
    """Lightweight UNet that conditions on a 3x3 rotation matrix."""

    def __init__(
        self, in_channels=3, out_channels=3, base_channels=32, emb_channels=64
    ):
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(1, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
        )

        self.enc = _ConvBlock(in_channels, base_channels, emb_channels=emb_channels)
        self.enc2 = ResBlock(base_channels, base_channels, emb_channels=emb_channels)
        self.enc3 = ResBlock(base_channels, base_channels, emb_channels=emb_channels)

        self.down1 = _Down(base_channels, base_channels * 2, emb_channels=emb_channels)
        self.down2 = _Down(
            base_channels * 2, base_channels * 4, emb_channels=emb_channels
        )

        self.attention1 = SelfAttention(base_channels * 4)
        self.bottleneck1 = ResBlock(
            base_channels * 4, base_channels * 4, emb_channels=emb_channels
        )
        self.attention2 = SelfAttention(base_channels * 4)
        self.bottleneck2 = ResBlock(
            base_channels * 4, base_channels * 8, emb_channels=emb_channels
        )
        self.attention3 = SelfAttention(base_channels * 8)

        self.up1 = _Up(
            base_channels * 8 + base_channels * 2,
            base_channels * 4,
            emb_channels=emb_channels,
        )
        self.up2 = _Up(
            base_channels * 4 + base_channels,
            base_channels * 2,
            emb_channels=emb_channels,
        )

        self.dec = ResBlock(
            base_channels * 2, base_channels * 2, emb_channels=emb_channels
        )
        self.dec2 = ResBlock(
            base_channels * 2, base_channels, emb_channels=emb_channels
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

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
            cond_tmp = torch.zeros(
                image.size(0), self.cond_embed[0].out_features, device=image.device
            )

        x1 = self.enc(image, cond_tmp)  # shape (B, base_channels, H, W)
        x2 = self.enc2(x1, cond_tmp)  # shape (B, base_channels*2, H, W)
        x3 = self.enc3(x2, cond_tmp)  # shape (B, base_channels*2, H, W)
        x4 = self.down1(x3, cond_tmp)  # shape (B, base_channels*2, H/2, W/2)
        x5 = self.down2(x4, cond_tmp)  # shape (B, base_channels*4, H/4, W/4)

        x = self.attention1(x5)
        x = self.bottleneck1(x, cond_tmp)  # shape (B, base_channels*4, H/4, W/4)
        x = self.attention2(x)
        x = self.bottleneck2(x, cond_tmp)  # shape (B, base_channels*8, H/4, W/4)
        x = self.attention3(x)
        x = self.up1(x, x4, cond_tmp)  # shape (B, base_channels*4, H/2, W/2)
        x = self.up2(x, x3, cond_tmp)  # shape (B, base_channels*2, H, W)
        x = self.dec(x, cond_tmp)  # shape (B, base_channels, H, W)
        x = self.dec2(x, cond_tmp)  # shape (B, base_channels, H, W)
        return self.out_conv(x)


class CustomNet(nn.Module):
    """Lightweight UNet that conditions on a 3x3 rotation matrix."""

    def __init__(
        self, in_channels=3, out_channels=3, base_channels=32, emb_channels=64
    ):
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(16, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU(),
        )

        self.enc = _ConvBlock(in_channels, base_channels, emb_channels=emb_channels)
        self.enc2 = ResBlock(base_channels, base_channels, emb_channels=emb_channels)

        self.down1 = _Down(base_channels, base_channels * 2, emb_channels=emb_channels)
        self.enc3 = ResBlock(
            base_channels * 2, base_channels * 2, emb_channels=emb_channels
        )
        self.down2 = _Down(
            base_channels * 2, base_channels * 4, emb_channels=emb_channels
        )
        self.enc4 = ResBlock(
            base_channels * 4, base_channels * 4, emb_channels=emb_channels
        )
        self.down3 = _Down(
            base_channels * 4, base_channels * 8, emb_channels=emb_channels
        )

        self.attention1 = SelfAttention(base_channels * 8)
        self.bottleneck1 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.attention2 = SelfAttention(base_channels * 8)
        self.bottleneck2 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.attention3 = SelfAttention(base_channels * 8)

        self.up1 = _Up(
            base_channels * 8 + base_channels * 4,
            base_channels * 4,
            emb_channels=emb_channels,
        )
        self.dec = ResBlock(
            base_channels * 4, base_channels * 4, emb_channels=emb_channels
        )
        self.up2 = _Up(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            emb_channels=emb_channels,
        )
        self.dec2 = ResBlock(
            base_channels * 2, base_channels * 2, emb_channels=emb_channels
        )
        self.up3 = _Up(
            base_channels * 2 + base_channels,
            base_channels,
            emb_channels=emb_channels,
        )
        self.dec3 = ResBlock(base_channels, base_channels, emb_channels=emb_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, image, condition=None):
        """Forward pass.

        Args:
            image: tensor of shape (B, C, H, W)
            condition: tensor of shape (B, C2)
        Returns:
            tensor of shape (B, out_channels, H, W)
        """
        # cond = torch.cat([plucker.view(batch_size, 6), timestep], dim=1)
        if condition is not None:
            cond_enc = self.cond_embed(condition)
        else:
            cond_enc = torch.zeros(
                image.size(0), self.cond_embed[0].out_features, device=image.device
            )

        x1 = self.enc(image, cond_enc)  # shape (B, base_channels, H, W)
        x2 = self.enc2(x1, cond_enc)  # shape (B, base_channels*2, H, W)
        x3 = self.down1(x2, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x4 = self.enc3(x3, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x5 = self.down2(x4, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x6 = self.enc4(x5, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x7 = self.down3(x6, cond_enc)  # shape (B, base_channels*8, H/8, W/8)

        x = self.attention1(x7)
        x = self.bottleneck1(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention2(x)
        x = self.bottleneck2(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention3(x)
        x = self.up1(x, x6, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.dec(x, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.up2(x, x4, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.dec2(x, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.up3(x, x2, cond_enc)  # shape (B, base_channels, H, W)
        x = self.dec3(x, cond_enc)  # shape (B, base_channels, H, W)
        return self.out_conv(x)


class CustomNetSpatialRotation(CustomNet):
    """Lightweight UNet that conditions on a 3x3 rotation matrix."""

    def __init__(
        self, in_channels=3, out_channels=3, base_channels=32, emb_channels=64
    ):
        super().__init__(in_channels, out_channels, base_channels, emb_channels)
        self.rotation_embed = nn.Sequential(
            _ConvBlock(in_channels=18, out_channels=base_channels),
            _ConvBlock(in_channels=base_channels, out_channels=base_channels * 2),
            _ConvBlock(in_channels=base_channels * 2, out_channels=base_channels * 4),
            _ConvBlock(in_channels=base_channels * 4, out_channels=base_channels * 8),
        )

    def forward(self, image, condition=None):
        """Forward pass.

        Args:
            image: tensor of shape (B, C, H, W)
            condition: tensor of shape (B, C2)
        Returns:
            tensor of shape (B, out_channels, H, W)
        """
        # cond = torch.cat([plucker.view(batch_size, 6), timestep], dim=1)
        if condition is not None:
            cond_enc = self.cond_embed(condition)
        else:
            cond_enc = torch.zeros(
                image.size(0), self.cond_embed[0].out_features, device=image.device
            )

        # create rotation image of shape (B, 18, H/4, W/4) from condition + sinusoidal 2D image coordinate encoding
        B, C, H, W = image.shape
        if condition is not None:
            rot_image = condition.view(B, 16, 1, 1).expand(-1, -1, H // 8, W // 8)
            ys = torch.linspace(
                -1.0, 1.0, H // 8, device=image.device, dtype=image.dtype
            )
            xs = torch.linspace(
                -1.0, 1.0, W // 8, device=image.device, dtype=image.dtype
            )
            yy, xx = torch.meshgrid(ys, xs)
            coord_enc = torch.stack(
                [torch.sin(math.pi * xx), torch.sin(math.pi * yy)], dim=0
            )
            coord_enc = coord_enc.unsqueeze(0).expand(B, -1, -1, -1)

            rot_image = torch.cat([rot_image, coord_enc], dim=1)
            rot_feat = self.rotation_embed(rot_image)
        else:
            rot_feat = torch.zeros(
                B, self.rotation_embed.out_channels, H // 8, W // 8, device=image.device
            )

        x1 = self.enc(image, cond_enc)  # shape (B, base_channels, H, W)
        x2 = self.enc2(x1, cond_enc)  # shape (B, base_channels*2, H, W)
        x3 = self.down1(x2, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x4 = self.enc3(x3, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x5 = self.down2(x4, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x6 = self.enc4(x5, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x7 = self.down3(x6, cond_enc)  # shape (B, base_channels*8, H/8, W/8)

        x = self.attention1(x7 + rot_feat)
        x = self.bottleneck1(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention2(x + rot_feat)
        x = self.bottleneck2(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention3(x + rot_feat)
        x = self.up1(x, x6, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.dec(x, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.up2(x, x4, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.dec2(x, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.up3(x, x2, cond_enc)  # shape (B, base_channels, H, W)
        x = self.dec3(x, cond_enc)  # shape (B, base_channels, H, W)
        return self.out_conv(x)
