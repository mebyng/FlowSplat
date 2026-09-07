"""UNet architecture for view synthesis, conditioned on a 3x3 rotation matrix.
Dummy version AI generated to test training loop. Not expected to produce good results.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.camera_utils import (
    compute_plucker,
    default_align_cameras,
    rotation_matrix_to_6d,
)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.cond1 = (
            nn.Linear(emb_channels, out_channels * 2)
            if emb_channels is not None
            else None
        )
        self.cond2 = (
            nn.Linear(emb_channels, out_channels * 2)
            if emb_channels is not None
            else None
        )

    def forward(self, x, cond=None):
        x = self.conv1(x)
        x = self.norm1(x)
        if cond is not None:
            scale, shift = self.cond1(cond)[:, :, None, None].chunk(2, dim=1)
            x = x * (1 + scale) + shift
        x = F.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if cond is not None:
            scale, shift = self.cond2(cond)[:, :, None, None].chunk(2, dim=1)
            x = x * (1 + scale) + shift
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


def _spatial_positional_embedding(height, width, channels, device, dtype):
    """Build a 2D sinusoidal positional embedding for flattened image tokens."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    frequency_count = max(1, (channels + 3) // 4)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(frequency_count, device=device, dtype=dtype)
        / frequency_count
    )
    x_angles = xx[None] * frequencies[:, None, None]
    y_angles = yy[None] * frequencies[:, None, None]
    embedding = torch.cat(
        [
            torch.sin(x_angles),
            torch.cos(x_angles),
            torch.sin(y_angles),
            torch.cos(y_angles),
        ],
        dim=0,
    )
    return embedding[:channels]


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


class CrossAttention(nn.Module):
    """Cross-attention module that combines query features with key and value features."""

    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.norm_q = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, query, key, value):
        """
        Args:
            query: (B, C, H, W) - primary features to query
            key: (B, C, H, W) - key features for attention
            value: (B, C, H, W) - value features for attention
        Returns:
            output: (B, C, H, W) - attended features
        """
        residual = query
        b, c, h, w = query.shape

        query_norm = self.norm_q(query)
        key_norm = self.norm_kv(key)
        value_norm = self.norm_kv(value)

        query_flat = query_norm.flatten(2).transpose(1, 2)
        key_flat = key_norm.flatten(2).transpose(1, 2)
        value_flat = value_norm.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(query_flat, key_flat, value_flat)
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

    def forward(self, x, skip=None, cond=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
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
        self.base_channels = base_channels
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

        self.bottleneck01 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.bottleneck02 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.attention1 = SelfAttention(base_channels * 8)
        self.bottleneck11 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.bottleneck12 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.attention2 = SelfAttention(base_channels * 8)
        self.bottleneck21 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.bottleneck22 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.attention3 = SelfAttention(base_channels * 8)
        self.bottleneck31 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )
        self.bottleneck32 = ResBlock(
            base_channels * 8, base_channels * 8, emb_channels=emb_channels
        )

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

        b, c, h, w = image.shape
        pos_emb = _spatial_positional_embedding(
            h, w, self.base_channels * 8, image.device, image.dtype
        ).unsqueeze(0)

        x1 = self.enc(image, cond_enc)  # shape (B, base_channels, H, W)
        x2 = self.enc2(x1, cond_enc)  # shape (B, base_channels*2, H, W)
        x3 = self.down1(x2, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x4 = self.enc3(x3, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x5 = self.down2(x4, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x6 = self.enc4(x5, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x7 = self.down3(x6, cond_enc)  # shape (B, base_channels*8, H/8, W/8)

        x = self.bottleneck01(x7, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck02(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention1(x + pos_emb)
        x = self.bottleneck11(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck12(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention2(x + pos_emb)
        x = self.bottleneck21(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck22(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention3(x + pos_emb)
        x = self.bottleneck31(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck32(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
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

    def forward(self, input, condition, rotation, timestep=None):
        """Forward pass.

        Args:
            image: tensor of shape (B, C, H, W)
            condition: tensor of shape (B, C2)
        Returns:
            tensor of shape (B, out_channels, H, W)
        """
        # cond = torch.cat([plucker.view(batch_size, 6), timestep], dim=1)
        if timestep is not None:
            cond_enc = self.cond_embed(timestep)
        else:
            cond_enc = torch.zeros(
                input.size(0), self.cond_embed[0].out_features, device=input.device
            )

        # create rotation image of shape (B, 18, H/4, W/4) from condition + sinusoidal 2D image coordinate encoding
        B, C, H, W = input.shape
        rot_image = rotation.view(B, 16, 1, 1).expand(-1, -1, H // 8, W // 8)
        ys = torch.linspace(-1.0, 1.0, H // 8, device=input.device, dtype=input.dtype)
        xs = torch.linspace(-1.0, 1.0, W // 8, device=input.device, dtype=input.dtype)
        yy, xx = torch.meshgrid(ys, xs)
        coord_enc = torch.stack(
            [torch.sin(math.pi * xx), torch.sin(math.pi * yy)], dim=0
        )
        coord_enc = coord_enc.unsqueeze(0).expand(B, -1, -1, -1)

        rot_image = torch.cat([rot_image, coord_enc], dim=1)
        rot_feat = self.rotation_emb
        pos_emb = _spatial_positional_embedding(
            H, W, self.base_channels * 8, input.device, input.dtype
        ).unsqueeze(0)

        # for the moment ignore input noise
        x1 = self.enc(condition, cond_enc)  # shape (B, base_channels, H, W)
        x2 = self.enc2(x1, cond_enc)  # shape (B, base_channels*2, H, W)
        x3 = self.down1(x2, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x4 = self.enc3(x3, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x5 = self.down2(x4, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x6 = self.enc4(x5, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x7 = self.down3(x6, cond_enc)  # shape (B, base_channels*8, H/8, W/8)

        x = self.bottleneck01(x7, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck02(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention1(x7 + rot_feat)  # Add rotation features to the bottleneck
        x = self.bottleneck11(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck12(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention2(x + rot_feat)
        x = self.bottleneck21(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck22(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.attention3(x + rot_feat)
        x = self.bottleneck31(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.bottleneck32(x, cond_enc)  # shape (B, base_channels*8, H/8, W/8)
        x = self.up1(x, x6, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.dec(x, cond_enc)  # shape (B, base_channels*4, H/4, W/4)
        x = self.up2(x, x4, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.dec2(x, cond_enc)  # shape (B, base_channels*2, H/2, W/2)
        x = self.up3(x, x2, cond_enc)  # shape (B, base_channels, H, W)
        x = self.dec3(x, cond_enc)  # shape (B, base_channels, H, W)
        return self.out_conv(x)


class AttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        mode="deterministic",
        in_channels=3,
        out_channels=3,
        base_channels=32,
        emb_channels=64,
        num_attn_heads=4,
    ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels
        self.num_attn_heads = num_attn_heads

        self.construct_time_embedding()
        self.construct_rotation_embedding()
        self.construct_image_encoder()
        self.construct_image_decoder()
        self.construct_core()

    def construct_time_embedding(self):
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
        )

    def apply_time_embedding(self, timestep):
        if timestep is not None:
            time_emb = self.time_embed(timestep)
        else:
            time_emb = None
        return time_emb

    def construct_rotation_embedding(self):
        self.rotation_embed = nn.Sequential(
            nn.Linear(6, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.SiLU(),
        )

    def apply_rotation_embedding(self, rotation):
        return self.rotation_embed(rotation)

    def construct_image_encoder(self):
        self.in_conv = _ConvBlock(
            self.in_channels, self.base_channels, emb_channels=self.emb_channels
        )

        self.enc2 = ResBlock(
            self.base_channels, self.base_channels, emb_channels=self.emb_channels
        )
        self.down1 = _Down(
            self.base_channels, self.base_channels * 2, emb_channels=self.emb_channels
        )

        self.enc3 = ResBlock(
            self.base_channels * 2,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )
        self.down2 = _Down(
            self.base_channels * 2,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )

        self.enc4 = ResBlock(
            self.base_channels * 4,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )
        self.down3 = _Down(
            self.base_channels * 4,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )

    def apply_image_encoder(self, input, condition=None):
        x = self.in_conv(input, condition)  # shape (B, base_channels, H, W)

        x = self.enc2(x, condition)  # shape (B, base_channels*2, H, W)
        x = self.down1(x, condition)  # shape (B, base_channels*2, H/2, W/2)

        x = self.enc3(x, condition)  # shape (B, base_channels*2, H/2, W/2)
        x = self.down2(x, condition)  # shape (B, base_channels*4, H/4, W/4)

        x = self.enc4(x, condition)  # shape (B, base_channels*4, H/4, W/4)
        x = self.down3(x, condition)  # shape (B, base_channels*8, H/8, W/8)

        return x

    def construct_image_decoder(self):
        self.up1 = _Up(
            self.base_channels * 8,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )
        self.dec1 = ResBlock(
            self.base_channels * 4,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )

        self.up2 = _Up(
            self.base_channels * 4,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )
        self.dec2 = ResBlock(
            self.base_channels * 2,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )

        self.up3 = _Up(
            self.base_channels * 2, self.base_channels, emb_channels=self.emb_channels
        )
        self.dec3 = ResBlock(
            self.base_channels, self.base_channels, emb_channels=self.emb_channels
        )

        self.out_conv = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1)

    def apply_image_decoder(self, latent, condition=None):
        x = self.up1(latent, condition)  # shape (B, base_channels*4, H/4, W/4)
        x = self.dec1(x, condition)  # shape (B, base_channels*4, H/4, W/4)

        x = self.up2(x, condition)  # shape (B, base_channels*2, H/2, W/2)
        x = self.dec2(x, condition)  # shape (B, base_channels*2, H/2, W/2)

        x = self.up3(x, condition)  # shape (B, base_channels, H, W)
        x = self.dec3(x, condition)  # shape (B, base_channels, H, W)

        x = self.out_conv(x)  # shape (B, out_channels, H, W)

        return x

    def construct_core(self):
        self.key_encoder1 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder1 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder1 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.cross_attn1 = CrossAttention(
            self.base_channels * 8, num_heads=self.num_attn_heads
        )
        self.bottleneck_res1 = ResBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.key_encoder2 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder2 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder2 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.cross_attn2 = CrossAttention(
            self.base_channels * 8, num_heads=self.num_attn_heads
        )
        self.bottleneck_res2 = ResBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.key_encoder3 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder3 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder3 = _ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.cross_attn3 = CrossAttention(
            self.base_channels * 8, num_heads=self.num_attn_heads
        )
        self.bottleneck_res3 = ResBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )

    def apply_core(self, current, source, condition, rotation):

        b, c, h, w = source.shape
        pos_emb = _spatial_positional_embedding(
            h, w, self.base_channels * 8, source.device, source.dtype
        ).unsqueeze(0)

        key1 = self.key_encoder1(source + pos_emb, rotation)
        value1 = self.value_encoder1(source)
        query1 = self.query_encoder1(current + pos_emb, rotation)
        x = self.cross_attn1(query1, key1, value1)
        x = self.bottleneck_res1(x, condition)

        key2 = self.key_encoder2(source + pos_emb, rotation)
        value2 = self.value_encoder2(source)
        query2 = self.query_encoder2(x + pos_emb, rotation)
        x = self.cross_attn2(query2, key2, value2)
        x = self.bottleneck_res2(x, condition)

        key3 = self.key_encoder3(source + pos_emb, rotation)
        value3 = self.value_encoder3(source)
        query3 = self.query_encoder3(x + pos_emb, rotation)
        x = self.cross_attn3(query3, key3, value3)
        x = self.bottleneck_res3(x, condition)

        return x

    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input, condition, cameras, timestep=None):
        """Forward pass.

        Args:
            input: input tensor of shape (B, in_channels, H, W)
            condition: conditioning tensor of shape (B, in_channels, H, W)
            cameras: camera parameters
            timestep: timestep/noise condition of shape (B, 1)
        Returns:
            output: tensor of shape (B, out_channels, H, W)
        """
        time_emb = self.apply_time_embedding(timestep)
        rotation = default_align_cameras(cameras["source"], cameras["target"])
        rotation = rotation_matrix_to_6d(rotation[:, :3, :3])
        rotation_emb = self.apply_rotation_embedding(rotation)

        source = self.apply_image_encoder(condition)

        if self.mode == "deterministic":
            latent = source.clone()
        elif self.mode == "flow":
            latent = self.apply_image_encoder(input)

        # Apply core processing with cross-attention
        core_output = self.apply_core(latent, source, time_emb, rotation_emb)

        output = self.apply_image_decoder(core_output, condition=time_emb)

        return output
