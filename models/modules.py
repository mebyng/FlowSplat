import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        # self.block = _ConvBlock(in_channels, out_channels, emb_channels=emb_channels)
        self.block = ResBlock(in_channels, out_channels, emb_channels=emb_channels)

    def forward(self, x, cond=None):
        x = self.pool(x)
        return self.block(x, cond)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=None):
        super().__init__()
        # self.block = _ConvBlock(in_channels, out_channels, emb_channels=emb_channels)
        self.block = ResBlock(in_channels, out_channels, emb_channels=emb_channels)

    def forward(self, x, skip=None, cond=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x, cond)


def spatial_positional_embedding(height, width, channels, device, dtype):
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
        b, c, h, w = x.shape
        x = self.norm(x)

        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(attn_out)
        return out


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
        return out
