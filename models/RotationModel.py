import torch
import torch.nn as nn

from .modules import (
    ConvBlock,
    ResBlock,
    Down,
    Up,
    CrossAttention,
    spatial_positional_embedding,
)

from utils.camera_utils import (
    compute_plucker,
    default_align_cameras,
    rotation_matrix_to_6d,
)


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
        self.in_conv = ConvBlock(
            self.in_channels, self.base_channels, emb_channels=self.emb_channels
        )

        self.enc2 = ResBlock(
            self.base_channels, self.base_channels, emb_channels=self.emb_channels
        )
        self.down1 = Down(
            self.base_channels, self.base_channels * 2, emb_channels=self.emb_channels
        )

        self.enc3 = ResBlock(
            self.base_channels * 2,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )
        self.down2 = Down(
            self.base_channels * 2,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )

        self.enc4 = ResBlock(
            self.base_channels * 4,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )
        self.down3 = Down(
            self.base_channels * 4,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )

    def apply_image_encoder(self, input, condition=None):
        x = self.in_conv(input, condition)  # shape (B, base_channels, H, W)

        x1 = self.enc2(x, condition)  # shape (B, base_channels, H, W)
        x2 = self.down1(x1, condition)  # shape (B, base_channels*2, H/2, W/2)

        x3 = self.enc3(x2, condition)  # shape (B, base_channels*2, H/2, W/2)
        x4 = self.down2(x3, condition)  # shape (B, base_channels*4, H/4, W/4)

        x5 = self.enc4(x4, condition)  # shape (B, base_channels*4, H/4, W/4)
        x6 = self.down3(x5, condition)  # shape (B, base_channels*8, H/8, W/8)

        return x6, (x1, x3, x5)

    def encode_source(self, source):
        return self.apply_image_encoder(source)

    def is_encoded_source(self, source):
        return source.ndim == 4 and source.shape[1] == self.base_channels * 8

    def construct_image_decoder(self):
        if self.mode == "deterministic":
            up1_channels = self.base_channels * 8
            up2_channels = self.base_channels * 4
            up3_channels = self.base_channels * 2
        elif self.mode == "flow":
            up1_channels = self.base_channels * 8 + self.base_channels * 4
            up2_channels = self.base_channels * 4 + self.base_channels * 2
            up3_channels = self.base_channels * 2 + self.base_channels

        self.up1 = Up(
            up1_channels,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )
        self.dec1 = ResBlock(
            self.base_channels * 4,
            self.base_channels * 4,
            emb_channels=self.emb_channels,
        )

        self.up2 = Up(
            up2_channels,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )
        self.dec2 = ResBlock(
            self.base_channels * 2,
            self.base_channels * 2,
            emb_channels=self.emb_channels,
        )

        self.up3 = Up(up3_channels, self.base_channels, emb_channels=self.emb_channels)
        self.dec3 = ResBlock(
            self.base_channels, self.base_channels, emb_channels=self.emb_channels
        )

        self.out_conv = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1)

    def apply_image_decoder(self, latent, skips=None, condition=None):
        if skips is not None:
            x1, x3, x5 = skips
        else:
            x1 = x3 = x5 = None
        x = self.up1(latent, skip=x5, cond=condition)  # (B, base_channels*4, H/4, W/4)
        x = self.dec1(x, emb=condition)  # (B, base_channels*4, H/4, W/4)

        x = self.up2(x, skip=x3, cond=condition)  # (B, base_channels*2, H/2, W/2)
        x = self.dec2(x, emb=condition)  # (B, base_channels*2, H/2, W/2)

        x = self.up3(x, skip=x1, cond=condition)  # (B, base_channels, H, W)
        x = self.dec3(x, emb=condition)  # (B, base_channels, H, W)

        x = self.out_conv(x)  # (B, out_channels, H, W)

        return x

    def construct_core(self):
        self.key_encoder1 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder1 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder1 = ConvBlock(
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
        self.key_encoder2 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder2 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder2 = ConvBlock(
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
        self.key_encoder3 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.value_encoder3 = ConvBlock(
            self.base_channels * 8,
            self.base_channels * 8,
            emb_channels=self.emb_channels,
        )
        self.query_encoder3 = ConvBlock(
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

    def apply_core(self, current, source, time, rotation):

        b, c, h, w = source.shape
        pos_emb = spatial_positional_embedding(
            h, w, self.base_channels * 8, source.device, source.dtype
        ).unsqueeze(0)

        key1 = self.key_encoder1(source + pos_emb, rotation)
        value1 = self.value_encoder1(source)
        query1 = self.query_encoder1(current + pos_emb, rotation)
        x = current + self.cross_attn1(query1, key1, value1)
        x = self.bottleneck_res1(x, time)

        key2 = self.key_encoder2(source + pos_emb, rotation)
        value2 = self.value_encoder2(source)
        query2 = self.query_encoder2(x + pos_emb, rotation)
        x = x + self.cross_attn2(query2, key2, value2)
        x = self.bottleneck_res2(x, time)

        key3 = self.key_encoder3(source + pos_emb, rotation)
        value3 = self.value_encoder3(source)
        query3 = self.query_encoder3(x + pos_emb, rotation)
        x = x + self.cross_attn3(query3, key3, value3)
        x = self.bottleneck_res3(x, time)

        return x

    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input, source, cameras, timestep=None):
        """Forward pass.

        Args:
            input: input image (noisy target) of shape (B, in_channels, H, W)
            source: source image of shape (B, in_channels, H, W)
            cameras: camera parameters
            timestep: timestep/noise condition of shape (B, 1)
        Returns:
            output: tensor of shape (B, out_channels, H, W)
        """
        time_emb = self.apply_time_embedding(timestep)
        rotation = default_align_cameras(cameras["source"], cameras["target"])
        rotation = rotation_matrix_to_6d(rotation[:, :3, :3])
        rotation_emb = self.apply_rotation_embedding(rotation)

        if not self.is_encoded_source(source):
            source, _ = self.encode_source(source)
        source = source
        skips = None

        if self.mode == "deterministic":
            latent = source.clone()
        elif self.mode == "flow":
            latent, skips = self.apply_image_encoder(input)

        # Apply core processing with cross-attention
        core_output = self.apply_core(latent, source, time_emb, rotation_emb)

        output = self.apply_image_decoder(core_output, skips, condition=time_emb)

        return output
