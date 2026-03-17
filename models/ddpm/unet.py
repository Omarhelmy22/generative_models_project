import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time-step conditioning.

    Uses GroupNorm and SiLU activation. A 1x1 convolution is applied
    to the skip connection when in_channels != out_channels.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip_conv(x)


class SelfAttention(nn.Module):
    """Single-head self-attention with residual connection."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.reshape(B, C, H, W)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for DDPM noise prediction.

    Follows the standard encoder-bottleneck-decoder pattern with:
      - Sinusoidal timestep embeddings projected through an MLP
      - ResBlocks with time conditioning at every resolution level
      - Self-attention at specified resolutions
      - Skip connections from encoder to decoder

    Default config for CIFAR-10 (32x32):
      base_channels=64, channel_mult=[1,2,4,8], num_res_blocks=2,
      attention at 16x16 resolution.
    """

    def __init__(self, in_channels=3, base_channels=64,
                 channel_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attention_resolutions=(16,), time_emb_dim=256,
                 dropout=0.0):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        # --- Time embedding MLP ---
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Initial convolution ---
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Build down (encoder) path ---
        self.downs = nn.ModuleList()
        channels_list = [base_channels]
        now_ch = base_channels
        current_res = 32  # assumes 32x32 input

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(now_ch, out_ch, time_emb_dim, dropout)]
                now_ch = out_ch
                if current_res in attention_resolutions:
                    layers.append(SelfAttention(now_ch))
                self.downs.append(nn.ModuleList(layers))
                channels_list.append(now_ch)

            if level != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(now_ch)]))
                channels_list.append(now_ch)
                current_res //= 2

        # --- Middle (bottleneck) ---
        self.mid = nn.ModuleList([
            ResBlock(now_ch, now_ch, time_emb_dim, dropout),
            SelfAttention(now_ch),
            ResBlock(now_ch, now_ch, time_emb_dim, dropout),
        ])

        # --- Build up (decoder) path ---
        self.ups = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                layers = [ResBlock(now_ch + skip_ch, out_ch, time_emb_dim, dropout)]
                now_ch = out_ch
                if current_res in attention_resolutions:
                    layers.append(SelfAttention(now_ch))
                self.ups.append(nn.ModuleList(layers))

            if level != 0:
                self.ups.append(nn.ModuleList([Upsample(now_ch)]))
                current_res *= 2

        # --- Final output ---
        self.final = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        hs = [h]

        # Encoder
        for module in self.downs:
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            hs.append(h)

        # Bottleneck
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Decoder
        for module in self.ups:
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = layer(h, t_emb)
                elif isinstance(layer, (Upsample, Downsample)):
                    h = layer(h)
                else:
                    h = layer(h)

        return self.final(h)
