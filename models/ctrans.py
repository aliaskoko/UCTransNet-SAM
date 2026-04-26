import torch
import torch.nn as nn

class CTrans(nn.Module):
    """
    Channel Transformer skip connection.
    Applies multi-head cross-channel attention between
    encoder feature E and decoder feature D, then fuses them.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = channels // num_heads
        self.scale       = self.head_dim ** -0.5

        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc, dec):
        B, C, H, W = enc.shape

        # Flatten spatial dims → sequence of channel descriptors
        e = enc.flatten(2).transpose(1, 2)   # [B, HW, C]
        d = dec.flatten(2).transpose(1, 2)   # [B, HW, C]

        q = self.q(d)
        k = self.k(e)
        v = self.v(e)

        attn   = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out    = (attn @ v).transpose(1, 2).reshape(B, C, H, W)

        fused  = self.fuse(torch.cat([out, dec], dim=1))
        return fused
