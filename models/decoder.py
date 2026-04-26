import torch
import torch.nn as nn
import torch.nn.functional as F
from .ctrans import CTrans

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_heads=8):
        super().__init__()
        self.ctrans = CTrans(skip_ch, num_heads)
        self.conv   = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x    = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.ctrans(skip, x)
        x    = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # channels match encoder outputs [64, 128, 256, 512]
        self.block1 = DecoderBlock(512, 256, 256)
        self.block2 = DecoderBlock(256, 128, 128)
        self.block3 = DecoderBlock(128,  64,  64)
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, f1, f2, f3, f4):
        x = self.block1(f4, f3)   # 7→14
        x = self.block2(x,  f2)   # 14→28
        x = self.block3(x,  f1)   # 28→56
        x = self.up_final(x)      # 56→224
        return self.head(x)       # [B, num_classes, 224, 224]
