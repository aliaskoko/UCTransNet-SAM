import torch
import torch.nn as nn
import timm

class SAMAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class ViTEncoderWithAdapters(nn.Module):
    """
    ViT-B/16 backbone (timm) with SAM-style adapter inserted
    after every attention block. Returns 4 feature maps at
    resolutions [H/4, H/8, H/16, H/32].
    """
    def __init__(self, adapter_hidden_dim=64, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            features_only=False
        )
        embed_dim = self.vit.embed_dim  # 768 for ViT-B

        # One adapter per transformer block (12 blocks in ViT-B)
        self.adapters = nn.ModuleList([
            SAMAdapter(embed_dim, adapter_hidden_dim)
            for _ in range(len(self.vit.blocks))
        ])

        # Project token sequence → spatial feature maps at 4 scales
        self.proj1 = nn.Sequential(nn.Linear(embed_dim, 64),  nn.ReLU())
        self.proj2 = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU())
        self.proj3 = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU())
        self.proj4 = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)          # [B, 196, 768]
        x = self.vit._pos_embed(x)

        features = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            x = self.adapters[i](x)
            if i in [2, 5, 8, 11]:           # capture 4 intermediate scales
                features.append(x)

        def reshape(tokens, proj, size):
            out = proj(tokens[:, 1:])         # drop CLS token
            out = out.transpose(1, 2).reshape(B, -1, size, size)
            return out

        f1 = reshape(features[0], self.proj1, 56)   # 56×56×64
        f2 = reshape(features[1], self.proj2, 28)   # 28×28×128
        f3 = reshape(features[2], self.proj3, 14)   # 14×14×256
        f4 = reshape(features[3], self.proj4, 7)    #  7×7×512
        return f1, f2, f3, f4
