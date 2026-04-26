import torch.nn as nn
from .encoder import ViTEncoderWithAdapters
from .decoder import Decoder

class UCTransNetSAM(nn.Module):
    def __init__(self, num_classes=9, adapter_hidden_dim=64, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoderWithAdapters(
            adapter_hidden_dim=adapter_hidden_dim,
            pretrained=pretrained
        )
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        return self.decoder(f1, f2, f3, f4)
