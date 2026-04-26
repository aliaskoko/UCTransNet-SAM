import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs   = torch.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.shape[1])
        targets = targets.permute(0, 3, 1, 2).float()
        num     = 2 * (probs * targets).sum(dim=(2, 3))
        den     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.eps
        return 1 - (num / den).mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice       = DiceLoss()
        self.ce         = nn.CrossEntropyLoss()
        self.dice_w     = dice_weight
        self.ce_w       = ce_weight

    def forward(self, logits, targets):
        return self.dice_w * self.dice(logits, targets) + \
               self.ce_w   * self.ce(logits, targets)
