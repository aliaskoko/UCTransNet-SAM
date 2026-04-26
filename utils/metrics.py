import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, target, num_classes):
    scores = []
    for c in range(1, num_classes):        # skip background
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        if p.sum() + t.sum() == 0:
            scores.append(1.0)             # both empty → perfect
            continue
        scores.append(2 * intersection / (p.sum() + t.sum() + 1e-5))
    return np.mean(scores)

def hausdorff95(pred, target, num_classes):
    hds = []
    for c in range(1, num_classes):
        p = np.argwhere(pred == c)
        t = np.argwhere(target == c)
        if len(p) == 0 or len(t) == 0:
            continue
        d1 = directed_hausdorff(p, t)[0]
        d2 = directed_hausdorff(t, p)[0]
        hds.append(max(d1, d2))
    return np.percentile(hds, 95) if hds else 0.0

def iou_score(pred, target, num_classes):
    scores = []
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        scores.append(inter / (union + 1e-5) if union > 0 else 1.0)
    return np.mean(scores)
