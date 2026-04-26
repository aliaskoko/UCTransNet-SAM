import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLORS = [
    [0,0,0],[255,0,0],[0,255,0],[0,0,255],
    [255,255,0],[0,255,255],[255,0,255],[128,0,0],[0,128,0]
]
LABELS = [
    'Background','Aorta','Gallbladder','Spleen',
    'L.Kidney','R.Kidney','Liver','Stomach','Pancreas'
]

def overlay_segmentation(image_slice, mask, alpha=0.45):
    rgb   = np.stack([image_slice]*3, axis=-1)
    rgb   = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
    color = np.zeros_like(rgb)
    for i, c in enumerate(COLORS):
        color[mask == i] = c
    blended = (rgb * (1 - alpha) + color * alpha).astype(np.uint8)
    return blended

def plot_results(image, gt, pred, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray');  axes[0].set_title('Input CT Slice')
    axes[1].imshow(overlay_segmentation(image, gt));   axes[1].set_title('Ground Truth')
    axes[2].imshow(overlay_segmentation(image, pred)); axes[2].set_title('Prediction')
    patches = [mpatches.Patch(color=[c/255 for c in COLORS[i]], label=LABELS[i])
               for i in range(len(LABELS))]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
