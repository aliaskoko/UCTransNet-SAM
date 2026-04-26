import torch
import numpy as np
import SimpleITK as sitk
import argparse, os
from models import UCTransNetSAM
from utils.visualize import plot_results

def run_inference(input_path, output_dir, checkpoint='checkpoints/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UCTransNetSAM(num_classes=9, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img    = sitk.GetArrayFromImage(sitk.ReadImage(input_path)).astype(np.float32)
    img    = np.clip(img, -125, 275)
    img    = (img - img.mean()) / (img.std() + 1e-8)

    preds  = []
    with torch.no_grad():
        for slc in img:
            t = torch.from_numpy(slc).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)
            t = torch.nn.functional.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
            p = model(t).argmax(dim=1).squeeze().cpu().numpy()
            preds.append(p)

    pred_vol = np.stack(preds, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    out_img  = sitk.GetImageFromArray(pred_vol.astype(np.uint8))
    sitk.WriteImage(out_img, os.path.join(output_dir, 'segmentation.nii.gz'))
    print(f"Saved to {output_dir}/segmentation.nii.gz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', default='results/')
    parser.add_argument('--ckpt',   default='checkpoints/best_model.pth')
    args = parser.parse_args()
    run_inference(args.input, args.output, args.ckpt)
