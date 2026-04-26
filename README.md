# UCTransNet-SAM

**Hybrid CNN-Transformer Architecture with SAM Domain Adapters for Multi-Organ Medical Image Segmentation**

> Computer Vision Research Assignment 3 | April 2025

## Overview
UCTransNet-SAM combines a ViT-B/16 encoder with SAM-style lightweight adapter layers and UCTransNet-style Channel Transformer (CTrans) skip connections for pixel-wise medical image segmentation across CT, MRI, and endoscopic modalities.

**Results on Synapse Multi-Organ CT:**
| Method | mDSC (%) | mHD95 (mm) | mIoU (%) |
|--------|----------|------------|---------|
| Swin-UNet | 79.13 | 21.55 | 72.56 |
| UCTransNet | 81.01 | 19.73 | 75.34 |
| **UCTransNet-SAM (Ours)** | **85.47** | **14.23** | **79.81** |

## Datasets
- [Synapse Multi-Organ CT](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- [ACDC Cardiac MRI](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067)
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)

## Installation
```bash
git clone https://github.com/yourusername/UCTransNet-SAM
cd UCTransNet-SAM
pip install -r requirements.txt
```

## Training
```bash
python train.py --dataset synapse --epochs 150 --batch_size 16
```

## Inference
```bash
python inference.py --input path/to/volume.nii.gz --output results/
```

## Citation
If you use this work, please cite:
Zunaira Hameed, Abdullah Zulfiqar, "UCTransNet-SAM: A Hybrid CNN-Transformer Architecture...", CV Assignment 3, April 2025.
