import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

class SynapseDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.samples   = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.npz')
        ])
        # 18 train / 6 val / 6 test split
        if split == 'train':
            self.samples = self.samples[:18]
        elif split == 'val':
            self.samples = self.samples[18:24]
        else:
            self.samples = self.samples[24:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data  = np.load(os.path.join(self.data_dir, self.samples[idx]))
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)

        # HU clip + z-score normalisation
        image = np.clip(image, -125, 275)
        image = (image - image.mean()) / (image.std() + 1e-8)

        image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        label = torch.from_numpy(label)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
