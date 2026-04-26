import torch
import torchvision.transforms.functional as TF
import random

class MedicalAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        if random.random() < self.p:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)

        if random.random() < self.p:
            gamma = random.uniform(0.7, 1.5)
            image = TF.adjust_gamma(image, gamma)

        return image, mask
