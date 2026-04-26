import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml, argparse
from models import UCTransNetSAM
from data.preprocessing import SynapseDataset
from data.augmentation import MedicalAugmentation
from utils.losses import CombinedLoss
from utils.metrics import dice_score
from torch.cuda.amp import GradScaler, autocast

def train(cfg_path='config.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = UCTransNetSAM(num_classes=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['training']['epochs'])
    criterion = CombinedLoss()
    scaler    = GradScaler()

    train_ds  = SynapseDataset('data/synapse', split='train', transform=MedicalAugmentation())
    val_ds    = SynapseDataset('data/synapse', split='val')
    train_dl  = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,  num_workers=4)
    val_dl    = DataLoader(val_ds,   batch_size=1,                             shuffle=False, num_workers=2)

    best_dice, patience_count = 0, 0

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_dice = []
        with torch.no_grad():
            for images, labels in val_dl:
                images = images.to(device)
                preds  = model(images).argmax(dim=1).cpu().numpy()
                val_dice.append(dice_score(preds[0], labels[0].numpy(), num_classes=9))

        mean_dice = sum(val_dice) / len(val_dice)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Val DSC: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_count = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"  → Saved best model (DSC: {best_dice:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg['training']['patience']:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    train()
