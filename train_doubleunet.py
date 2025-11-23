#!/usr/bin/env python
"""
DoubleU-Net Training Script for Kvasir-SEG Dataset
Standalone script to train only DoubleU-Net model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

from models import DoubleUNet

# ============================================================================
# DATASET
# ============================================================================

class KvasirSegDataset(Dataset):
    """Kvasir-SEG dataset loader with preprocessing and augmentation"""
    
    def __init__(self, img_dir, mask_dir, img_size=352, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform
        
        self.img_files = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        print(f"[Dataset] Found {len(self.img_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        
        mask = (mask > 127).astype(np.uint8) * 255
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            else:
                mask = mask.unsqueeze(0) / 255.0 if mask.ndim == 2 else mask / 255.0
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return image, mask


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEWithDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# METRICS
# ============================================================================

class SegmentationMetrics:
    @staticmethod
    def dice(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        inter = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        return dice.mean().item()
    
    @staticmethod
    def iou(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        inter = (pred_binary * target).sum(dim=(2, 3))
        union = (pred_binary + target - pred_binary * target).sum(dim=(2, 3))
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean().item()
    
    @staticmethod
    def accuracy(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        correct = (pred_binary == target).float().mean()
        return correct.item()


# ============================================================================
# AUGMENTATION
# ============================================================================

def get_augmentations(img_size=352):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.CLAHE(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# ============================================================================
# TRAINING
# ============================================================================

class DoubleUNetTrainer:
    def __init__(self, model, device, loss_fn, optimizer, lr_scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = GradScaler()
        
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.best_dice = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                preds = self.model(images)
                loss = self.loss_fn(preds, masks)
            
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            dice = SegmentationMetrics.dice(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        self.train_losses.append(avg_loss)
        self.train_dices.append(avg_dice)
        
        print(f'  [Train] Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}')
    
    def val_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                with autocast(device_type=self.device.type):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, masks)
                
                dice = SegmentationMetrics.dice(preds, masks)
                
                total_loss += loss.item()
                total_dice += dice
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        self.val_losses.append(avg_loss)
        self.val_dices.append(avg_dice)
        
        print(f'  [Val] Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}')
        
        # Early stopping and checkpointing
        if avg_dice > self.best_dice:
            self.best_dice = avg_dice
            self.patience_counter = 0
            self.save_checkpoint('checkpoints/doubleunet_best.pth')
            print(f'  ✓ Best model saved (Dice: {avg_dice:.4f})')
        else:
            self.patience_counter += 1
        
        if self.lr_scheduler:
            self.lr_scheduler.step(avg_loss)
        
        return self.patience_counter
    
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
        }, path)
    
    def plot_history(self, save_dir='results'):
        Path(save_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.train_losses, label='Train', linewidth=2)
        axes[0].plot(self.val_losses, label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('DoubleU-Net Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.train_dices, label='Train', linewidth=2)
        axes[1].plot(self.val_dices, label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('DoubleU-Net Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/doubleunet_history.png', dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_dir}/doubleunet_history.png')
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DoubleU-Net Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss', type=str, default='bce_dice', choices=['dice', 'bce', 'bce_dice'])
    args = parser.parse_args()
    
    # Setup
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 352
    
    print(f"\n{'='*80}")
    print(f"DoubleU-Net Training on Kvasir-SEG Dataset")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss: {args.loss}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = KvasirSegDataset('images', 'masks', img_size=IMG_SIZE)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_transform, val_transform = get_augmentations(IMG_SIZE)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = DoubleUNet(in_channels=3, out_channels=1).to(DEVICE)
    print(f"[Model] DoubleU-Net (Cascaded U-Net) initialized")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")
    
    # Loss and Optimizer
    if args.loss == 'dice':
        loss_fn = DiceLoss()
    elif args.loss == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    else:  # bce_dice
        loss_fn = BCEWithDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Trainer
    trainer = DoubleUNetTrainer(model, DEVICE, loss_fn, optimizer, lr_scheduler)
    
    # Training loop
    EARLY_STOPPING_PATIENCE = 25
    
    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        patience_counter = trainer.val_epoch(val_loader, epoch)
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n✓ Early stopping triggered (patience {EARLY_STOPPING_PATIENCE} reached)")
            break
    
    # Save history
    Path('results').mkdir(exist_ok=True)
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_dices': trainer.train_dices,
        'val_dices': trainer.val_dices,
    }
    with open('results/doubleunet_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved: results/doubleunet_history.json")
    
    # Plot history
    trainer.plot_history()
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best Validation Dice: {trainer.best_dice:.4f}")
    print(f"Best Model: checkpoints/doubleunet_best.pth")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
