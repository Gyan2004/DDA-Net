#!/usr/bin/env python
"""
Complete Medical Image Segmentation Pipeline for Kvasir-SEG Dataset
Implements: U-Net, DoubleU-Net, DDANet with full training infrastructure
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import warnings
warnings.filterwarnings('ignore')

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
            # Transform already returns tensors via ToTensorV2
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            else:
                mask = mask.unsqueeze(0) / 255.0 if mask.ndim == 2 else mask / 255.0
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return image, mask


# ============================================================================
# MODELS
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov1 = DoubleConv(1024 + 512, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov2 = DoubleConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov3 = DoubleConv(256 + 128, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov4 = DoubleConv(128 + 64, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.cov1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.cov2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.cov3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.cov4(x)
        
        return self.outc(x)


class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.unet1 = UNet(in_channels, out_channels)
        self.unet2 = UNet(in_channels + out_channels, out_channels)
    
    def forward(self, x):
        out1 = self.unet1(x)
        out1_sigmoid = torch.sigmoid(out1)
        x_cat = torch.cat([x, out1_sigmoid], dim=1)
        out2 = self.unet2(x_cat)
        return out2


class DDANet(nn.Module):
    """Dense Dual Attention Network"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Attention blocks
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1),
            nn.Sigmoid()
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov1 = DoubleConv(1024 + 512, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov2 = DoubleConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov3 = DoubleConv(256 + 128, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov4 = DoubleConv(128 + 64, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dual attention
        spatial = self.spatial_att(x5)
        channel = self.channel_att(x5)
        x5 = x5 * spatial * channel
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.cov1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.cov2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.cov3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.cov4(x)
        
        return self.outc(x)


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


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky.mean()


# ============================================================================
# METRICS
# ============================================================================

class SegmentationMetrics:
    @staticmethod
    def dice(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        inter = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        return (2.0 * inter + 1e-6) / (union + 1e-6)
    
    @staticmethod
    def iou(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        inter = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - inter
        return (inter + 1e-6) / (union + 1e-6)
    
    @staticmethod
    def accuracy(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        correct = (pred_binary == target).sum()
        return correct.float() / target.numel()
    
    @staticmethod
    def precision(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        return (tp + 1e-6) / (tp + fp + 1e-6)
    
    @staticmethod
    def recall(pred, target, threshold=0.5):
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        tp = (pred_binary * target).sum()
        fn = ((1 - pred_binary) * target).sum()
        return (tp + 1e-6) / (tp + fn + 1e-6)


# ============================================================================
# TRAINING
# ============================================================================

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device, loss_fn, 
                 optimizer, scheduler, model_name='model', checkpoint_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
        self.best_val_dice = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(self.train_loader, desc=f'{self.model_name} Train')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu'):
                preds = self.model(images)
                loss = self.loss_fn(preds, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_dice += SegmentationMetrics.dice(preds, masks).item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        self.history['train_dice'].append(avg_dice)
        
        return avg_loss, avg_dice
    
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(self.val_loader, desc=f'{self.model_name} Val')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            with autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu'):
                preds = self.model(images)
                loss = self.loss_fn(preds, masks)
            
            total_loss += loss.item()
            total_dice += SegmentationMetrics.dice(preds, masks).item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        self.history['val_dice'].append(avg_dice)
        
        return avg_loss, avg_dice
    
    def save_checkpoint(self, epoch, is_best=False):
        suffix = 'best' if is_best else f'epoch_{epoch}'
        path = self.checkpoint_dir / f'{self.model_name}_{suffix}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_dice': self.best_val_dice,
        }, path)
        return path
    
    def train(self, epochs=100):
        print(f'\n{"="*70}')
        print(f'Training {self.model_name}')
        print(f'{"="*70}\n')
        
        for epoch in range(1, epochs + 1):
            train_loss, train_dice = self.train_epoch()
            val_loss, val_dice = self.val_epoch()
            
            print(f'Epoch {epoch}/{epochs} | TL: {train_loss:.4f} VL: {val_loss:.4f} | TD: {train_dice:.4f} VD: {val_dice:.4f}')
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                path = self.save_checkpoint(epoch, is_best=True)
                print(f'  ✓ Best model saved (Dice: {val_dice:.4f})')
            else:
                self.patience_counter += 1
                if self.patience_counter % 10 == 0:
                    self.save_checkpoint(epoch)
                
                if self.patience_counter >= 25:
                    print(f'✓ Early stopping at epoch {epoch}')
                    break
        
        print(f'\n✓ Training complete. Best Val Dice: {self.best_val_dice:.4f}\n')


# ============================================================================
# UTILITIES
# ============================================================================

def get_augmentations(img_size=352):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.CLAHE(p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)
    
    return train_transform, val_transform


def create_dataloaders(img_dir='images', mask_dir='masks', img_size=352, batch_size=8):
    dataset = KvasirSegDataset(img_dir, mask_dir, img_size=img_size)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_transform, val_transform = get_augmentations(img_size)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train segmentation models')
    parser.add_argument('--model', type=str, default='all', choices=['unet', 'doubleunet', 'ddanet', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='bce_dice', choices=['bce', 'dice', 'bce_dice', 'focal_tversky'])
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)
    
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'bce_dice':
        criterion = BCEWithDiceLoss()
    else:
        criterion = FocalTverskyLoss()
    
    models = {
        'unet': UNet(3, 1),
        'doubleunet': DoubleUNet(3, 1),
        'ddanet': DDANet(3, 1),
    }
    
    models_to_train = ['unet', 'doubleunet', 'ddanet'] if args.model == 'all' else [args.model]
    
    for model_name in models_to_train:
        model = models[model_name].to(device)
        
        params = sum(p.numel() for p in model.parameters())
        print(f'{model_name.upper()}: {params:,} parameters')
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        trainer = SegmentationTrainer(model, train_loader, val_loader, device, criterion, 
                                     optimizer, scheduler, model_name=model_name)
        
        trainer.train(epochs=args.epochs)
        
        with open(f'history_{model_name}.json', 'w') as f:
            json.dump(trainer.history, f)
