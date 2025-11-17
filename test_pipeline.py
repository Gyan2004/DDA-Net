#!/usr/bin/env python
"""
Quick test script to verify the segmentation pipeline works correctly.
This trains a U-Net for just 2 epochs on a subset of data to validate:
- Dataset loading
- Model initialization
- Training loop
- Loss computation
- Inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from train.py
from train import KvasirSegDataset, UNet, DDANet, BCEWithDiceLoss, SegmentationMetrics

def test_pipeline():
    print("\n" + "="*60)
    print("SEGMENTATION PIPELINE TEST")
    print("="*60)
    
    # Configuration
    IMG_SIZE = 352
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Image Size: {IMG_SIZE}")
    print(f"✓ Batch Size: {BATCH_SIZE}")
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = KvasirSegDataset('images', 'masks', img_size=IMG_SIZE)
    print(f"✓ Loaded {len(dataset)} images")
    
    # Use subset for quick test (100 samples)
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    print(f"✓ Using subset of {len(subset)} samples for quick test")
    
    # Split
    val_size = int(0.2 * len(subset))
    train_size = len(subset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(subset, [train_size, val_size])
    
    # Step 2: Setup augmentations
    print("\n[2/6] Setting up augmentations...")
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
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
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)
    
    # Apply transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Step 3: Initialize models
    print("\n[3/6] Initializing models...")
    unet = UNet(3, 1).to(DEVICE)
    ddanet = DDANet(3, 1).to(DEVICE)
    
    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    ddanet_params = sum(p.numel() for p in ddanet.parameters() if p.requires_grad)
    
    print(f"✓ U-Net parameters: {unet_params:,}")
    print(f"✓ DDANet parameters: {ddanet_params:,}")
    
    # Step 4: Test forward pass
    print("\n[4/6] Testing forward pass...")
    model = unet
    model.eval()
    
    with torch.no_grad():
        sample_batch, sample_mask = next(iter(train_loader))
        sample_batch = sample_batch.to(DEVICE)
        output = model(sample_batch)
        
        print(f"✓ Input shape: {sample_batch.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected mask shape: {sample_mask.shape}")
        
        # Verify shapes match
        assert output.shape == sample_mask.shape, "Output shape mismatch!"
        print("✓ Shape validation passed")
    
    # Step 5: Test training step
    print("\n[5/6] Testing training loop (2 epochs)...")
    
    optimizer = optim.Adam(unet.parameters(), lr=1e-3)
    loss_fn = BCEWithDiceLoss(bce_weight=0.5, dice_weight=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    
    unet.train()
    for epoch in range(1, 3):
        total_loss = 0
        total_dice = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if 'cuda' in str(DEVICE) else 'cpu'):
                preds = unet(images)
                loss = loss_fn(preds, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_dice += SegmentationMetrics.dice(preds, masks)
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")
    
    # Step 6: Test validation loop
    print("\n[6/6] Testing validation loop...")
    unet.eval()
    
    with torch.no_grad():
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            with autocast(device_type='cuda' if 'cuda' in str(DEVICE) else 'cpu'):
                preds = unet(images)
                loss = loss_fn(preds, masks)
            
            val_loss += loss.item()
            val_dice += SegmentationMetrics.dice(preds, masks)
            val_iou += SegmentationMetrics.iou(preds, masks)
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        print(f"✓ Validation Loss: {val_loss:.4f}")
        print(f"✓ Validation Dice: {val_dice:.4f}")
        print(f"✓ Validation IoU: {val_iou:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nThe pipeline is working correctly!")
    print("You can now run: python train.py")
    print("\n")

if __name__ == '__main__':
    test_pipeline()
