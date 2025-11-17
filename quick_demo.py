#!/usr/bin/env python
"""
Quick start script - Train for just 5 epochs to demonstrate the pipeline
"""

import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import json

from train import (
    KvasirSegDataset, UNet, DoubleUNet, DDANet, AttentionUNet,
    BCEWithDiceLoss, SegmentationTrainer, get_augmentations
)

def train_quick_demo():
    """Quick training demo - 5 epochs on subset of data"""
    
    print("\n" + "="*70)
    print("MEDICAL IMAGE SEGMENTATION - QUICK START DEMO")
    print("="*70)
    print("This demo trains all models for 5 epochs to verify everything works")
    print("For full training, run: python train.py --epochs 100\n")
    
    # Configuration
    IMG_SIZE = 352
    BATCH_SIZE = 8
    EPOCHS = 5  # Quick demo
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS} (demo mode)")
    print(f"  Learning Rate: {LEARNING_RATE}\n")
    
    # Load dataset and create data loaders
    print("[1/4] Preparing dataset...")
    dataset = KvasirSegDataset('images', 'masks', img_size=IMG_SIZE)
    print(f"  Total samples: {len(dataset)}")
    
    # Use only 20% of data for quick demo
    demo_size = len(dataset) // 5
    val_size = int(0.2 * demo_size)
    train_size = demo_size - val_size
    
    # Sample from full dataset
    indices = list(range(demo_size))
    train_dataset, val_dataset = random_split(
        torch.utils.data.Subset(dataset, indices),
        [train_size, val_size]
    )
    
    # Apply augmentations
    train_transform, val_transform = get_augmentations(IMG_SIZE)
    train_dataset.dataset.dataset.transform = train_transform
    val_dataset.dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")
    
    # Train models
    models_config = [
        ('UNet', UNet(3, 1)),
        ('DoubleU-Net', DoubleUNet(3, 1)),
        ('DDANet', DDANet(3, 1)),
        ('Attention U-Net', AttentionUNet(3, 1)),
    ]
    
    print("[2/4] Initializing models...")
    for model_name, _ in models_config:
        print(f"  ✓ {model_name}")
    print()
    
    results = {}
    
    print("[3/4] Training models...\n")
    for model_name, model in models_config:
        model = model.to(DEVICE)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{model_name} ({params:,} parameters)")
        print("-" * 50)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        loss_fn = BCEWithDiceLoss(bce_weight=0.5, dice_weight=0.5)
        
        trainer = SegmentationTrainer(
            model, train_loader, val_loader, DEVICE,
            loss_fn, optimizer, scheduler, model_name=model_name
        )
        
        trainer.train(epochs=EPOCHS)
        
        results[model_name] = {
            'trainer': trainer,
            'model': model,
            'params': params
        }
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    print("\nResults Summary:")
    print("-" * 70)
    for model_name, info in results.items():
        trainer = info['trainer']
        best_dice = trainer.best_val_dice
        params = info['params']
        print(f"{model_name:20s} | Params: {params:>10,} | Best Val Dice: {best_dice:.4f}")
    
    print("\n✓ Quick demo completed successfully!")
    print("\nNext steps:")
    print("  1. Run full training: python train.py --epochs 100")
    print("  2. Evaluate results: python evaluate_models.py")
    print("  3. View documentation: cat COMPLETE_PIPELINE.md")
    print("\nFor more options, run: python train.py --help")
    print("="*70 + "\n")


if __name__ == '__main__':
    train_quick_demo()
