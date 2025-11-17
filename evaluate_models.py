#!/usr/bin/env python
"""Evaluation and visualization script for segmentation models"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from train import KvasirSegDataset, SegmentationMetrics, UNet, DoubleUNet, DDANet, get_augmentations
from torch.utils.data import DataLoader, random_split


def load_model(model_name, checkpoint_path, device):
    """Load model from checkpoint"""
    models = {
        'unet': UNet(3, 1),
        'doubleunet': DoubleUNet(3, 1),
        'ddanet': DDANet(3, 1),
    }
    
    model = models[model_name].to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded {model_name} from {Path(checkpoint_path).name}")
        return model, checkpoint.get('history', {})
    except:
        print(f"⚠ Using untrained {model_name} (no checkpoint found)")
        return model, {}


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    metrics = {'dice': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            
            metrics['dice'].append(SegmentationMetrics.dice(preds, masks).item())
            metrics['iou'].append(SegmentationMetrics.iou(preds, masks).item())
            metrics['accuracy'].append(SegmentationMetrics.accuracy(preds, masks).item())
            metrics['precision'].append(SegmentationMetrics.precision(preds, masks).item())
            metrics['recall'].append(SegmentationMetrics.recall(preds, masks).item())
    
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def plot_training_history(history, model_name, output_dir='results'):
    """Plot training curves"""
    Path(output_dir).mkdir(exist_ok=True)
    
    if not history or 'train_loss' not in history:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history['train_dice'], label='Train', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val', linewidth=2)
    axes[1].set_title(f'{model_name} - Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    path = f"{output_dir}/{model_name}_history.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def visualize_predictions(model, images, masks, model_name, device, output_dir='results'):
    """Visualize sample predictions"""
    Path(output_dir).mkdir(exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device))
    
    n_samples = min(3, images.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Denormalize image
        img = images[i].cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        pred_prob = torch.sigmoid(preds[i]).squeeze().cpu().numpy()
        axes[i, 2].imshow(pred_prob, cmap='gray')
        axes[i, 2].set_title('Predicted Prob')
        axes[i, 2].axis('off')
        
        pred_binary = (pred_prob > 0.5).astype(np.uint8)
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title('Predicted Mask')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    path = f"{output_dir}/{model_name}_predictions.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def create_comparison_table(results, output_dir='results'):
    """Create comparison table"""
    Path(output_dir).mkdir(exist_ok=True)
    
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name.upper(),
            'Dice': f"{metrics['dice'][0]:.4f}±{metrics['dice'][1]:.4f}",
            'IoU': f"{metrics['iou'][0]:.4f}±{metrics['iou'][1]:.4f}",
            'Accuracy': f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}",
            'Precision': f"{metrics['precision'][0]:.4f}±{metrics['precision'][1]:.4f}",
            'Recall': f"{metrics['recall'][0]:.4f}±{metrics['recall'][1]:.4f}",
        })
    
    df = pd.DataFrame(data)
    csv_path = f"{output_dir}/comparison.csv"
    df.to_csv(csv_path, index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Performance Comparison', fontsize=14, weight='bold', pad=20)
    table_path = f"{output_dir}/comparison.png"
    plt.savefig(table_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return csv_path, table_path


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load dataset
    print("[1/3] Loading dataset...")
    dataset = KvasirSegDataset('images', 'masks', img_size=352)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    _, val_transform = get_augmentations(352)
    val_dataset.dataset.transform = val_transform
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Get sample batch
    sample_images, sample_masks = next(iter(val_loader))
    print(f"✓ Validation set: {len(val_dataset)} samples\n")
    
    # Evaluate models
    print("[2/3] Evaluating models...")
    models_config = [
        ('unet', 'checkpoints/unet_best.pth'),
        ('doubleunet', 'checkpoints/doubleunet_best.pth'),
        ('ddanet', 'checkpoints/ddanet_best.pth'),
    ]
    
    results = {}
    for model_name, checkpoint_path in models_config:
        print(f"\n  Evaluating {model_name}...")
        model, history = load_model(model_name, checkpoint_path, device)
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, device)
        results[model_name] = metrics
        
        for metric, (mean, std) in metrics.items():
            print(f"    {metric:12s}: {mean:.4f} ± {std:.4f}")
        
        # Visualize
        plot_training_history(history, model_name)
        visualize_predictions(model, sample_images, sample_masks, model_name, device)
    
    # Create comparison
    print("\n[3/3] Creating comparison tables...")
    csv_path, table_path = create_comparison_table(results)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to 'results/' directory")
    print(f"✓ Checkpoints saved to 'checkpoints/' directory")
