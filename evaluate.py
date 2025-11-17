import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
import pickle
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import from train.py
from train import (
    KvasirSegDataset, SegmentationMetrics, UNet, DoubleUNet, DDANet, AttentionUNet
)

# ============================================================================
# 1. COMPREHENSIVE EVALUATION
# ============================================================================

class SegmentationEvaluator:
    def __init__(self, model, val_loader, device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, threshold=0.5):
        all_dice = []
        all_iou = []
        all_acc = []
        all_prec = []
        all_rec = []
        all_preds = []
        all_targets = []
        
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            preds = self.model(images)
            
            all_dice.append(SegmentationMetrics.dice(preds, masks, threshold))
            all_iou.append(SegmentationMetrics.iou(preds, masks, threshold))
            all_acc.append(SegmentationMetrics.accuracy(preds, masks, threshold))
            all_prec.append(SegmentationMetrics.precision(preds, masks, threshold))
            all_rec.append(SegmentationMetrics.recall(preds, masks, threshold))
            all_prec.append(SegmentationMetrics.precision(preds, masks, threshold))
            all_rec.append(SegmentationMetrics.recall(preds, masks, threshold))
            
            all_preds.append((torch.sigmoid(preds) > threshold).float().cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        metrics = {
            'Dice': np.mean(all_dice),
            'IoU': np.mean(all_iou),
            'Accuracy': np.mean(all_acc),
            'Precision': np.mean(all_prec),
            'Recall': np.mean(all_rec),
        }
        
        return metrics, all_preds, all_targets

# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(results, save_dir='results'):
    """Plot training vs validation loss and Dice"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 4*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, data) in enumerate(results.items()):
        trainer = data['trainer']
        
        # Loss
        axes[idx, 0].plot(trainer.train_losses, label='Train', linewidth=2)
        axes[idx, 0].plot(trainer.val_losses, label='Validation', linewidth=2)
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].set_title(f'{model_name} - Training Loss')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Dice
        axes[idx, 1].plot(trainer.train_dices, label='Train', linewidth=2)
        axes[idx, 1].plot(trainer.val_dices, label='Validation', linewidth=2)
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('Dice Score')
        axes[idx, 1].set_title(f'{model_name} - Dice Score')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_dir}/training_curves.png')
    plt.close()

def plot_sample_predictions(model, dataset, device, model_name, num_samples=4, save_dir='results'):
    """Show side-by-side predictions: original, ground truth, prediction"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Get validation dataset
    dataset.transform = None  # Disable transform temporarily
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Handle raw numpy arrays
            if isinstance(image, np.ndarray):
                image_display = (image * 255).astype(np.uint8)
                if image_display.shape[0] == 3:
                    image_display = np.transpose(image_display, (1, 2, 0))
                image_tensor = torch.from_numpy(image).unsqueeze(0)
            else:
                image_tensor = image.unsqueeze(0)
                image_display = image.permute(1, 2, 0).numpy()
                image_display = (image_display * 255).astype(np.uint8)
            
            if isinstance(mask, np.ndarray):
                mask_display = mask.squeeze()
            else:
                mask_display = mask.squeeze().numpy()
            
            # Normalize image if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            image_tensor = image_tensor.to(device)
            
            # Predict
            pred = model(image_tensor)
            pred_mask = (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            # Display
            axes[row, 0].imshow(image_display)
            axes[row, 0].set_title('Original Image')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(mask_display, cmap='gray')
            axes[row, 1].set_title('Ground Truth Mask')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(pred_mask, cmap='gray')
            axes[row, 2].set_title('Predicted Mask')
            axes[row, 2].axis('off')
    
    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_predictions.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_dir}/{model_name}_predictions.png')
    plt.close()

def plot_metrics_comparison(metrics_dict, save_dir='results'):
    """Create comparison table and visualization"""
    Path(save_dir).mkdir(exist_ok=True)
    
    df = pd.DataFrame(metrics_dict).T
    
    # Print table
    print(f'\n{"="*80}')
    print(f'Model Performance Comparison (Validation Set)')
    print(f'{"="*80}')
    print(df.to_string())
    print(f'{"="*80}')
    
    # Save to CSV
    df.to_csv(f'{save_dir}/metrics_comparison.csv')
    print(f'✓ Saved: {save_dir}/metrics_comparison.csv')
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = df.columns.tolist()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_dir}/metrics_comparison.png')
    plt.close()

def plot_attention_maps(model, dataset, device, model_name, num_samples=3, save_dir='results'):
    """Visualize attention maps for DDANet"""
    if model_name not in ['DDANet', 'Attention U-Net']:
        return
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            if isinstance(image, np.ndarray):
                image_display = (image * 255).astype(np.uint8)
                if image_display.shape[0] == 3:
                    image_display = np.transpose(image_display, (1, 2, 0))
                image_tensor = torch.from_numpy(image).unsqueeze(0)
            else:
                image_tensor = image.unsqueeze(0)
                image_display = image.permute(1, 2, 0).numpy()
                image_display = (image_display * 255).astype(np.uint8)
            
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            image_tensor = image_tensor.to(device)
            
            # Predict
            pred = model(image_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            # Display
            axes[row, 0].imshow(image_display)
            axes[row, 0].set_title('Original Image')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(pred_mask, cmap='hot')
            axes[row, 1].set_title('Attention Map')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(image_display, alpha=0.6)
            axes[row, 2].imshow(pred_mask, cmap='hot', alpha=0.4)
            axes[row, 2].set_title('Overlay')
            axes[row, 2].axis('off')
    
    plt.suptitle(f'{model_name} - Attention Visualization', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_attention.png', dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {save_dir}/{model_name}_attention.png')
    plt.close()

# ============================================================================
# 3. MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 352
    BATCH_SIZE = 8
    
    # Load results
    print('Loading training results...')
    with open('training_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Prepare validation dataset
    dataset = KvasirSegDataset('images', 'masks', img_size=IMG_SIZE)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # No transform for evaluation (raw images for visualization)
    val_dataset.dataset.transform = None
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Evaluate all models
    metrics_dict = {}
    Path('results').mkdir(exist_ok=True)
    
    for model_name, data in results.items():
        print(f'\nEvaluating {model_name}...')
        model = data['model'].to(DEVICE)
        
        evaluator = SegmentationEvaluator(model, val_loader, DEVICE)
        metrics, preds, targets = evaluator.evaluate()
        
        metrics_dict[model_name] = metrics
        print(f'  Dice: {metrics["Dice"]:.4f} | IoU: {metrics["IoU"]:.4f} | Accuracy: {metrics["Accuracy"]:.4f}')
        
        # Visualizations
        plot_sample_predictions(model, val_dataset.dataset, DEVICE, model_name, num_samples=4)
        plot_attention_maps(model, val_dataset.dataset, DEVICE, model_name, num_samples=3)
    
    # Comparison plots
    print(f'\nGenerating comparison visualizations...')
    plot_training_curves(results)
    plot_metrics_comparison(metrics_dict)
    
    # Summary
    best_model = max(metrics_dict.items(), key=lambda x: x[1]['Dice'])
    print(f'\n{"="*60}')
    print(f'BEST MODEL: {best_model[0]}')
    print(f'Dice Score: {best_model[1]["Dice"]:.4f}')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()
