"""
DoubleU-Net Model for Medical Image Segmentation
"""

import torch
import torch.nn as nn
from .unet import UNet


class DoubleUNet(nn.Module):
    """
    DoubleU-Net (Cascaded U-Net) for medical image segmentation.
    
    Architecture:
    - First U-Net generates coarse segmentation from input image
    - Second U-Net refines the segmentation by concatenating:
      * Original input
      * Output from first U-Net (after sigmoid)
    - Cascaded approach allows refinement and better boundary detection
    
    Parameters:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
    
    Paper: DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation
    https://arxiv.org/abs/2006.04868
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.unet1 = UNet(in_channels, out_channels)
        self.unet2 = UNet(in_channels + out_channels, out_channels)
    
    def forward(self, x):
        """
        Forward pass through cascaded U-Nets.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Refined segmentation map of shape (B, out_channels, H, W)
        """
        # First U-Net generates initial segmentation
        out1 = self.unet1(x)
        out1_sigmoid = torch.sigmoid(out1)
        
        # Concatenate original input with first U-Net output
        x_cat = torch.cat([x, out1_sigmoid], dim=1)
        
        # Second U-Net refines the segmentation
        out2 = self.unet2(x_cat)
        
        return out2
