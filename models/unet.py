"""
U-Net Model for Medical Image Segmentation
"""

import torch
import torch.nn as nn
from .blocks import DoubleConv


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.
    
    Architecture:
    - 4-level encoder with skip connections
    - 4-level decoder with upsampling
    - Skip connections concatenate encoder features to decoder
    
    Parameters:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
    
    Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov1 = DoubleConv(1024 + 512, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov2 = DoubleConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov3 = DoubleConv(256 + 128, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cov4 = DoubleConv(128 + 64, 64)
        
        # Output
        self.outc = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output segmentation map of shape (B, out_channels, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
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
