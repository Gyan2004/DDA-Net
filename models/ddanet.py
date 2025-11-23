"""
DDANet (Dense Dual Attention Network) for Medical Image Segmentation
"""

import torch
import torch.nn as nn
from .blocks import DoubleConv


class DDANet(nn.Module):
    """
    Dense Dual Attention Network for medical image segmentation.
    
    Architecture:
    - U-Net backbone with encoder-decoder structure
    - Dual attention mechanism at bottleneck:
      * Spatial attention: learns which regions are important
      * Channel attention: learns which feature channels are important
    - Combines both attention maps multiplicatively for fine-grained focus
    
    Parameters:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
    
    Features:
    - Spatial and channel attention at bottleneck improves feature representation
    - Better gradient flow for deeper networks
    - Adaptive feature recalibration
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Spatial Attention Block
        # Generates attention map based on spatial locations
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1),
            nn.Sigmoid()
        )
        
        # Channel Attention Block
        # Generates attention map based on channel importance
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1),
            nn.Sigmoid()
        )
        
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
        Forward pass with dual attention mechanism.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Segmentation map of shape (B, out_channels, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dual attention at bottleneck
        spatial = self.spatial_att(x5)      # Spatial attention map
        channel = self.channel_att(x5)      # Channel attention map
        x5 = x5 * spatial * channel         # Element-wise multiplication
        
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
