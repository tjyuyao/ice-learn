from typing import Type
import ice
import torch
import torch.nn as nn
from .upcatconv1x1 import UpCatConv1x1
from .local_attn_2d import local_attn_2d


@ice.configurable
class DensePrediction(nn.Module):
    """Classify each pixel."""

    def __init__(
        self,
        inplanes,
        num_classes,
        dropout_ratio=-1,
    ):
        super().__init__()
        self.conv_seg = nn.Conv2d(inplanes, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

    def forward(self, feat, _ = None):
        feat = self.dropout(feat)
        out = self.conv_seg(feat)
        return out


class LAHead15b(nn.Module):
    
    def __init__(self, in_coarser_channels, in_finer_channels, out_channels, hidden_channels, bias=True, kernel_size=3, dilation=1) -> None:
        super().__init__()
        self.conv_xq = nn.Sequential(nn.Conv2d(3, hidden_channels, 1), nn.ReLU(True), nn.Conv2d(hidden_channels, hidden_channels, 1))
        self.conv_xk = nn.Sequential(nn.Conv2d(3, hidden_channels, 1), nn.ReLU(True), nn.Conv2d(hidden_channels, hidden_channels, 1))
        self.upconv_xv = UpCatConv1x1(in_coarser_channels, in_finer_channels, out_channels, bias=bias)
        self.local_attn_2d_kwds = dict(kernel_size=kernel_size, dilation=dilation)
    
    def forward(self, coarse_feature, raw_image):
        xv = self.upconv_xv.forward(coarse_feature, raw_image)
        x = local_attn_2d(self.conv_xq(raw_image), self.conv_xk(raw_image), xv, **self.local_attn_2d_kwds)
        return x