from typing import Type
import ice
import torch
import torch.nn as nn


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

    def forward(self, feat):
        feat = self.dropout(feat)
        out = self.conv_seg(feat)
        return out
