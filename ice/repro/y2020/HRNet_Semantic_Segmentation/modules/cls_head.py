from typing import List, Type

import ice
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import BottleNeckResBlock, ParallelConv, TransformFunction

ice.make_configurable(nn.BatchNorm2d)


@ice.configurable
class ClassificationHead(nn.Module):

    def __init__(
        self,
        inplanes: List[int],
        num_classes: int = 1000,
        planes: List[int] = [32, 64, 128, 256],
        block_type=BottleNeckResBlock,
        norm_cfg: Type[nn.BatchNorm2d] = nn.BatchNorm2d(momentum=0.1),
        expansion: int = 2,
    ) -> None:
        super().__init__()

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024 (planes * block_type.expansion)
        self.incre_modules = ParallelConv(inplanes, planes, block_type, 1)
        head_channels = self.incre_modules.out_channels

        # downsampling modules
        downsamp_modules = []
        for r in range(len(inplanes) - 1):
            downsamp_modules.append(
                TransformFunction(head_channels[r],
                                  head_channels[r + 1],
                                  r,
                                  r + 1,
                                  upsampler=None))
        self.downsamp_modules = nn.ModuleList(downsamp_modules)

        # Feature Head
        feat_channels = head_channels[-1] * expansion  # 2048 = 1024 * 2
        self.feat_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[-1],
                out_channels=feat_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ), norm_cfg(feat_channels), nn.ReLU(inplace=True))

        # Classification Head
        self.classifier = nn.Linear(feat_channels, num_classes)

    def forward(self, branches):

        incre = self.incre_modules(branches)

        for i, downsamp in enumerate(self.downsamp_modules):
            incre[i + 1] = incre[i + 1] + downsamp(incre[i])

        feat: torch.Tensor = self.feat_layer(incre[-1])

        if torch._C._get_tracing_state():
            feat = feat.flatten(start_dim=2).mean(dim=2)
        else:
            feat = F.avg_pool2d(feat, kernel_size=feat.size()[2:]).view(
                feat.size(0), -1)

        out = self.classifier(feat)

        return out
