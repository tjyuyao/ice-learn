from typing import List, Optional, Type, Union

import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from ice.llutil.argparser import as_list

ice.make_configurable(nn.Conv2d, nn.BatchNorm2d, nn.Sequential)

Conv3x3 = nn.Conv2d(kernel_size=3, padding=1, bias=False)
Conv1x1 = nn.Conv2d(kernel_size=1, bias=False)


@ice.configurable
class ResBlock(nn.Module):

    def __init__(
        self,
        branch: nn.Module,
        inplanes: int,
        planes: int,
        stride: int = 1,
        expansion:int = 1,
        identity: Optional[nn.Module]=None,
        checkpoint_enabled=False,
    ) -> None:
        super().__init__()
        self.branch = branch
        self.ckpt_on = checkpoint_enabled
        self.planes = planes
        self.expansion = expansion
        if identity is None:
            out_planes = self.out_channels
            if stride != 1 or inplanes != out_planes:
                identity = nn.Sequential(
                    Conv1x1(inplanes, out_planes, stride=stride),
                    nn.BatchNorm2d(out_planes)
                )
            else:
                identity = nn.Identity()
        self.identity = identity

    def forward(self, x: torch.Tensor):

        def _inner_forward(x):
            identity = self.identity(x)
            out = self.branch(x)
            out += identity
            return out

        if self.ckpt_on and x.requires_grad:
            out = ckpt.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return F.relu(out, inplace=True)

    @property
    def out_channels(self):
        return self.planes * self.expansion


@ice.configurable
class BasicResBlock(ResBlock):

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        expansion=1,
        identity: Optional[nn.Module]=None,
        checkpoint_enabled=False,
        ) -> None:

        branch = nn.Sequential(
            Conv3x3(inplanes, planes, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            Conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )

        super().__init__(branch, inplanes, planes, stride, expansion, identity, checkpoint_enabled)


@ice.configurable
class BottleNeckResBlock(ResBlock):

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        expansion=4,
        identity: Optional[nn.Module]=None,
        checkpoint_enabled=False,
        ) -> None:

        branch = nn.Sequential(
            Conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            Conv3x3(planes, planes, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            Conv1x1(planes, planes * expansion),
            nn.BatchNorm2d(planes * expansion),
        )

        super().__init__(branch, inplanes, planes, stride, expansion, identity, checkpoint_enabled)


@ice.configurable
class ResLayer(nn.Sequential):

    def __init__(self,
                 inplanes,
                 planes,
                 block_type,
                 num_blocks,
                 stride=1,
                 identity=None,
                 ) -> None:
        layers:List[ResBlock] = [block_type(inplanes, planes, stride, identity=identity)]
        for _ in range(1, num_blocks):
            layers.append(block_type(layers[-1].out_channels, planes))
        super().__init__(*layers)
        self._out_channels = layers[-1].out_channels
    
    @property
    def out_channels(self):
        return self._out_channels
    

@ice.configurable
class UpsampleConv1x1(nn.Module):
    # When align_corners=True, the output would be more aligned if input size is `x+1` and out size is `nx+1`
    def __init__(self,
        inplanes,
        planes,
        scale_factor,
        align_corners=False,
        mode='bilinear',
    ) -> None:
        super().__init__()
        self.upskwds = dict(
            scale_factor=scale_factor,
            align_corners=align_corners,
            mode=mode
        )
        self.conv1x1 = Conv1x1(inplanes, planes)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.bn(out)
        out = F.interpolate(out, **self.upskwds)
        return out


@ice.configurable
class TransformFunction(nn.Sequential):

    def __init__(self, inplanes, planes, r_in, r_out, upsampler) -> None:
        super().__init__()
        # `r` (0-indexed) means [2^(r+1) - 1] times downsample of raw size (after stem downsampling).
        if r_in == r_out:
            if inplanes == planes:
                super().__init__(nn.Identity())
            else:
                super().__init__(
                    Conv3x3(inplanes, planes),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(True),
                )
        elif r_in < r_out:
            num_layers = r_out - r_in
            conv_downsamples = []
            for _ in range(num_layers - 1):
                conv_downsamples += [
                    Conv3x3(inplanes, inplanes, stride=2),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(True),
                ]
            conv_downsamples += [
                Conv3x3(inplanes, planes, stride=2),
                nn.BatchNorm2d(planes),
                nn.ReLU(True),
            ]
            super().__init__(*conv_downsamples)
        else:
            super().__init__(
                upsampler(inplanes, planes, scale_factor=2**(r_in - r_out))
            )


@ice.configurable
class BranchingNewResolution(nn.Module):

    def __init__(self, inplanes:List[int], planes:List[int], upsampler):
        super().__init__()
        modules = []
        for r, (c_in, c_out) in enumerate(zip(as_list(inplanes), planes)):
            modules.append(TransformFunction(c_in, c_out, r, r, upsampler))
        for r_out in range(r+1, len(planes)):
            modules.append(TransformFunction(c_in, planes[r_out], r, r_out, upsampler))
        self.layers = nn.ModuleList(modules)

    def forward(self, branches):
        out = []
        for r, (branch, module) in enumerate(zip(as_list(branches), self.layers)):
            out.append(module(branch))
        for r_out in range(r+1, len(self.layers)):
            out.append(self.layers[r_out](branch))
        return out


@ice.configurable
class MultiResolutionFusion(nn.Module):

    def __init__(self, inplanes, planes, upsampler) -> None:
        super().__init__()
        fuse_layers = []
        for r_out, c_out in enumerate(as_list(planes)):
            fuse_layer = []
            for r_in, c_in in enumerate(as_list(inplanes)):
                fuse_layer.append(
                    TransformFunction(c_in, c_out, r_in, r_out, upsampler)
                )
            fuse_layers.append(nn.ModuleList(fuse_layer))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, branches):
        out = []
        for fuse_layers in self.fuse_layers:
            out.append(sum(layer(branch) for layer, branch in zip(fuse_layers, branches)))
        return out


@ice.configurable
class HRNetModule(nn.Module):

    def __init__(self, inplanes:List[int], planes:List[int], block_type, num_blocks, upsampler) -> None:
        super().__init__()
        self.parallel_convs = nn.ModuleList([
            ResLayer(c_in, c_in, block_type, num_blocks)
            for c_in in as_list(inplanes)
        ])
        inplanes = [res_layer.out_channels for res_layer in self.parallel_convs]
        self.fusion = MultiResolutionFusion(inplanes, inplanes, upsampler) \
            if len(inplanes) > 1 else nn.Identity()
        self.transition = BranchingNewResolution(inplanes, planes, upsampler) \
            if len(planes) != len(inplanes) else nn.Identity()
    
    def forward(self, branches):
        out = []
        for conv, branch in zip(self.parallel_convs, as_list(branches)):
            out.append(conv(branch))
        out = self.fusion(out)
        out = self.transition(out)
        return out


@ice.configurable
class HRNetStage(nn.Sequential):

    def __init__(self, inplanes:List[int], planes:List[int], block_type, num_blocks, num_modules, upsampler) -> None:
        super().__init__()
        modules = [
            HRNetModule(inplanes, inplanes, block_type, num_blocks, upsampler)
            for _ in range(num_modules - 1)
        ]
        modules.append(
            HRNetModule(inplanes, planes, block_type, num_blocks, upsampler)
        )
        super().__init__(*modules)


@ice.configurable
class StemDownsample(nn.Sequential):

    def __init__(self, inplanes, planes, r=2):
        modules = [
            Conv3x3(inplanes, planes, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        ]
        for _ in range(1, r):
            modules += [
                Conv3x3(planes, planes, stride=2),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
        super().__init__(*modules)


@ice.configurable
class HRNet18(nn.Sequential):

    out_channels = [18, 36, 72, 144]
    
    def __init__(self, upsampler=UpsampleConv1x1):

        _HRNetStage:Type[HRNetStage] = HRNetStage(upsampler=upsampler)
        NC = self.out_channels

        super().__init__(
            StemDownsample(3, 64, r=2),
            _HRNetStage([64],   NC[:2], BottleNeckResBlock, num_blocks=4, num_modules=1),
            _HRNetStage(NC[:2], NC[:3],      BasicResBlock, num_blocks=4, num_modules=1),
            _HRNetStage(NC[:3], NC[:4],      BasicResBlock, num_blocks=4, num_modules=4),
            _HRNetStage(NC[:4], NC[:4],      BasicResBlock, num_blocks=4, num_modules=3),
        )