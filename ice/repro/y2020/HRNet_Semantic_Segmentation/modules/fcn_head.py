from typing import Type
import ice
import torch
import torch.nn as nn


ice.make_configurable(nn.Conv2d)


@ice.configurable
class FCNHead(nn.Sequential):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        inplanes,
        planes,
        num_convs=2,
        kernel_size=3,
        dilation=1,
        norm_cfg=nn.BatchNorm2d(),
    ):

        padding = (kernel_size // 2) * dilation
        ConvModule: Type[nn.Conv2d] = nn.Conv2d(kernel_size=kernel_size,
                                                padding=padding,
                                                dilation=dilation)

        if num_convs == 0:
            assert planes == inplanes
            super().__init__(nn.Identity())
        else:
            convs = []
            convs.append(
                nn.Sequential(ConvModule(inplanes, planes),
                              norm_cfg(planes), nn.ReLU(True)))
            for _ in range(1, num_convs):
                convs.append(
                    nn.Sequential(ConvModule(inplanes, planes),
                                  norm_cfg(planes), nn.ReLU(True)))
            super().__init__(*convs)
