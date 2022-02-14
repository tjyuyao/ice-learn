from typing import Type
import ice
import torch.nn as nn


ice.make_configurable(nn.Conv2d, nn.BatchNorm2d)


@ice.configurable
class ConvModule(nn.Module):

    
    def __init__(
        self,
        inplanes, 
        planes,
        conv_cfg:Type[nn.Conv2d]=nn.Conv2d(),
        norm_cfg:Type[nn.BatchNorm2d]=nn.BatchNorm2d(),
        act_cfg=nn.ReLU(True),
        order=('conv', 'norm', 'act'),
    ) -> None:
        super().__init__()

        if ice.frozen(conv_cfg):
            self.conv = conv_cfg
        else:
            self.conv = conv_cfg(inplanes, planes)

        if norm_cfg is None:
            self.norm = nn.Identity()
        elif ice.frozen(norm_cfg):
            self.norm = norm_cfg
        elif "num_features" in norm_cfg:
            self.norm = norm_cfg(num_features=planes)
        elif "num_channels" in norm_cfg:
            self.norm = norm_cfg(num_channels=planes)
        else:
            self.norm = norm_cfg().freeze()
        
        if act_cfg is None:
            self.act = nn.Identity()
        elif ice.frozen(act_cfg):
            self.act = act_cfg
        elif "inplace" in act_cfg:
            self.act = act_cfg(inplace=True)
        else:
            self.act = act_cfg().freeze()
        
        self.order = order
    
    def forward(self, x):
        for o in self.order:
            x = getattr(self, o)(x)
        return x