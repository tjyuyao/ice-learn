from typing import List
import torch

import torch.nn as nn
import torch.nn.functional as F
from ice.llutil.argparser import as_list


class ResizeConcat(nn.Module):

    def __init__(self, inplanes:List[int], target_branch_idx:int=0, align_corners=False, mode="bilinear") -> None:
        super().__init__()
        self.target_idx = target_branch_idx
        self.upskwds = dict(align_corners=align_corners, mode=mode)
        self.planes = sum(as_list(inplanes))

    def forward(self, branches):
        target_size = branches[self.target_idx].shape[-2:]
        out = [
            F.interpolate(branch, target_size, **self.upskwds)
            for branch in branches
        ]
        out = torch.cat(out, dim=1)
        return out
    
    @property
    def out_channels(self) -> int:
        return self.planes


class Select(nn.Module):

    def __init__(self, inplanes:List[int], target_branch_idx:int=0) -> None:
        super().__init__()
        self.target_idx = target_branch_idx
        self.planes = inplanes[target_branch_idx]

    def forward(self, branches):
        out = branches[self.target_idx]
        return out
    
    @property
    def out_channels(self) -> int:
        return self.planes


class MultipleSelect(nn.Module):

    def __init__(self, inplanes:List[int], target_branch_idx:List[int]=[0]) -> None:
        super().__init__()
        self.target_idx = as_list(target_branch_idx)
        self.planes = [inplanes[i] for i in self.target_idx]

    def forward(self, branches):
        out = [branches[i] for i in self.target_idx]
        return out
    
    @property
    def out_channels(self) -> List[int]:
        return self.planes
