from typing import List, Type
import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neck import ResizeConcat
from .fcn_head import FCNHead
from .ocr_context import OCRContext
from .hat import DensePrediction


@ice.configurable
class FCNOCRHead(nn.Module):
    
    def __init__(
        self,
        inplanes: List[int],
        ocr_planes:int=256,
        num_classes=19,
        neck_cfg:Type[ResizeConcat]=ResizeConcat,
        fcn_head:Type[FCNHead]=FCNHead(num_convs=1, kernel_size=1),
        soft_region_pred:Type[DensePrediction]=DensePrediction(dropout_ratio=-1),
        final_pred:Type[DensePrediction]=DensePrediction(dropout_ratio=-1),
        norm_cfg=nn.BatchNorm2d(),
    ) -> None:
        super().__init__()

        self.neck = neck_cfg(inplanes)

        fcn_planes = self.neck.out_channels
        self.fcn_head = fcn_head(fcn_planes, fcn_planes, norm_cfg=norm_cfg)
        self.sr_hat = soft_region_pred(fcn_planes, num_classes)

        self.ocr_head:OCRContext = OCRContext(fcn_planes, ocr_planes, norm_cfg=norm_cfg)
        self.final_hat = final_pred(self.ocr_head.out_channels, num_classes)
    
    def forward(self, branches):
        feat = self.neck(branches)
        sr_pred = self.sr_hat(self.fcn_head(feat))
        ocr_pred = self.final_hat(self.ocr_head(feat, sr_pred))
        return {"soft_region": sr_pred, "pred": ocr_pred}