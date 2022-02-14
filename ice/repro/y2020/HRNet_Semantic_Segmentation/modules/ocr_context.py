import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_module import ConvModule

from .self_attention_block import SelfAttentionBlock as _SelfAttentionBlock

ice.make_configurable(nn.Conv2d, nn.BatchNorm2d, nn.ReLU)


@ice.configurable
class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


@ice.configurable
class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg, act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None

        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        conv1x1 = conv_cfg(kernel_size=1)

        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            conv_cfg=conv1x1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = self.query_downsample(query_feats)

        return output


@ice.configurable
class OCRContext(nn.Module):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, in_channels, ocr_channels, scale=1, expansion=2, conv_cfg=nn.Conv2d(), norm_cfg=nn.BatchNorm2d(), act_cfg=nn.ReLU()):
        super(OCRContext, self).__init__()
        self.in_channels = in_channels
        self.ocr_channels = ocr_channels
        self.expansion = expansion
        self.scale = scale

        self.object_context_block = ObjectAttentionBlock(
            self.ocr_channels * expansion,
            self.ocr_channels,
            self.scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.spatial_gather_module = SpatialGatherModule(self.scale)

        conv3x3 = conv_cfg(kernel_size=3, padding=1)
        self.bottleneck = ConvModule(
            self.in_channels,
            self.ocr_channels * expansion,
            conv_cfg=conv3x3,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, inputs, prev_output):
        """Forward function."""
        feats = self.bottleneck(inputs)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        return object_context
    
    @property
    def out_channels(self):
        return self.ocr_channels * self.expansion