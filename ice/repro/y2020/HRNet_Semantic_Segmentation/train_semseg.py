from typing import Type

import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from datasets.cityscapes import Cityscapes, eval_pipeline, train_aug_pipeline
from lr_updators import Poly
from metrics.semseg import SemsegIoUMetric
from modules.fcn_head import FCNHead
from modules.fcn_ocr_head import FCNOCRHead
from modules.hat import DensePrediction, LAHead18a
from modules.hrnet import GuidedUpsampleConv1x1, HRNet18, UpsampleConv1x1
from modules.local_attn_2d import local_attn_2d
from modules.neck import ResizeConcat
from modules.weight_init import kaiming_init

FASTER_TRAINING = True
if FASTER_TRAINING:
    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

REPRODUCABLE_TRAINING = False
if REPRODUCABLE_TRAINING:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

ice.make_configurable(nn.BatchNorm2d)

HOSTNAME = ice.get_hostname()
assert HOSTNAME in ("2080x8-1",), f"unknown host {HOSTNAME}"

PATHS = ice.ConfigDict()
PATHS["2080x8-1"].CITYSCAPES_ROOT = "/mnt/sdc/hyuyao/cityscapes_mmseg"
PATHS["2080x8-1"].HRNET18_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_cvt.pth"
PATHS["2080x8-1"].HRNET18_CRELA_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_crela.pth"
PATHS = PATHS[HOSTNAME]

DatasetNode:Type[ice.DatasetNode] = ice.DatasetNode(num_workers=4, pin_memory=True)

SGDPoly40k = ice.Optimizer(
    SGD, dict(lr=0.01, momentum=0.9, weight_decay=0.0005),
    updators_per_step=[Poly(power=0.9, min_lr=1e-4, max_updates=40000)]
)

ice.add(name="dataset",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "train"),
            batch_size=4,
            shuffle=True,
            pipeline=train_aug_pipeline(),
        ),
        tags=["train", "cityscapes", "bilinear"])

import cv2
import numpy as np
import ice.api.transforms as IT

@ice.dictprocess
def make_edge_gt(seg):
    img_blur = cv2.GaussianBlur(seg, (3,3), 0)
    edge = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    edge = edge.astype(np.float) / 255.
    return edge

def train_aug_pipeline_with_edge(
    resize_ratio_low = .5,
    resize_ratio_high = 2.,
    crop_height = 512,
    crop_width = 1024,
    cat_max_ratio = 0.75,
    ignore_index = 255,
    random_flip_prob = .5,
    normalize_mean = [123.675, 116.28, 103.53],
    normalize_std = [58.395, 57.12, 57.375],
):
    return [
        # Load
        IT.image.Load(img_path="img_path", dst="img"),
        IT.semseg.LoadAnnotation(seg_path="seg_path", dst="seg"),
        # Random Resize
        IT.random.RandomFloats(low=resize_ratio_low, high=resize_ratio_high, dst="resize_ratio"),
        IT.image.spatial.Resize(scale="resize_ratio", src="img", dst="img"),
        IT.image.spatial.Resize(scale="resize_ratio", src="seg", dst="seg"),
        # Random Crop
        IT.semseg.RandomCrop(dst_h=crop_height, dst_w=crop_width, cat_max_ratio=cat_max_ratio, ignore_index=ignore_index,
                                img="img", seg="seg", dst=dict(img="img_roi", seg="seg_roi")),
        # Random Flip
        IT.random.RandomDo([IT.image.spatial.Flip(src="img", dst="img"),
                            IT.image.spatial.Flip(src="seg", dst="seg")], prob=random_flip_prob),
        make_edge_gt(seg="seg", dst="edge"),
        # Photometric Augmentation
        IT.image.photometric.PhotoMetricDistortion(img="img", dst="img"),
        # Normalize & ToTensor
        IT.image.Normalize(
            img="img", dst="img", mean=normalize_mean,
            std=normalize_std, to_rgb=True),
        IT.image.ToTensor(img="img", dst="img"),
        IT.semseg.LabelToTensor(src="seg", dst="seg"),
        IT.Collect("img", "seg", "edge")
    ]

ice.add(name="dataset",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "train"),
            batch_size=2,
            shuffle=True,
            pipeline=train_aug_pipeline_with_edge(),
        ),
        tags=["train", "cityscapes", "crela"])

ice.add(name="dataset",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "val"),
            batch_size=1,
            shuffle=False,
            pipeline=eval_pipeline(),
        ),
        tags=["val", "cityscapes"])

ice.add(name="backbone",
        node=ice.ModuleNode(
            module=HRNet18(UpsampleConv1x1()),
            forward=lambda n, x: n.module(x["dataset"]["img"]),
            optimizers=SGDPoly40k,
            weight_init_fn=lambda m: m.load_state_dict(torch.load(PATHS.HRNET18_PRETRAINED)),
        ),
        tags=["hrnet18", "bilinear"])

ice.add(name="backbone",
        node=ice.ModuleNode(
            module=HRNet18(GuidedUpsampleConv1x1(window_size=5), checkpoint_enabled=True),
            forward=lambda n, x: n.module(x["dataset"]["img"]),
            optimizers=SGDPoly40k,
            weight_init_fn=lambda m: m.load_state_dict(torch.load(PATHS.HRNET18_CRELA_PRETRAINED)),
        ),
        tags=["hrnet18", "crela"])

def weight_init_fcn_ocr_head(m: nn.Module):
    def _init(m:nn.Module):
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
    m.apply(_init)

    def _init(m:nn.Module):
        if hasattr(m, "init_weights"):
            m.init_weights()
    m.apply(_init)

ice.add(name="head",
        node=ice.ModuleNode(
            module=FCNOCRHead(
                inplanes=HRNet18.out_channels,
                ocr_planes=256,
                num_classes=Cityscapes.NUM_CLASSES,
                neck_cfg=ResizeConcat,
                fcn_head=FCNHead(num_convs=1, kernel_size=1),
                soft_region_pred=DensePrediction(dropout_ratio=-1),
            ),
            forward=lambda n, x: n.module(x["backbone"]),
            optimizers=SGDPoly40k,
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes"])

ice.add(name="hat",
        node=ice.ModuleNode(
            module=DensePrediction(512, Cityscapes.NUM_CLASSES, dropout_ratio=-1),
            forward=lambda n, x: {"pred":n.module(x["head"]["feat"])},
            optimizers=SGDPoly40k,
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes", "bilinear"])

ice.add(name="hat",
        node=ice.ModuleNode(
            module=LAHead18a(
                in_coarser_channels=512,
                out_channels=Cityscapes.NUM_CLASSES, 
                in_finer_channels=3,
                hidden_channels=32,
                kernel_size=7,
                dilation=2,
                upsample_mode="bilinear",
            ),
            forward=lambda n, x: n.module(x["head"]["feat"], x["dataset"]["img"]),
            optimizers=SGDPoly40k,
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes", "crela"])

def bilinear(x, guide):
    return F.interpolate(
        x, 
        size=guide.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

def cross_entropy(seg_logit, seg_gt):
    loss = F.cross_entropy(
        seg_logit,
        seg_gt,
        ignore_index=255,
    )
    return loss

ice.add(name="loss",
    node=ice.LossNode(
        forward= lambda n, x: (
            cross_entropy(bilinear(x["head"]["soft_region"], x["dataset"]["img"]), x["dataset"]["seg"]) * 0.4 +
            cross_entropy(bilinear(x["hat"], x["dataset"]["img"]), x["dataset"]["seg"])
        ),
    ),
    tags=["bilinear", "train"]
)

ice.add(name="loss",
    node=ice.LossNode(
        forward= lambda n, x: (
            cross_entropy(bilinear(x["head"]["soft_region"], x["dataset"]["img"]), x["dataset"]["seg"]) * 0.4 +
            cross_entropy(x["hat"]["pred"], x["dataset"]["seg"]) +
            ((1 - x["hat"]["non_edge"]) - x["dataset"]["edge"]).abs().mean()
        ),
    ),
    tags=["crela", "train"]
)

def report(n: ice.MetricNode):
    iou = n.metric.evaluate()
    miou = torch.mean(iou)
    if n.launcher.local_rank == 0:
        print("iou =", iou.cpu().numpy().tolist(), ",")
        print("miou =", miou.item(), ",", flush=True)

ice.add(name="miou",
    node=ice.MetricNode(
        metric=SemsegIoUMetric(num_classes=Cityscapes.NUM_CLASSES, ignore_index=Cityscapes.IGNORE_INDEX),
        forward = lambda n, x: [x["hat"]["pred"], x["dataset"]["seg"]],
        epoch_end=report
    )
)

ice.print_forward_output("loss", every=100)

common_tags = ["hrnet18", "cityscapes", "crela"]
run_id = '_'.join(common_tags) + "_maskloss"

ice.run(
    run_id=run_id,
    tasks=[
        # lambda g: g.load_checkpoint("out/hrnet18_cityscapes_gsgacu_j/ckpts/E0S114.pth"),
        ice.Repeat(
            [
                ice.Task(train=True, steps=4000, tags=["train", *common_tags]),
                lambda g: g.save_checkpoint(),
                ice.Task(train=False, epochs=1, tags=["val", *common_tags]),
            ],
            times=10
        ),
    ],
    devices="cuda:0-4,6-7",
    tee="3",
    master_port=9000
)