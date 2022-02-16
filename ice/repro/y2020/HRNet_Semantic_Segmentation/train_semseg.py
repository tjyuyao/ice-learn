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
from modules.hat import DensePrediction
from modules.hrnet import HRNet18, UpsampleConv1x1
from modules.neck import ResizeConcat
from modules.weight_init import kaiming_init

HOSTNAME = ice.get_hostname()
assert HOSTNAME in ("2080x8-1",), f"unknown host {HOSTNAME}"

PATHS = ice.ConfigDict()
PATHS["2080x8-1"].CITYSCAPES_ROOT = "/mnt/sdc/hyuyao/cityscapes_mmseg"
PATHS["2080x8-1"].HRNET18_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_cvt.pth"
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
        tags=["train", "cityscapes"])

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
            module=HRNet18(UpsampleConv1x1),
            forward=lambda n, x: n.module(x["dataset"]["img"]),
            optimizers=SGDPoly40k,
            weight_init_fn=lambda m: m.load_state_dict(torch.load(PATHS.HRNET18_PRETRAINED)),
        ),
        tags="hrnet18")

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
                final_pred=DensePrediction(dropout_ratio=-1),
            ),
            forward=lambda n, x: n.module(x["backbone"]),
            optimizers=SGDPoly40k,
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes"])

def resize_cross_entropy(seg_logit, seg_gt):
    seg_logit = F.interpolate(
        seg_logit, 
        size=seg_gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    loss = F.cross_entropy(
        seg_logit,
        seg_gt,
        ignore_index=255,
    )
    return loss

ice.add(name="loss",
    node=ice.LossNode(
        forward= lambda n, x: (
            resize_cross_entropy(x["head"]["soft_region"], x["dataset"]["seg"]) * 0.4 +
            resize_cross_entropy(x["head"]["pred"], x["dataset"]["seg"])
        ),
    )
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
        forward = lambda n, x: [x["head"]["pred"], x["dataset"]["seg"]],
        epoch_end=report
    )
)

ice.print_forward_output("loss", every=100)

common_tags = ["hrnet18", "cityscapes"]
run_id = '_'.join(common_tags)

ice.run(
    run_id=run_id,
    tasks=[
        # lambda g: g.load_checkpoint("out/hrnet18_cityscapes_gsgacu_j/ckpts/E0S114.pth"),
        ice.Repeat(
            [
                ice.Task(train=True, steps=4000, tags=["train", *common_tags]),
                ice.Task(train=False, epochs=1, tags=["val", *common_tags]),
            ],
            times=10
        ),
        lambda g: g.save_checkpoint()
        ],
    devices="cuda:0-3",
    tee="3"
)