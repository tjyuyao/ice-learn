from typing import Type

import torch

import ice
from torch.optim import SGD

from datasets.cityscapes import Cityscapes, eval_pipeline, train_aug_pipeline
from ice.repro.y2020.HRNet_Semantic_Segmentation.modules.fcn_head import FCNHead
from ice.repro.y2020.HRNet_Semantic_Segmentation.modules.hat import DensePrediction
from ice.repro.y2020.HRNet_Semantic_Segmentation.modules.neck import ResizeConcat
from lr_updators import Poly
from metrics.semseg import SemsegIoUMetric
from modules.hrnet import HRNet18, UpsampleConv1x1
from modules.fcn_ocr_head import FCNOCRHead

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
        ),
        tags=["hrnet18", "cityscapes"])

ice.print_forward_output("head", every=1)

common_tasks = ["hrnet18", "cityscapes"]
run_id = '_'.join(common_tasks)

ice.run(
    run_id=run_id,
    tasks=ice.Task(train=True, steps=1, tags=["train", *common_tasks]),
    devices="cuda:0",
)
