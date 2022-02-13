from typing import Type

import torch

import ice
from torch.optim import SGD

from datasets.cityscapes import Cityscapes, eval_pipeline, train_aug_pipeline
from lr_updators import Poly
from metrics.semseg import SemsegIoUMetric
from modules.hrnet import HRNet18, UpsampleConv1x1

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

ice.add(name="cityscapes",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "train"),
            batch_size=4,
            shuffle=True,
            pipeline=train_aug_pipeline(),
        ),
        tags="train")

ice.add(name="cityscapes",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "val"),
            batch_size=1,
            shuffle=False,
            pipeline=eval_pipeline(),
        ),
        tags="val")

ice.add(name="backbone",
        node=ice.ModuleNode(
            module=HRNet18(UpsampleConv1x1),
            forward=lambda n, x: n.module(x["cityscapes"]["img"]),
            optimizers=SGDPoly40k,
            weight_init_fn=lambda m: m.load_state_dict(torch.load(PATHS.HRNET18_PRETRAINED)),
        ),
        tags="hrnet18"
)

ice.print_forward_output("backbone", every=1)

ice.run(
    run_id="test",
    tasks=ice.Task(train=True, steps=1, tags=["train", "hrnet18"]),
    devices="cuda:0",
)
