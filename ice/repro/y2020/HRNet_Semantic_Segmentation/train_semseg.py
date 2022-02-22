import os
from typing import Type

import cv2
import numpy as np

import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from datasets.cityscapes import Cityscapes, eval_pipeline, train_aug_pipeline
from ice.core.graph import GraphOutputCache, StopAllTasks
from lr_updators import Poly
from metrics.semseg import SemsegIoUMetric
from modules.fcn_head import FCNHead
from modules.fcn_ocr_head import FCNOCRHead
from modules.hat import DensePrediction, LAHead18a
from modules.hrnet import GuidedUpsampleConv1x1, HRNet18, UpsampleConv1x1
from modules.neck import ResizeConcat
from modules.weight_init import kaiming_init

ice.make_configurable(nn.BatchNorm2d)

HOSTNAME = ice.get_hostname()
assert HOSTNAME in ("2080x8-1", "3090x4-2"), f"unknown host {HOSTNAME}"

PATHS = ice.ConfigDict()
PATHS["2080x8-1"].CITYSCAPES_ROOT = "/mnt/sdc/hyuyao/cityscapes_mmseg"
PATHS["2080x8-1"].HRNET18_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_cvt.pth"
PATHS["2080x8-1"].HRNET18_CRELA_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_crela.pth"
PATHS["3090x4-2"].CITYSCAPES_ROOT = "/home/hyuyao/2021/data/CityScapes/basic"
PATHS["3090x4-2"].HRNET18_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_cvt.pth"
PATHS["3090x4-2"].HRNET18_CRELA_PRETRAINED = "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006_crela.pth"
PATHS = PATHS[HOSTNAME]

ice.args.setdefault("model", "bilinear", str)  # "bilinear", "crela"
ice.args.setdefault("batch_size", 4, int)
ice.args.setdefault("power", 0.9, float)
ice.args.setdefault("run", "train", str)  # "train", "viz"
ice.args.setdefault("devices", "cuda:0-3", str)
ice.args.setdefault("lr_scale", 1., float)
ice.args.setdefault("step_scale", 1, int)
ice.args.setdefault("grad_acc", 1, int)
ice.args.setdefault("checkpoint", False, bool)

DatasetNode:Type[ice.DatasetNode] = ice.DatasetNode(num_workers=12, pin_memory=True)
SGDPoly40k = ice.Optimizer(
    SGD, dict(lr=0.01*ice.args.lr_scale, momentum=0.9, weight_decay=0.0005),
    updators_per_step=[Poly(power=ice.args.power, min_lr=1e-4*ice.args.lr_scale, max_updates=40000 * ice.args.step_scale)],
    gradient_accumulation_steps=ice.args.grad_acc
)
ModuleNode:Type[ice.ModuleNode] = ice.ModuleNode(
    optimizers=SGDPoly40k,
    autocast_enabled=True,
)

ice.add(name="dataset",
        node=DatasetNode(
            dataset=Cityscapes(PATHS.CITYSCAPES_ROOT, "train"),
            batch_size=ice.args.batch_size,
            shuffle=True,
            pipeline=train_aug_pipeline(),
            prefetch_factor=4,
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
        node=ModuleNode(
            module=HRNet18(UpsampleConv1x1()),
            forward=lambda n, x: n.module(x["dataset"]["img"]),
            weight_init_fn=lambda m: m.load_state_dict(torch.load(PATHS.HRNET18_PRETRAINED)),
        ),
        tags=["hrnet18", "bilinear"])

ice.add(name="backbone",
        node=ModuleNode(
            module=HRNet18(GuidedUpsampleConv1x1(window_size=5), checkpoint_enabled=ice.args.checkpoint),
            forward=lambda n, x: n.module(x["dataset"]["img"]),
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
        node=ModuleNode(
            module=FCNOCRHead(
                inplanes=HRNet18.out_channels,
                ocr_planes=256,
                num_classes=Cityscapes.NUM_CLASSES,
                neck_cfg=ResizeConcat,
                fcn_head=FCNHead(num_convs=1, kernel_size=1),
                soft_region_pred=DensePrediction(dropout_ratio=-1),
            ),
            forward=lambda n, x: n.module(x["backbone"]),
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes"])

ice.add(name="hat",
        node=ModuleNode(
            module=DensePrediction(512, Cityscapes.NUM_CLASSES, dropout_ratio=-1),
            forward=lambda n, x: {"pred":n.module(x["head"]["feat"])},
            weight_init_fn=weight_init_fcn_ocr_head,
        ),
        tags=["hrnet18", "cityscapes", "bilinear"])

ice.add(name="hat",
        node=ModuleNode(
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
            cross_entropy(bilinear(x["hat"]["pred"], x["dataset"]["img"]), x["dataset"]["seg"])
        ),
    ),
    tags=["bilinear", "train"]
)

ice.add(name="loss",
    node=ice.LossNode(
        forward= lambda n, x: (
            cross_entropy(bilinear(x["head"]["soft_region"], x["dataset"]["img"]), x["dataset"]["seg"]) * 0.4 +
            cross_entropy(x["hat"]["pred"], x["dataset"]["seg"])
            + (x["hat"]["non_edge"]).abs().mean() * 0.1
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

common_tags = ["hrnet18", "cityscapes", ice.args["model"]]
run_id = '_'.join(common_tags) + "_maskloss"

if ice.args["run"] == "train":

    ice.run(
        run_id=run_id,
        tasks=[
            # lambda g: g.load_checkpoint("out/hrnet18_cityscapes_gsgacu_j/ckpts/E0S114.pth"),
            ice.Repeat(
                [
                    ice.Task(train=True, steps=4000 * ice.args.step_scale, tags=["train", *common_tags]),
                    lambda g: g.save_checkpoint(),
                    ice.Task(train=False, epochs=1, tags=["val", *common_tags]),
                ],
                times=10
            ),
        ],
        devices=ice.args.devices,
        tee="3",
        master_port=9000
    )

elif ice.args["run"] == "viz":

    def viz(n: ice.Node, x: GraphOutputCache):
        viz_dir = os.path.join(n.out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        with torch.no_grad():
            seg = x['dataset']['seg']
            mask = torch.logical_and(seg < n.num_classes, seg > 0)
            seg = F.one_hot(seg * mask, num_classes=n.num_classes).permute(0, 3, 1, 2)
            pred = F.softmax(x['hat']['pred'], dim=1)
            diff = torch.max((seg-pred).abs_(), dim=1)[0] * mask * 255
            error = diff[0].cpu().numpy().astype("uint8")
            error = cv2.applyColorMap(error, cv2.COLORMAP_JET)
            raw_img = x['dataset']['raw_img'][0].cpu().numpy()
            view = np.concatenate((raw_img, error), axis=0)
            cv2.imwrite(f"{viz_dir}/error_{n.task_steps:08d}.jpg", view)

        with torch.no_grad():
            seg = x['dataset']['seg']
            mask = torch.logical_and(seg < n.num_classes, seg > 0)
            seg = (seg * mask)[0].cpu().numpy() / float(n.num_classes) * 255.
            seg = cv2.applyColorMap(seg.astype("uint8"), cv2.COLORMAP_JET)
            edge = x["hat"]["non_edge"][0].permute(1,2,0).cpu().numpy() * 255.
            edge = cv2.cvtColor(edge.astype("uint8"), cv2.COLOR_GRAY2BGR)
            edge_gt = x["dataset"]['edge'].permute(1,2,0).cpu().numpy() * 255.
            edge_gt = cv2.cvtColor(edge_gt.astype("uint8"), cv2.COLOR_GRAY2BGR)
            view = np.concatenate((raw_img.cpu().numpy().astype("uint8"), edge, edge_gt), axis=0)
            cv2.imwrite(f"{viz_dir}/edge_{n.task_steps:08d}.jpg", view)

        if n.task_steps > 20:
            tgzfile = os.path.join(n.out_dir, f"{n.run_id}.tgz")
            os.system(f"tar czf {tgzfile} {viz_dir}")
            print(tgzfile)
            raise StopAllTasks()

    ice.add(
        "viz",
        node=ice.Node(
            forward=viz,
            num_classes=19,
        )
    )

    ice.run(
        run_id="viz",
        resume_from="/home/hyuyao/2021/ice-learn-public/saved/hrnet18_cityscapes_crela_maskloss_dffczip2/ckpts/E0S40000.pth",
        devices=ice.args.devices,
        tasks=ice.Task(train=False, epochs=1, tags=["val", *common_tags]),
        monitor_interval=0.1
    )