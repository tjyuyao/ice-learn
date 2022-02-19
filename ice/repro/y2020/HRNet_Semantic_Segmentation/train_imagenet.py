from typing import Type
import torch

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

import ice
import os.path as osp
import torchvision.datasets as datasets
import torchvision.transforms as transforms

ice.make_configurable(datasets.ImageFolder)

IMAGENET_ROOT = "/root/autodl-nas/imagenet"
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
DATASETNODE:Type[ice.DatasetNode] = ice.DatasetNode(
    num_workers=128//6,
    batch_size=128,
    pin_memory=True,
)
ice.add(
    name="dataset",
    node=DATASETNODE(
        dataset=datasets.ImageFolder(
            root=osp.join(IMAGENET_ROOT, "train"),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ])
        ),
        shuffle=True,
    ),
    tags="train"
)

ice.add(
    name="dataset",
    node=DATASETNODE(
        dataset=datasets.ImageFolder(
            root=osp.join(IMAGENET_ROOT, "val"),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ])
        ),
        shuffle=False,
    ),
    tags="val"
)

import torch.nn as nn
import torch.nn.functional as F
from modules.weight_init import kaiming_init
from modules.hrnet import BottleNeckResBlock, GuidedUpsampleConv1x1, HRNet18, UpsampleConv1x1
from modules.cls_head import ClassificationHead
from lars import LARS

BATCHNORM = nn.BatchNorm2d(momentum=0.05)
OPTIMIZER = ice.Optimizer(LARS, dict(lr=0.01))

def weight_init(m: nn.Module):
    def _init(m:nn.Module):
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, bias=0.001)
    m.apply(_init)

    def _init(m:nn.Module):
        if hasattr(m, "init_weights"):
            m.init_weights()
    m.apply(_init)

ice.add(name="backbone",
        node=ice.ModuleNode(
            module=HRNet18(UpsampleConv1x1(), norm_cfg=BATCHNORM),
            forward=lambda n, x: n.module(x["dataset"][0]),
            optimizers=OPTIMIZER,
            weight_init_fn=weight_init,
        ),
        tags=["hrnet18", "bilinear"])

ice.add(name="backbone",
        node=ice.ModuleNode(
            module=HRNet18(GuidedUpsampleConv1x1(window_size=5), norm_cfg=BATCHNORM),
            forward=lambda n, x: n.module(x["dataset"][0]),
            optimizers=OPTIMIZER,
            weight_init_fn=weight_init,
        ),
        tags=["hrnet18", "crela"])

ice.add(name="pred",
        node=ice.ModuleNode(
            module=ClassificationHead(
                inplanes=HRNet18.out_channels,
                num_classes=1000,
                planes=[32, 64, 128, 256],
                block_type=BottleNeckResBlock(),
                norm_cfg=BATCHNORM,
                expansion=2,
            ),
            forward=lambda n, x: n.module(x['backbone']),
            optimizers=OPTIMIZER,
            weight_init_fn=weight_init,
        ),
        tags="hrnet18")

ice.add(name="loss",
        node=ice.LossNode(
            forward = lambda n, x: F.cross_entropy(x["pred"], x["dataset"][1])
        ),
        tags="train")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"acc-{k}"] = correct_k.mul_(100.0 / batch_size)
        return res

def report(n: ice.MetricNode):
    acc = n.metric.evaluate()
    print(f"Epoch {n.global_epochs}: ACC-1 = {acc['acc-1'].item()}, ACC-5 = {acc['acc-5'].item()}")

ice.add(name="eval",
        node=ice.MetricNode(
            metric=ice.AverageMeter(),
            forward = lambda n, x: accuracy(x["pred"], x["dataset"][1], topk=(1, 5)),
            epoch_end=report,
        ),
        tags="val")

ice.print_forward_output("loss", every=100)

import datetime

if __name__ == "__main__":
    ice.run(
        run_id="imagenet",
        tasks=[
            lambda : print(datetime.datetime.today()),
            ice.Task(train=True, epochs=1, tags=["train", "hrnet18", "crela"]),
            ice.Task(train=False, epochs=1, tags=["val", "hrnet18", "crela"]),
            lambda : print(datetime.datetime.today()),
            lambda g: g.save_checkpoint(tags=["hrnet18", "crela"])
        ],
        devices="cuda:0",
        tee='3',
    )

    # import os
    # os.system("shutdown")