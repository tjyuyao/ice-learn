from typing import Type
import ice

from datasets.cityscapes import Cityscapes, train_aug_pipeline, eval_pipeline
from modules.hrnet import HRNet18
from metrics.semseg import SemsegIoUMetric

HOSTNAME = ice.get_hostname()
assert HOSTNAME in ("2080x8-1",), f"unknown host {HOSTNAME}"

CITYSCAPES_ROOT = {
    "2080x8-1": "/mnt/sdc/hyuyao/cityscapes_mmseg"
}[HOSTNAME]

DatasetNode:Type[ice.DatasetNode] = ice.DatasetNode(num_workers=4, pin_memory=True)

ice.add(name="cityscapes",
        node=DatasetNode(
            dataset=Cityscapes(CITYSCAPES_ROOT, "train"),
            batch_size=4,
            shuffle=True,
            pipeline=train_aug_pipeline(),
        ),
        tags="train")

ice.add(name="cityscapes",
        node=DatasetNode(
            dataset=Cityscapes(CITYSCAPES_ROOT, "val"),
            batch_size=1,
            shuffle=False,
            pipeline=eval_pipeline(),
        ),
        tags="val")

ice.print_forward_output("cityscapes", every=1)

ice.run(
    run_id="test",
    tasks=ice.Task(train=True, steps=1, tags="train"),
    devices="cuda:0",
)
