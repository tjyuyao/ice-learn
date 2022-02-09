import ice
import ice.api.transforms as IT

from datasets.cityscapes import Cityscapes, train_aug_pipeline, eval_pipeline

HOSTNAME = ice.get_hostname()
assert HOSTNAME in ("2080x8-1",), f"unknown host {HOSTNAME}"

CITYSCAPES_ROOT = {
    "2080x8-1": "/mnt/sdc/hyuyao/cityscapes_mmseg"
}[HOSTNAME]

g = ice.HyperGraph()

g.add(
    name="cityscapes",
    node=ice.DatasetNode(
        dataset=Cityscapes(CITYSCAPES_ROOT, "train"),
        batch_size=4, shuffle=True, num_workers=4, pin_memory=True,
        pipeline=train_aug_pipeline(),
    ),
    tags="train"
)

g.add(
    name="cityscapes",
    node=ice.DatasetNode(
        dataset=Cityscapes(CITYSCAPES_ROOT, "val"),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
        pipeline=eval_pipeline(),
    ),
    tags="val"
)

g.print_forward_output("cityscapes", every=1)

g.run(
    run_id="test",
    tasks=ice.Task(train=True, steps=1, tags="train"),
    devices="cuda:0",
)
