import torch
import ice
from torch.utils.data import Dataset


class Int100Dataset(Dataset):
    
    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        return torch.Tensor([index+1])


g = ice.HyperGraph()
    
g.add(
    name="int",
    node=ice.DatasetNode(
        dataset=Int100Dataset(),
        batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    ),
)

g.add(
    name="sum",
    node=ice.MetricNode(
        ice.SummationMeter(),
        forward=lambda n, x: x['int'].sum()
    ),
)

def test_dataset_metric():

    def _test(g:ice.HyperGraph):
        assert g['sum'].metric.evaluate().item() == 5050

    g.run(
        [
            ice.Task(train=True, epochs=1),
            _test,
        ],
        devices="cpu:0,1"
    )