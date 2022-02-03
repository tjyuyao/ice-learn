import ice
import torch
from torch import log_, nn
from torch.nn import functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

@ice.configurable
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


_C = ice.ConfigDict()

_C.DATASETS.MNIST.TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
_C.DATASETS.MNIST.TRAIN = MNIST(download=True, root="/tmp/MNIST94117", transform=_C.DATASETS.MNIST.TRANSFORM, train=True)
_C.DATASETS.MNIST.TRAIN_NODE = ice.DatasetNode(_C.DATASETS.MNIST.TRAIN, batch_size=2, shuffle=True)

_C.MODULES.NET_NODE = ice.ModuleNode(
    module=Net(),
    forward=lambda n, x: n.module(x['mnist'][0]),
    optimizers=ice.Optimizer(SGD, dict(lr=0.01, momentum=0.5))
    )

_C.GRAPHS.G1 = ice.HyperGraph()
_C.GRAPHS.G1.add("mnist", _C.DATASETS.MNIST.TRAIN_NODE, tags="train")
_C.GRAPHS.G1.add("net", _C.MODULES.NET_NODE)
_C.GRAPHS.G1.print_output_of("net")
_C.GRAPHS.G1.run(
    ice.Task(train=True, steps=10, tags="train"),
    devices="cuda:0",
)
