import ice
import torch
from ice.core.loss import LossNode
from ice.core.metric import MetricNode
from torch import autocast, nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# arguments

ice.args.setdefault("lr", 0.01, float, hparam=True)

# initialization

ice.init_autocast()
ice.make_configurable(SGD, Adam)
ice.set_gradient_accumulate(2)

# node

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


def make_mnist(train:bool, batch_size:int):
    TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    return ice.DatasetNode(
        dataset=MNIST(download=True, root="/tmp/MNIST94117", transform=TRANSFORM, train=train),
        batch_size=batch_size,
        shuffle=train,
    )

def report(n: MetricNode):
    if n.training: return
    avg_nll_loss = n.metric.evaluate().item()
    if n.launcher.rank == 0:
        print(f"steps={n.global_train_steps} avg_nll_loss={avg_nll_loss}")


# hypergraph

ice.add("mnist", make_mnist(train=True, batch_size=100), tags="train")
ice.add("mnist", make_mnist(train=False, batch_size=100), tags="val")
ice.add("net", ice.ModuleNode(
    module=Net(),
    forward=lambda n, x: n.module(x['mnist'][0]),
    optimizers=ice.Optimizer(Adam(lr=ice.args.lr))
    ))
ice.add("nll_loss", LossNode(forward=lambda n, x: F.nll_loss(x["net"], x["mnist"][1])))
ice.add("avg_nll_loss", 
    ice.MetricNode(
        ice.AverageMeter(),
        forward=lambda n, x: (x['nll_loss'], x['mnist'][1].size(0)),
        epoch_end=report,
    ))
ice.print_forward_output("nll_loss", every=100)


# training shedule
ice.run(
    [
        ice.Repeat([
            ice.Task(train=True, epochs=1, tags="train"),
            ice.SaveCheckpointTask(),
            ice.Task(train=False, epochs=1, tags="val"),
        ], times=3)
    ],
    devices="cuda:0,0",
    omp_num_threads=4,
    monitor_interval=1,
    # tee="3"
)
