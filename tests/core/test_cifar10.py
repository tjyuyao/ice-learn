import ice
import torch
from ice.core.loss import LossNode
from ice.core.metric import MetricNode
from torch import autocast, nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# arguments

ice.args.setdefault("lr", 0.0001, float, hparam=True)

# initialization

ice.init_autocast()
ice.make_configurable(Adam)
ice.set_gradient_accumulate(2)

# node

@ice.configurable
class Net(nn.Module):
    # define VGG 16
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=3)


def make_cifar10(train:bool, batch_size:int):
    TRANSFORM = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return ice.DatasetNode(
        dataset=CIFAR10(download=True, root="/home/wangling/TMP/cifar10", transform=TRANSFORM, train=train),
        batch_size=batch_size,
        shuffle=train,
    )

def report(n: MetricNode):
    if n.training: return
    avg_nll_loss = n.metric.evaluate().item()
    if n.launcher.rank == 0:
        print(f"steps={n.global_train_steps} avg_nll_loss={avg_nll_loss}")


# hypergraph

ice.add("cifar10", make_cifar10(train=True, batch_size=100), tags="train")
ice.add("cifar10", make_cifar10(train=False, batch_size=100), tags="val")
ice.add("net", ice.ModuleNode(
    module=Net(),
    forward=lambda n, x: n.module(x['cifar10'][0]),
    optimizers=ice.Optimizer(Adam(lr=ice.args.lr))
    ))
ice.add("nll_loss", LossNode(forward=lambda n, x: F.nll_loss(x["net"], x["cifar10"][1])))
ice.add("avg_nll_loss",
    ice.MetricNode(
        ice.AverageMeter(),
        forward=lambda n, x: (x['nll_loss'], x['cifar10'][1].size(0)),
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
