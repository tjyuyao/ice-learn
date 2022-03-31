import ice
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

ice.make_configurable(SGD, Adam)  # This makes SGD and Adam become a configurable class such that it will not be initialized immediately after passed in parameters.

# set command line arguments defaults

ice.args.setdefault("lr", 0.01, float, hparam=True)
ice.args.setdefault("max_iter", 0, int, hparam=True)
ice.args.setdefault("optim", Adam, lambda x: eval(x), hparam=True)

# nodes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def make_mnist_dataset(train: bool, batch_size: int):
    TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    return ice.DatasetNode(
        dataset=MNIST(
            download=True, root="/tmp/MNIST94117", transform=TRANSFORM, train=train
        ),
        batch_size=batch_size,
        shuffle=train,
    )


def epoch_end_report_hook(n: ice.MetricNode):
    if n.training:
        return
    avg_nll_loss = n.metric.evaluate().item()
    if n.launcher.rank == 0:
        print(f"steps={n.global_train_steps} avg_nll_loss={avg_nll_loss}")
        

class MultiClassAccuracy(ice.SummationMeter):

    def __init__(self, num_classes:int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        with torch.no_grad():
            pred = pred.argmax(dim=-1)
            confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=pred.device)
            for p, t in zip(pred.view(-1), gt.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            super().update(confusion_matrix)
            
    def evaluate(self):
        confusion_matrix = super().evaluate()
        accuracy = confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-8)
        return accuracy.mean()


# hypergraph
ice.add("mnist", make_mnist_dataset(train=True, batch_size=64), tags="train")
ice.add("mnist", make_mnist_dataset(train=False, batch_size=64), tags="val")
ice.add(
    "net",
    ice.ModuleNode(
        module=Net(),
        forward=lambda n, x: n.module(x["mnist"][0]),
        optimizers=ice.Optimizer(ice.args.optim(lr=ice.args.lr)),
    ),
)
ice.add(
    "loss",
    ice.LossNode(forward=lambda n, x: F.nll_loss(x["net"], x["mnist"][1])),
    tags="train",
)
ice.add(
    "accuracy",
    node=ice.MetricNode(
        MultiClassAccuracy(10),
        forward=lambda n, x: (x['net'], x['mnist'][1]),
    ),
    tags="val"
)


ice.print_forward_output("loss", every=200)


# training shedule
ice.run(
    [
        ice.Repeat(
            [
                ice.Task(train=True, epochs=1, tags="train"),
                ice.Task(train=False, epochs=1, tags="val"),
                ice.SaveCheckpointTask(),
            ],
            times=1,
        )
    ],
    devices="cuda:0",
    run_id=f"mnist",
)
