# Get Started 快速开始

## Hello, MNIST! 你好，MNIST!

When you have plenty of combinations of ideas to try, or many different datasets and various tasks and want to share some network modules, or hyperparameter configurations, etc., but don't want to maintain a pile of code and configuration files, use ice-learn to orchestrate your training is the most convenient!

当你有许多想法的排列组合可以尝试，或者有许多不同数据集和不同的任务想要共享一些网络模块、超参数配置，又不想维护一大堆代码和配置文件，使用 ice-learn 来编排你的训练就是最为方便的！

To illustrate several parts of the sample program, imagine that you are cooking a dish with one of the latest programmable rice cookers! You first need to source and prepare the ingredients (corresponding to lines 1-76 of the following code, please scan it quickly when you read it for the first time!), then pre-program your rice cooker (corresponding to lines 78-102), and finally Press the start button to start automatic cooking (line 105 to the end). After you've read the code, we'll walk through the code's functionality section by section.

为了说明示例程序的几个部分，请设想您在使用一台最新款可编程的电饭锅烹饪菜肴！您首先需要采购和准备食材（对应接下来代码的1-76行，请您在首次阅读时快速扫过它！），然后对你的电饭锅进行事先编程（对应78-102行），最后按下启动按扭开始自动烹饪（105行到最后）。在您阅读完代码后，我们将逐个部分介绍代码的功能。

```py

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

# ingredients

@ice.configurable
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
        return x


def make_mnist_dataset(train: bool, batch_size: int):
    TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    return ice.DatasetNode(
        dataset=MNIST(
            download=True, root="/tmp/MNIST94117", transform=TRANSFORM, train=train
        ),
        batch_size=batch_size,
        shuffle=train,
    )

def binary_focal_loss(logits, target, gamma=2.0, weight=None, reduction="none", modulator_no_grad=False):
    t = F.one_hot(target, num_classes=logits.shape[-1])

    if modulator_no_grad:
        p = torch.sigmoid(logits.detach())
    else:
        p = torch.sigmoid(logits)
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = (1-pt).pow(gamma)
    if weight:
        w = weight*t + (1-weight)*(1-t)  # w = alpha if t > 0 else 1-alpha
    return F.binary_cross_entropy_with_logits(logits, t, w, reduction=reduction)


def categorical_focal_loss(
    logits, target, gamma=2.0, weight=None, reduction="none", modulator_no_grad=False
):
    log_prob = F.log_softmax(logits, dim=-1)
    if modulator_no_grad:
        prob = torch.exp(log_prob.detach())
    else:
        prob = torch.exp(log_prob)
    return F.nll_loss(
        ((1 - prob) ** gamma) * log_prob, target, weight=weight, reduction=reduction
    )
        

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


def report_hook(n: ice.MetricNode):
    if n.training:
        return
    avg_nll_loss = n.metric.evaluate().item()
    if n.launcher.rank == 0:
        print(f"steps={n.global_train_steps} accuracy={avg_nll_loss}")


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
    epoch_end=report_hook,
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

```

## Configuration as Code! 配置即代码！

In the world of ice-learn, we do not use additional configuration files. The entry python script of the program is responsible for both programming and storing various configuration items. The purpose of this design is to avoid duplication.

在 ice-learn 的世界中，我们不使用额外的配置文件，程序的入口python脚本即承担编程的功能，也承担存储各种配置项的功能。这样设计的目的是避免重复。

**TODO**: 添加一些第一段的代码说明，包括整体的实验目标的说明。

## Use `HyperGraph` to store all ingredients just like a refridgerator!

An executable configuration graph.

Note:
    We describe the concept of this core module in following few lines and show some pesudo-codes. This is very close to but not the same as the real code.

An acyclic directed hypergraph $G$ consists of a set of vertices $V$ and a set of hyperarcs $H$, where a hyperarc is a pair $<X, Y>$ , $X$ and $Y$ non empty subset of $V$.

We have a tag system that split the vertices $V$ into maybe overlapping subsets $V_i$, that each of which is a degenerated hypergraph $G_i$ that only consists of vertices $V_i$ and a set of hyperarcs $H_i$ so that each hyperarc is a pair $<x, Y>$, where $x \in V_i$ and $Y \subset V_i$. We call tails $x$ as producers and heads $Y$ as consumers in each hyperarc, this states the dependencies.

User defines a vertice (`Node` in the code) by specify a computation process $f$ (`forward` in the code) and the resources $R$ (`Dataset`s, `nn.Module`s, imperatively programmed function definitions such as losses and metrics, etc.) needed by it.

```python
vertice_1 = Node(
    name = "consumer_node_name",
    resources = ...,
    forward = lambda n, x: do_something_with(n.resources, x["producer_node_name"]),
    tags = ["group1", "group2"],
)
```

A longer version of `forward` parameter that corresponds to the previous notation would be `forward = lambda self, V_i: do_something_with(self.resources, V_i["x"])`,  but we will stick to the shorter version in the code.

So at the time of configuration, we are able to define every material as a node, and the name of nodes can be duplicated, i.e. multiple $x\in V$ can have the same identifier, as long as they does not have the same tag $i$ that selects $V_i$. The tags mechanism is flexible. Every node can have multiple of them, and multiple tags can be specified so that a union of subsets will be retrieved. If no tag is specified for a node, a default tag `*` will be used and a retrival will always include the `*` group.

```python
hyper_graph = HyperGraph([
    vertice_1,
    vertice_2,
    ...,
    vertice_n,
])

activated_graph = hyper_graph["group1", "group3", "group5"]
freeze_and_execute(activated_graph)
```

**TODO**: 包含第二段代码的说明。翻译上面一段的文字。


## Each meal will make use of only a part of the ingredients!

**TODO**: 包含第三段代码的说明。简要介绍`ice.Task`模块的功能。

## Conclusion 小结

恭喜！您现在已经了解了 ice-learn 的基本工作哲学。

**What Next?** 接下来可以做什么？
