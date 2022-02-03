import dill
import pytest
import torch.nn as nn
from torch.nn import functional as F
from ice.llutil.config import configurable, make_configurable
from ice.llutil.launcher import ElasticLauncher
from ice.core.optim import Optimizer
from ice.core.module import ModuleNode
from torch.optim import SGD

make_configurable(nn.Conv2d)
make_configurable(nn.Conv2d)

def test_configurable():

    net = nn.Conv2d(7, 11, kernel_size=3, padding=1)

    assert net[0] == 7
    assert net[1] == 11
    assert net['kernel_size'] == 3
    assert net['padding'] == 1

    net['kernel_size'] = 5
    net['padding'] = 2

    assert net['kernel_size'] == 5
    assert net['padding'] == 2

    net.freeze()

    assert net.padding == (2, 2)
    assert net.kernel_size == (5, 5)


def test_clone():
    net = nn.Conv2d(7, 11, kernel_size=3, padding=1)
    net1 = net.clone().freeze()

    assert net1.padding == (1, 1)

    net['kernel_size'] = 5
    net['padding'] = 2

    net2 = net.clone().freeze()

    assert net2.padding == (2, 2)

@configurable
class AClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

def test_decoration():
    
    # partial initialization.
    i = AClass(b=0)

    # alter positional and keyword arguments afterwards.
    i[0] = 2
    i['b'] = 1

    # unfrozen configurable can be printed as a legal construction python statement.
    assert str(i) == "AClass(a=2, b=1)"

    # real initialization of original object.
    i.freeze()
    assert i.a == 2 and i.b == 1


def test_pickable():
    i = AClass(1, 2)
    i.__reduce__()
    buf = dill.dumps(i)
    j = dill.loads(buf)
    j.freeze()
    assert j.a == 1 and j.b == 2

@configurable
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


def test_module_pickable():
    i = ModuleNode(
        module=Net(),
        forward=lambda n, x: n.module(x['mnist'][0]),
        optimizers=Optimizer(SGD, dict(lr=0.01, momentum=0.5))
        )
    i.__reduce__()
    buf = dill.dumps(i)
    _ = dill.loads(buf)


def worker(aobj_buffer):
    aobj = dill.loads(aobj_buffer).freeze()
    assert aobj.a == 1 and aobj.b == 2


@pytest.mark.slow
def test_multiprocessing():
    launcher = ElasticLauncher(devices="auto:0,0").freeze()
    aobj = AClass(1, 2)
    launcher(worker, dill.dumps(aobj, byref=True))