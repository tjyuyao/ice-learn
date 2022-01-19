import torch.nn as nn
from ice.llutil.config import make_configurable

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

def test_decoration():
    import ice

    @ice.configurable
    class AClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b
    
    # partial initialization.
    i = AClass(b=0)

    # alter positional and keyword arguments afterwards.
    i[0] = 2
    i['b'] = 1

    # unfrozen configurable can be printed as a legal construction python statement.
    assert repr(i) == "AClass(a=2, b=1)"

    # real initialization of original object.
    i.freeze()
    assert i.a == 2 and i.b == 1
