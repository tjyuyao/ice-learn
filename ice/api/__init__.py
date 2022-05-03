from __future__ import annotations
from typing import TYPE_CHECKING, List, overload

from ice.llutil.argparser import args, as_dict, as_list, get_hostname, isa
from ice.llutil.collections import ConfigDict, Dict
from ice.llutil.config import (Configurable, clone, configurable, freeze,
                               frozen, is_configurable, make_configurable)
from ice.llutil.debug import set_trace
from ice.llutil.launcher import ElasticLauncher, get_current_launcher
from ice.llutil.print import _print as print

try:
    from ice.llutil.pycuda import CUDAModule
except ImportError:
    pass

from ice.core.graph import ExecutableGraph, GraphOutputCache, Node
from ice.core.hypergraph import HyperGraph, Repeat, Task
from ice.llutil.dictprocess import dictprocess

if TYPE_CHECKING:

    from typing import (Callable, Dict, Iterator, List, Optional, Union,
                        overload)

    from ice.core.graph import Node
    from ice.llutil.argparser import as_dict, as_list
    from ice.llutil.config import freeze
    from ice.llutil.dictprocess import DictProcessor
    from torch.utils.data import Dataset


from ice.core.dataset import DatasetNode
from ice.core.loss import LossNode
from ice.core.metric import (AverageMeter, DictMetric, Meter, MetricNode,
                             MovingAverageMeter, SummationMeter, ValueMeter)
from ice.core.module import ModuleNode
from ice.core.optim import Optimizer

from .utils import *
from ice.llutil.utils import *
from .hypergraph import *
