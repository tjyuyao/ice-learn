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


@overload
def DatasetNode(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    batch_size_in_total: bool = False,
    num_iters_per_epoch: int = None,
    prefetch_factor: int = 2,
    worker_init_fn: Optional[Callable] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    pipeline: Union[DictProcessor, List[DictProcessor]] = None,
):
    ...

def DatasetNode(*args, **kwds):
    from ice.core import dataset
    return dataset.DatasetNode(*args, **kwds)



from ice.core.loss import LossNode
from ice.core.metric import (AverageMeter, DictMetric, Meter, MetricNode,
                             MovingAverageMeter, SummationMeter, ValueMeter)
from ice.core.module import ModuleNode
from ice.core.optim import Optimizer

from .utils import *
from ice.llutil.utils import *
from .hypergraph import *
