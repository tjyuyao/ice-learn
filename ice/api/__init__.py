from typing import List, overload
from ice.llutil import test
from ice.llutil.collections import Dict, ConfigDict
from ice.llutil.argparser import args, as_dict, as_list, isa, get_hostname
from ice.llutil.config import (clone, configurable, Configurable, freeze, is_configurable,
                               make_configurable, frozen)
from ice.llutil.debug import set_trace
from ice.llutil.launcher import ElasticLauncher
from ice.llutil.print import _print as print

from ice.llutil.pycuda import CUDAModule
from ice.llutil.dictprocess import dictprocess
from ice.llutil.multiprocessing import in_main_process

from ice.core.graph import Node, ExecutableGraph
from ice.core.hypergraph import HyperGraph, Task, Repeat
from ice.core.dataset import DatasetNode
from ice.core.optim import Optimizer
from ice.core.module import ModuleNode
from ice.core.loss import LossNode
from ice.core.metric import Meter, DictMetric, ValueMeter, SummationMeter, AverageMeter, MovingAverageMeter, MetricNode


default_graph = HyperGraph()

@overload
def run(tasks, devices="auto", run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

@overload
def run(tasks, launcher:ElasticLauncher=None, run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

@overload
def run(
    tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
    node_rank=0, master_addr="127.0.0.1", master_port=None,
    redirects="0", tee="0", out_dir=None, resume_from=None, seed=0,
    role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
): ...

@overload
def run(
    tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
    rdzv_endpoint="", rdzv_backend="static", rdzv_configs="", standalone=False,
    redirects="0", tee="0", out_dir=None, resume_from=None, seed=0,
    role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
): ...

def run(*args, **kwds):
    default_graph.run(*args, **kwds)

def add(name, node:Node, tags="*"):
    default_graph.add(name, node, tags)

@overload
def print_forward_output(*nodenames, every=1, total=None, tags:List[str] = "*", train_only=True, localrank0_only=True): ...

def print_forward_output(*args, **kwds):
    default_graph.print_forward_output(*args, **kwds)