from __future__ import annotations
from typing import List, Union, overload, TYPE_CHECKING

from ice.core.graph import Node

if TYPE_CHECKING:
    from ice.llutil.launcher import ElasticLauncher
    from torch.cuda.amp.grad_scaler import GradScaler

def LoadCheckpointTask():
    from ice.core import hypergraph
    return hypergraph.LoadCheckpointTask()

def SaveCheckpointTask():
    from ice.core import hypergraph
    return hypergraph.SaveCheckpointTask()

_DEFAULT_GRAPH = None

def get_default_graph():
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        from ice.core.hypergraph import HyperGraph
        _DEFAULT_GRAPH = HyperGraph()
    return _DEFAULT_GRAPH

@overload
def run(tasks, devices="auto", run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

@overload
def run(tasks, launcher:ElasticLauncher=None, run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

@overload
def run(
    tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
    node_rank=0, master_addr="127.0.0.1", master_port=None,
    redirects="2", tee="1", out_dir=None, resume_from=None, seed=0,
    role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
):    ...

@overload
def run(
    tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
    rdzv_endpoint="", rdzv_backend="static", rdzv_configs="", standalone=False,
    redirects="2", tee="3", out_dir=None, resume_from=None, seed=0,
    role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
):    ...

def run(*args, **kwds):
    get_default_graph().run(*args, **kwds)

def add(name, node:Node, tags="*"):
    get_default_graph().add(name, node, tags)


@overload
def print_forward_output(*nodenames, every=1, total=None, tags:List[str] = "*", train_only=True, localrank0_only=True): ...

def print_forward_output(*args, **kwds):
    get_default_graph().print_forward_output(*args, **kwds)


@overload
def init_grad_scaler(grad_scaler: Union[bool, "GradScaler"] = True,
                     *,
                     init_scale=2.**16,
                     growth_factor=2.0,
                     backoff_factor=0.5,
                     growth_interval=2000,
                     enabled=True):
    ...

def init_grad_scaler(*args, **kwds):
    get_default_graph().init_grad_scaler(*args, **kwds)


@overload
def init_autocast(autocast_enabled=True,
                  autocast_dtype="torch.float16",
                  grad_scaler: Union[bool, "GradScaler"] = None):
    ...

def init_autocast(*args, **kwds):
    get_default_graph().init_autocast(*args, **kwds)

def set_gradient_accumulate(every):
    get_default_graph().set_gradient_accumulate(every)
    
def backup_source_files(entrypoint:str):
    get_default_graph().backup_source_files(entrypoint)