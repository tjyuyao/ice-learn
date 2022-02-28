from typing import List, Union, overload

import torch
from ice.core.graph import Node
from ice.core.hypergraph import HyperGraph, LoadCheckpointTask, SaveCheckpointTask
from ice.llutil.launcher import ElasticLauncher
from torch.cuda.amp.grad_scaler import GradScaler


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
):    ...

@overload
def run(
    tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
    rdzv_endpoint="", rdzv_backend="static", rdzv_configs="", standalone=False,
    redirects="0", tee="0", out_dir=None, resume_from=None, seed=0,
    role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
):    ...

def run(*args, **kwds):
    default_graph.run(*args, **kwds)


def add(name, node:Node, tags="*"):
    default_graph.add(name, node, tags)


@overload
def print_forward_output(*nodenames, every=1, total=None, tags:List[str] = "*", train_only=True, localrank0_only=True): ...

def print_forward_output(*args, **kwds):
    default_graph.print_forward_output(*args, **kwds)


@overload
def init_grad_scaler(grad_scaler: Union[bool, GradScaler] = True,
                     *,
                     init_scale=2.**16,
                     growth_factor=2.0,
                     backoff_factor=0.5,
                     growth_interval=2000,
                     enabled=True):
    ...

def init_grad_scaler(*args, **kwds):
    default_graph.init_grad_scaler(*args, **kwds)


@overload
def init_autocast(autocast_enabled=True,
                  autocast_dtype=torch.float16,
                  grad_scaler: Union[bool, GradScaler] = None):
    ...

def init_autocast(*args, **kwds):
    default_graph.init_autocast(*args, **kwds)

def set_gradient_accumulate(every):
    default_graph.set_gradient_accumulate(every)
    
def backup_source_files(entrypoint:str):
    default_graph.backup_source_files(entrypoint)