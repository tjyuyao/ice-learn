#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import random
import signal
import time
import uuid
from types import FrameType
from typing import TYPE_CHECKING, List, overload

from ice.llutil.config import Configurable
from ice.llutil.ignore_me import IgnoreMe
from ice.llutil.logger import get_logger

from .events import Events

if TYPE_CHECKING:
    import torch


log = get_logger()


def _parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def _parse_devices_and_backend(devices: str = "auto", dist_backend: str = "auto"):
    import torch

    # determine device type
    if devices[:3] == "cpu" or (
        devices[:4] == "auto" and not torch.cuda.is_available()
    ):
        device_type = "cpu"
        if dist_backend == "auto":
            dist_backend = "gloo"
    elif devices[:4] == "cuda" or (
        devices[:4] == "auto" and torch.cuda.is_available()
    ):
        if not torch.cuda.is_available():
            raise ValueError("Cuda is not available.")
        device_type = "cuda"
        if dist_backend == "auto":
            dist_backend = "nccl"
    else:
        raise ValueError(f"Unsupported devices value: {devices}")

    # determine device indices
    idid = devices.find(":")
    if -1 == idid:
        out = [torch.device(device_type, 0)]
    else:
        out = []
        for indices in devices[idid+1:].split(","):
            if -1 != indices.find("-"):
                s, e = indices.split("-")
                for i in range(int(s), int(e) + 1):
                    out.append(torch.device(device_type, i))
            elif indices == "*":
                if device_type == "cuda":
                    device_count = torch.cuda.device_count()
                elif device_type == "cpu":
                    device_count = os.cpu_count()
                else:
                    assert False
                for i in range(device_count):
                    out.append(torch.device(device_type, i))
            else:
                out.append(torch.device(device_type, int(indices)))
        if 0 == len(out):
            raise ValueError("Empty devices indices.")
        if dist_backend == "nccl" and len(set(out)) != len(out):
            dist_backend = "gloo"
    return out, dist_backend


def _ignore_sigint(signum: int, frame: FrameType) -> None: ...


class EagerLauncher:

    def __init__(self) -> None:
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.events = IgnoreMe()

        import torch
        
        if torch.cuda.is_available():
            self.assigned_device = torch.device("cuda", 0)
        else:
            self.assigned_device = torch.device("cpu", 0)

        self.eager_mode = True


_current_launcher = EagerLauncher()

def get_current_launcher() -> "ElasticLauncher":
    global _current_launcher
    return _current_launcher

def _wrap(launcher:"ElasticLauncher", entrypoint, *args):
    import ice.llutil.shadow_tb as shadow_tb
    import torch
    import torch.distributed as dist

    global _current_launcher
    _current_launcher = launcher
    if not shadow_tb.DEBUG_ICE:
        signal.signal(signal.SIGINT, _ignore_sigint)
    dist.init_process_group(
        backend=launcher.dist_backend,
        rank=launcher.rank,
        world_size=launcher.world_size,
    )
    if launcher.assigned_device.type == "cuda":
        torch.cuda.set_device(launcher.assigned_device)
    try:
        entrypoint(*args)
    except Exception as e:
        if launcher.local_rank == 0:
            shadow_tb.shadow(type(e), e, e.__traceback__)
            time.sleep(1.)
        raise
    time.sleep(launcher.config.monitor_interval + 1.)
    _current_launcher = None
    # dist.destroy_process_group()


class ElasticLauncher(Configurable):
    """A helper ``Configurable`` class for `torchrun` and `torch.distributed.launch`.

    PyTorch's elastic launch ability is embeded in this Configurable, for details please see [here](https://pytorch.org/docs/stable/elastic/run.html).

    ``HyperGraph.run()`` uses this class to launch multiple processes. Directly usage is also possible (see the example below).

    **Example:**

    ```python
    def worker(launcher):
        print("rank", launcher.rank)
        print("local_rank", launcher.local_rank)
        print("device", launcher.assigned_device)


    if __name__ == "__main__":
        launcher = ElasticLauncher("cuda:*").freeze()
        launcher(worker, launcher)
    ```
    """

    @overload
    def __init__(self,

        # Worker/node size related arguments.
        devices="auto",
        nnodes="1:1",
        dist_backend="auto",

        # Rendezvous related arguments
        rdzv_id="none",
        rdzv_endpoint="",
        rdzv_backend="static",
        rdzv_configs="",
        standalone=False,

        # User-code launch related arguments.
        max_restarts=0,
        monitor_interval=5,
        start_method="spawn",
        redirects="2",
        tee="3",
        log_dir=None,
        role="default",

        # Backwards compatible parameters with caffe2.distributed.launch.
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=None,

        omp_num_threads = 1,
        
        events:Events = None,
    ): 
        """

        **Args:**
        - **Worker/node size related arguments:**
            - **`devices`** (str, optional): enumerates devices on this node, e.g.: [`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:*"`, `"auto:*"`, `"cuda:1,3"`, `"cuda:0-2,7"`]. Defaults to `"auto"`.
            - **`dist_backend`** (str, optional): supports: [`"nccl"`, `"gloo"`, `"mpi"`, `"auto"`]. If given `"auto"`, will use `"nccl"` for `"cuda"` and `"gloo"` for `"cpu"` in general. Defaults to `"auto"`.
            - **`nnodes`** (str, optional): Number of nodes, or the range of nodes in form `<minimum_nodes>:<maximum_nodes>`
            . Defaults to `"1:1"`.
        - **Rendezvous related arguments:**
            - **`rdzv_id`** (str, optional): User-defined group id.
            - **`rdzv_endpoint`** (str, optional): Rendezvous backend endpoint; usually in form `<host>:<port>`.
            - **`rdzv_backend`** (str, optional): Rendezvous backend.
            - **`rdzv_configs`** (str, optional): Additional rendezvous configuration (`<key1>=<value1>,<key2>=<value2>,...`).
            - **`standalone`** (bool, optional): Start a local standalone rendezvous backend that is represented by a C10d TCP store on port 29400. Useful when launching single-node, multi-worker job. If specified rdzv_backend, rdzv_endpoint, rdzv_id are auto-assigned; any explicitly set values are ignored. Defaults to `False`.
        - **User-code launch related arguments:**
            - **`max_restarts`** (int, optional): Maximum number of worker group restarts before failing. Defaults to 0.
            - **`monitor_interval`** (int, optional): Interval, in seconds, to monitor the state of workers. Defaults to 5.
            - **`start_method`** (str, optional): Multiprocessing start method to use when creating workers. Defaults to `"spawn"`.
            - **`redirects`** (str, optional): Redirect std streams into a log file in the log directory (e.g. `3` redirects both stdout+stderr for all workers, `0:1,1:2` redirects stdout for local rank 0 and stderr for local rank 1). Defaults to `"0"`.
            - **`tee`** (str, optional): Tee std streams into a log file and also to console (see redirects for format). Defaults to `"0"`.
            - **`log_dir`** ([type], optional): Base directory to use for log files (e.g. /var/log/torch/elastic). The same directory is re-used for multiple runs (a unique job-level sub-directory is created with rdzv_id as the prefix). Defaults to None.
            - **`role`** (str, optional): User-defined role for the workers. Defaults to `"default"`.
        - **Backwards compatible parameters with `caffe2.distributed.launch`:**
            - **`node_rank`** (int, optional): "Rank of the node for multi-node distributed training."). Defaults to 0.
            - **`master_addr`** (str, optional): Address of the master node (rank 0). It should be either the IP address or the hostname of rank 0. For single node multi-proc training the master_addr can simply be 127.0.0.1; IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.") Defaults to "127.0.0.1".
            - **`master_port`** ([type], optional): Port on the master node (rank 0) to be used for communication during distributed training. Defaults will generate a random port between `16894` and `17194`.
            - **`omp_num_threads`** (int, optional): set `OMP_NUM_THREADS` environment if not exists. Defaults to 1.
        """

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(self,

        # Worker/node size related arguments.
        devices="auto",
        nnodes="1:1",
        dist_backend="auto",

        # Rendezvous related arguments
        rdzv_id="none",
        rdzv_endpoint="",
        rdzv_backend="static",
        rdzv_configs="",
        standalone=False,

        # User-code launch related arguments.
        max_restarts=0,
        monitor_interval=None,
        start_method="spawn",
        redirects="2",
        tee="3",
        log_dir=None,
        role="default",

        # Backwards compatible parameters with caffe2.distributed.launch.
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=None,

        omp_num_threads = None,
        
        events:Events = None,
    ):

        from torch.distributed.elastic.multiprocessing import Std
        from torch.distributed.elastic.rendezvous.utils import \
            _parse_rendezvous_config

        from .launch_agent import LaunchConfig

        if standalone:
            rdzv_backend = "c10d"
            rdzv_endpoint = "localhost:29400"
            rdzv_id = str(uuid.uuid4())
            log.info(
                f"\n**************************************\n"
                f"Rendezvous info:\n"
                f"--rdzv_backend={rdzv_backend} "
                f"--rdzv_endpoint={rdzv_endpoint} "
                f"--rdzv_id={rdzv_id}\n"
                f"**************************************\n"
            )

        min_nodes, max_nodes = _parse_min_max_nnodes(nnodes)
        assert 0 < min_nodes <= max_nodes
        assert max_restarts >= 0

        self._devices, self._dist_backend = _parse_devices_and_backend(devices, dist_backend)

        nproc_per_node = len(self._devices)
        logging.info(f"Using nproc_per_node={nproc_per_node}.")

        if "OMP_NUM_THREADS" not in os.environ and omp_num_threads is None and nproc_per_node > 1:
            os.environ["OMP_NUM_THREADS"] = '1'
        elif "OMP_NUM_THREADS" not in os.environ and omp_num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        elif "OMP_NUM_THREADS" in os.environ and omp_num_threads is None:
            omp_num_threads = os.environ["OMP_NUM_THREADS"]
        elif "OMP_NUM_THREADS" in os.environ and omp_num_threads is not None:
            if int(os.environ["OMP_NUM_THREADS"]) != int(omp_num_threads):
                raise ValueError(f'os.environ["OMP_NUM_THREADS"] set to {os.environ["OMP_NUM_THREADS"]} but omp_num_threads set to {omp_num_threads}.')

        rdzv_configs = _parse_rendezvous_config(rdzv_configs)

        if rdzv_backend == "static":
            rdzv_configs["rank"] = node_rank

        if master_port is None:
            seed = time.time_ns() if max_nodes == 1 else rdzv_id
            master_port = random.Random(seed).randint(16894, 17194)

        if rdzv_backend == "static" and not rdzv_endpoint:
            rdzv_endpoint = f"{master_addr}:{master_port}"
        else:
            rdzv_endpoint = rdzv_endpoint
            
        if monitor_interval is None:
            if nproc_per_node == 1 and max_nodes == 1:
                monitor_interval = 1  # for faster debugging
            else:
                monitor_interval = 5

        self.config = LaunchConfig(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            nproc_per_node=nproc_per_node,
            run_id=rdzv_id,
            role=role,
            rdzv_endpoint=rdzv_endpoint,
            rdzv_backend=rdzv_backend,
            rdzv_configs=rdzv_configs,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
            start_method=start_method,
            redirects=Std.from_str(redirects),
            tee=Std.from_str(tee),
            log_dir=log_dir,
        )
        
        self.events = events
        self.eager_mode = False

    def __call__(self, entrypoint, *args):
        from torch.distributed.elastic.multiprocessing.errors import (
            ChildFailedError, record)

        from .launch_agent import launch_agent
        args = [self, entrypoint] + list(args)
        try:
            record(launch_agent)(self.config, _wrap, list(args), self.events)
        except ChildFailedError as e:
            raise

    @property
    def devices(self) -> List[torch.device]:
        return self._devices

    @property
    def dist_backend(self):
        return self._dist_backend

    #
    # following properties should be called from the subprocesses, you can pass the launcher as argument for custom entrypoint function.
    #

    @property
    def assigned_device(self) -> torch.device:
        return self.devices[self.local_rank]

    @property
    def local_rank(self):
        return int(os.environ["LOCAL_RANK"])

    @property
    def rank(self):
        return int(os.environ["RANK"])

    @property
    def group_rank(self):
        return int(os.environ["GROUP_RANK"])

    @property
    def role_rank(self):
        return int(os.environ["ROLE_RANK"])

    @property
    def role_name(self):
        return os.environ["ROLE_NAME"]

    @property
    def local_world_size(self):
        return int(os.environ["LOCAL_WORLD_SIZE"])

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    @property
    def group_world_size(self):
        return int(os.environ["GROUP_WORLD_SIZE"])

    @property
    def role_world_size(self):
        return int(os.environ["ROLE_WORLD_SIZE"])

    @property
    def master_addr(self):
        return os.environ["MASTER_ADDR"]

    @property
    def master_port(self):
        return os.environ["MASTER_PORT"]

    @property
    def restart_count(self):
        return int(os.environ["TORCHELASTIC_RESTART_COUNT"])

    @property
    def max_restarts(self):
        return int(os.environ["TORCHELASTIC_MAX_RESTARTS"])

    @property
    def rdzv_id(self):
        return os.environ["TORCHELASTIC_RUN_ID"]
