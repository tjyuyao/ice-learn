#!/usr/bin/env python3

import logging
import os
import time
from typing import List
import uuid
import random

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.launcher.api import LaunchConfig, launch_agent
from ice.llutil.config import Configurable

from ice.llutil.logging import get_logger


log = get_logger()


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def parse_devices_and_backend(devices: str, dist_backend: str):

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
        out = [torch.device(device_type)]
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


def _wrap(launcher:"ElasticLauncher", entrypoint, *args):
    dist.init_process_group(
        backend=launcher.dist_backend,
        rank=launcher.rank,
        world_size=launcher.world_size,
    )
    entrypoint(*args)
    dist.destroy_process_group()


class ElasticLauncher(Configurable):
    """
    
    **Example:**

    ```python
    def run(args):
        local_rank = int(os.environ["LOCAL_RANK"])
        print(local_rank, args)


    if __name__ == "__main__":
        launch = ElasticLauncher()
        launch['nproc_per_node'] = 2
        launch.freeze()
        launch(run, "blabla")
    ```

    """
    def __freeze__(self,

        # Worker/node size related arguments.
        devices="auto",  # devices per node, e.g.: ["auto", "cpu", "cuda", "cuda:0", "cuda:*", "auto:*", "cuda:1,3", "cuda:0-2,7"]
        nnodes="1:1",  # Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.
        dist_backend="auto",  # supports: ["nccl", "gloo", "mpi", "auto"]. If given "auto", will use "nccl" for "cuda" and "gloo" for "cpu" in general.

        # Rendezvous related arguments
        rdzv_id="none",  # User-defined group id.
        rdzv_endpoint="",  # Rendezvous backend endpoint; usually in form <host>:<port>.
        rdzv_backend="static",  # Rendezvous backend.
        rdzv_configs="",  # Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).
        standalone=False,  # Start a local standalone rendezvous backend that is represented by a C10d TCP store on port 29400. Useful when launching single-node, multi-worker job. If specified rdzv_backend, rdzv_endpoint, rdzv_id are auto-assigned; any explicitly set values are ignored.

        # User-code launch related arguments.
        max_restarts=0,  # Maximum number of worker group restarts before failing.
        monitor_interval=5,  # Interval, in seconds, to monitor the state of workers.
        start_method="spawn",  # Multiprocessing start method to use when creating workers.
        redirects="0",  # Redirect std streams into a log file in the log directory (e.g. `3` redirects both stdout+stderr for all workers, `0:1,1:2` redirects stdout for local rank 0 and stderr for local rank 1).
        tee="0",  # Tee std streams into a log file and also to console (see redirects for format).
        log_dir=None,  # Base directory to use for log files (e.g. /var/log/torch/elastic). The same directory is re-used for multiple runs (a unique job-level sub-directory is created with rdzv_id as the prefix).
        role="default",  # "User-defined role for the workers."

        # Backwards compatible parameters with caffe2.distributed.launch.
        node_rank=0, #  "Rank of the node for multi-node distributed training.")
        master_addr="127.0.0.1",  # Address of the master node (rank 0). It should be either the IP address or the hostname of rank 0. For single node multi-proc training the master_addr can simply be 127.0.0.1; IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.")
        master_port=None,  # "Port on the master node (rank 0) to be used for communication during distributed training.")

        omp_num_threads = 1,
    ):

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

        min_nodes, max_nodes = parse_min_max_nnodes(nnodes)
        assert 0 < min_nodes <= max_nodes
        assert max_restarts >= 0

        self._devices, self._dist_backend = parse_devices_and_backend(devices, dist_backend)

        nproc_per_node = len(self._devices)
        logging.info(f"Using nproc_per_node={nproc_per_node}.")

        if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
            log.warning(
                f"\n***********************************************************************************\n"
                  f"Setting OMP_NUM_THREADS for each process to be {omp_num_threads} in default, to avoid your system \n"
                  f"being overloaded, this is often not optimal, please consider tuning it.\n"
                  f"***********************************************************************************"
            )
            # This env variable will be passed down to the subprocesses
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        elif "OMP_NUM_THREADS" in os.environ:
            omp_num_threads = os.environ["OMP_NUM_THREADS"]

        rdzv_configs = _parse_rendezvous_config(rdzv_configs)

        if rdzv_backend == "static":
            rdzv_configs["rank"] = node_rank

        if master_port is None:
            seed = time.time_ns() if rdzv_id == "none" else rdzv_id
            master_port = random.Random(seed).randint(16894, 17194)

        if rdzv_backend == "static" and not rdzv_endpoint:
            rdzv_endpoint = f"{master_addr}:{master_port}"
        else:
            rdzv_endpoint = rdzv_endpoint

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

    @record
    def __call__(self, entrypoint, *args):
        args = [self, entrypoint] + list(args)
        launch_agent(self.config, _wrap, list(args))

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