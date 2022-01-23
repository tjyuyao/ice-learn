#!/usr/bin/env python3

import logging
import os
import time
import uuid
import random

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from ice.llutil.config import configurable


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


def determine_local_world_size(nproc_per_node: str):
    try:
        logging.info(f"Using nproc_per_node={nproc_per_node}.")
        return int(nproc_per_node)
    except ValueError:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.")
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "auto":
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = "gpu"
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}")

        log.info(
            f"Using nproc_per_node={nproc_per_node},"
            f" seting to {num_proc} since the instance "
            f"has {os.cpu_count()} {device_type}"
        )
        return num_proc


@configurable
class ElasticLauncher():
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
    def __init__(self,
        # Worker/node size related arguments.
        nnodes="1:1",  # Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.
        nproc_per_node="1",  # Number of workers per node; supported values: [auto, cpu, gpu, int]

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

        nproc_per_node = determine_local_world_size(nproc_per_node)
        if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
            log.warning(
                f"\n***********************************************************************************\n"
                  f"Setting OMP_NUM_THREADS for each process to be {omp_num_threads} in default, to avoid your system \n"
                  f"being overloaded, this is often not the optimal setting, please consider tuning it.\n"
                  f"***********************************************************************************"
            )
            # This env variable will be passed down to the subprocesses
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        else:
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
        launch_agent(self.config, entrypoint, list(args))
