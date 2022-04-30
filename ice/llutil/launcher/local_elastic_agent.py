#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import signal
import sys
import threading
import time
import tempfile
from typing import Any, Dict, Optional, Tuple

from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.metrics.api import prof, put_metric
from torch.distributed.elastic.utils import macros
from ice.llutil.logger import get_logger
from tqdm import tqdm
from .elastic_multiprocessing import PContext, start_processes, SignalException

from .events import Events
import ice.llutil.shadow_tb as shadow_tb


DEFAULT_ROLE = "default"
log = get_logger()


class LocalElasticAgent(SimpleElasticAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.


    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    Example launching function

    ::

        def trainer(args) -> str:
            return "do train"

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint=trainer,
                        args=("foobar",),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            results = agent.run()

            if results.is_failed():
                print("trainer failed")
            else:
                print(f"rank 0 return value: {results.return_values[0]}")
                # prints -> rank 0 return value: do train

    Example launching binary

    ::

        def main():
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint="/usr/local/bin/trainer",
                        args=("--trainer_args", "foobar"),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec)
            results = agent.run()

            if not results.is_failed():
                print("binary launches do not have return values")

    """

    def __init__(
        self,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
        events: Events = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        rdzv_run_id = spec.rdzv_handler.get_run_id()
        self._log_dir = self._make_log_dir(log_dir, rdzv_run_id)
        self.events: Events = events
        self.progbar_events = {
            "trigger_clear" : threading.Event(),
            "cleared" : threading.Event(),
        }

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        dir = log_dir or tempfile.mkdtemp(prefix=f"{rdzv_run_id}_")
        os.makedirs(dir, exist_ok=True)
        log.info(f"log directory set to: {dir}")
        return dir

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        self._shutdown()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": str(1),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        assert spec.entrypoint is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
            progbar_events=self.progbar_events,
        )

        return self._pcontext.pids()

    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None:
        if self._pcontext:
            self._pcontext.close(death_sig)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        role = worker_group.spec.role
        worker_pids = {w.id for w in worker_group.workers}
        assert self._pcontext is not None
        pc_pids = set(self._pcontext.pids().values())
        if worker_pids != pc_pids:
            log.error(
                f"[{role}] worker pids do not match process_context pids."
                f" Expected: {worker_pids}, actual: {pc_pids}"
            )
            return RunResult(state=WorkerState.UNKNOWN)

        result = self._pcontext.wait(0)
        if result:
            if result.is_failed():
                # map local rank failure to global rank
                worker_failures = {}
                for local_rank, failure in result.failures.items():
                    worker = worker_group.workers[local_rank]
                    worker_failures[worker.global_rank] = failure
                return RunResult(
                    state=WorkerState.FAILED,
                    failures=worker_failures,
                )
            else:
                # copy ret_val_queue into a map with a global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                )
        else:
            return RunResult(state=WorkerState.HEALTHY)
        
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        spec = self._worker_group.spec
        monitor_interval = spec.monitor_interval
        role = spec.role
        log.info(f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()}")
        self._initialize_workers(self._worker_group)

        try:
            prog_iter = -1
            prog_total = -1
            bar:tqdm = None
            while True:
                self.progbar_events["cleared"].clear()
                if self.progbar_events["trigger_clear"].is_set():
                    if bar is not None: bar.clear()
                    self.progbar_events["cleared"].set()
                    while self.progbar_events["trigger_clear"].is_set():
                        time.sleep(0.001)
                if prog_total != self.events.progress_bar_total.value or \
                    prog_iter > self.events.progress_bar_iter.value:
                    prog_total = self.events.progress_bar_total.value
                    if bar is not None:
                        bar.clear()
                        bar.close()
                    bar = tqdm(total=prog_total, leave=False, position=0, ncols=60, smoothing=0.9)
                if prog_iter != self.events.progress_bar_iter.value:
                    prog_iter = self.events.progress_bar_iter.value
                    if bar is not None:
                        bar.n = prog_iter
                        bar.refresh()
            
                time.sleep(monitor_interval)

                result = self._invoke_monitor_once(role)
                
                if result is not None:
                    if bar is not None: bar.close()
                    break
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except SignalException as e:
            if shadow_tb.DEBUG_ICE:
                raise
            self.events.resume.clear()
            self.events.pause.set()
            self.events.paused.wait()
            sys.stdout.flush()
            sys.stderr.flush()
            log.warning(f"Received {e.sigval.name} death signal, gracefully shutting down workers.")
            self.events.stop_all_tasks.set()
            if e.sigval == signal.SIGINT:
                while True:
                    yn = input("\nShould I save the checkpoint? [Y/n]").lower()
                    if yn in ("y", ""):
                        self.events.finished_save_checkpoint.clear()
                        self.events.trigger_save_checkpoint.set()
                        self.events.pause.clear()
                        self.events.resume.set()
                        self.events.finished_save_checkpoint.wait()
                        break
                    elif yn == "n":
                        self.events.pause.clear()
                        self.events.resume.set()
                        break
                    else:
                        continue
            else:
                self.events.pause.clear()
                self.events.resume.set()                
            raise
        finally:
            if not shutdown_called:
                self._shutdown()
            # record the execution time in case there were any exceptions during run.
            self._total_execution_time = int(time.monotonic() - start_time)
            
    def _invoke_monitor_once(self, role: str = DEFAULT_ROLE) -> RunResult:
        spec = self._worker_group.spec
        role = spec.role

        rdzv_handler = spec.rdzv_handler

        assert self._worker_group.state != WorkerState.INIT
        run_result = self._monitor_workers(self._worker_group)
        state = run_result.state
        self._worker_group.state = state

        put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
        put_metric(f"workers.{role}.{state.name.lower()}", 1)

        if state == WorkerState.SUCCEEDED:
            log.info(
                f"[{role}] worker group successfully finished."
                f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
            )
            self._exit_barrier()
            return run_result
        elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
            if self._remaining_restarts > 0:
                log.info(
                    f"[{role}] Worker group {state.name}. "
                    f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                    f" will restart worker group"
                )
                self._remaining_restarts -= 1
                self._restart_workers(self._worker_group)
            else:
                self._stop_workers(self._worker_group)
                self._worker_group.state = WorkerState.FAILED
                self._exit_barrier()
                return run_result
        elif state == WorkerState.HEALTHY:
            # membership changes do not count as retries
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
            group_rank = self._worker_group.group_rank
            if num_nodes_waiting > 0:
                log.info(
                    f"[{role}] Detected {num_nodes_waiting} "
                    f"new nodes from group_rank={group_rank}; "
                    f"will restart worker group"
                )
                # TODO: trigger save&load checkpoint
                self._restart_workers(self._worker_group)
        else:
            raise Exception(f"[{role}] Worker group in {state.name} state")