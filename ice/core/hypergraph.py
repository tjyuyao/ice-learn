from argparse import Namespace
import os
import random
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
import time
from typing import Dict, List, Optional, overload
import numpy as np

import torch.cuda
from ice.core.graph import (ExecutableGraph, GraphOutputCache, InvalidURIError,
                            Node, StopAllTasks, StopTask)
from ice.llutil.argparser import as_list, isa
from ice.llutil.collections import Dict as iDict
from ice.llutil.config import Configurable, frozen, is_configurable
from ice.llutil.launcher import ElasticLauncher, Events, global_shared_events
from ice.llutil.logging import get_logger
from ice.llutil.multiprocessing import in_main_process
from ice.llutil.print import _print
from torch.autograd.grad_mode import set_grad_enabled


class ResumeTaskFailed(Exception):
    """raised when task structure does not match during resuming."""

class _Task(Configurable):

    def __freeze__(self, *, steps: int = 0, epochs: int = 0):
        self.total_steps = steps
        self.total_epochs = epochs

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, _state_dict, strict):
        raise NotImplementedError()


class Task(_Task):

    @overload
    def __init__(self, *, train: bool, steps: int, tags="*"): ...

    @overload
    def __init__(self, *, train: bool, epochs: int, tags="*"): ...

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(self, *, train: bool, tags="*", **kwds):
        super().__freeze__(**kwds)
        assert self.total_epochs == 0 or self.total_steps == 0
        self.training = train
        self.tags = [as_group_name(tag) for tag in as_list(tags)]
        self.step_mode = self.total_epochs == 0
        self.task_steps = 0
        self.task_epochs = 0
        self.finished = False
        return self
    
    def __call__(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
        if self.finished: return  # for resuming progress
        with set_grad_enabled(self.training):
            self.__call__impl(hyper_graph, launcher)
        self.finished = True

    def __call__impl(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
        # maintain running progress.
        self.hyper_graph = hyper_graph

        # prepare states.
        self.launcher = launcher
        self.egraph: ExecutableGraph = hyper_graph[self.tags]
        self.egraph.task = self

        if self.egraph is not hyper_graph.last_executed_egraph:
            if hyper_graph.last_executed_egraph is not None:
                hyper_graph.last_executed_egraph.clean_up_nodes()
            self.egraph.prepare_nodes()
            if self.launcher.assigned_device.type == "cuda":
                torch.cuda.empty_cache() # result in more precise value in `nvidia-smi`.
        hyper_graph.last_executed_egraph = self.egraph

        # run epochs: assert self.total_epochs == 0 or self.total_steps == 0
        if self.total_epochs:
            if self.task_steps != 0: self.global_epochs -= 1  # already started before last interruption
            for self.task_epochs in range(self.task_epochs, self.total_epochs):
                self.egraph.apply("epoch_start")
                self.global_epochs += 1
                while True:
                    try:
                        self.task_steps += 1
                        self._iterate()
                    except StopIteration:
                        self.task_steps = 0
                        break
                    except StopTask: return
                self.egraph.apply("epoch_end")
        else:
            for self.task_steps in range(self.task_steps, self.total_steps):
                try:
                    self._iterate()
                except StopTask: return


    def _iterate(self):
        self.egraph.iterate()
        self._process_events()
        
    def _process_events(self):
        events:Events = self.launcher.events
        if events.pause.is_set():
            events.paused.set()
            events.resume.wait()
            events.paused.clear()
        if events.trigger_save_checkpoint.is_set():
            self.hyper_graph.save_checkpoint()
            if self.launcher.rank == 0:
                events.trigger_save_checkpoint.clear()
                events.finished_save_checkpoint.set()
            else:
                events.finished_save_checkpoint.wait()
        if events.stop_all_tasks.is_set():
            raise StopAllTasks()

    def state_dict(self):
        _state_dict = {
            "total_steps" : self.total_steps,
            "total_epochs" : self.total_epochs,
            "task_steps" : self.task_steps,
            "task_epochs" : self.task_epochs,
            "finished" : self.finished,
        }
        return _state_dict

    def load_state_dict(self, _state_dict, dry_run=False):
        
        if self.total_epochs != _state_dict["total_epochs"] or \
            self.total_steps != _state_dict["total_steps"]:
            raise ResumeTaskFailed()
        
        if not dry_run:
            self.task_steps = _state_dict["task_steps"]
            self.task_epochs = _state_dict["task_epochs"]
            self.finished = _state_dict["finished"]

    @property
    def _train_str(self):
        return "train" if self.training else "eval"

    @property
    def global_epochs(self):
        return self.hyper_graph.global_counters.epochs[self._train_str]

    @global_epochs.setter
    def global_epochs(self, value):
        self.hyper_graph.global_counters.epochs[self._train_str] = value
        return value

    @property
    def global_steps(self):
        return self.hyper_graph.global_counters.steps[self._train_str]

    @global_steps.setter
    def global_steps(self, value):
        self.hyper_graph.global_counters.steps[self._train_str] = value
        return value

class Repeat(_Task):

    @overload
    def __init__(self, tasks:List[_Task], times:int) -> None: ...

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(self, tasks:List[_Task], times:int) -> None:
        self.tasks = as_list(tasks)
        self.repeat = times
        self.etasks:List[Task] = [deepcopy(t) for _ in range(times) for t in self.tasks]
        [t.freeze() for t in self.etasks if is_configurable(t)]
        super().__freeze__(
            steps=sum(t.total_steps for t in self.etasks if isa(t, _Task)),
            epochs=sum(t.total_steps for t in self.etasks if isa(t, _Task)),
        )
        _tags = []
        for task in self.tasks:
            if isa(task, _Task):
                _tags.extend(task.freeze().tags)
        self.tags = list(set(_tags))
        return self

    def __call__(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
        hyper_graph.exec_tasks(self.etasks, launcher)

    def __repr__(self) -> str:
        reprstr = ",".join([repr(subtask) for subtask in self.tasks])
        reprstr = f"[{reprstr}] * {self.repeat}"
        return reprstr
    
    def state_dict(self):
        _state_dict = [t.state_dict() for t in self.etasks if isa(t, _Task)]
        return _state_dict

    def load_state_dict(self, _state_dict, dry_run=False):
        if not isa(_state_dict, list): raise ResumeTaskFailed()
        _etasks = [t for t in self.etasks if isa(t, _Task)]
        if len(_etasks) != len(_state_dict): raise ResumeTaskFailed()
        for t, s in zip(_etasks, _state_dict):
            t.load_state_dict(s, dry_run=dry_run)


class Counter:

    def __init__(self) -> None:
        self.train = 0
        self.eval = 0

    def __getitem__(self, key):
        if key == "train":
            return self.train
        elif key == "eval":
            return self.eval
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "train":
            self.train = value
        elif key == "eval":
            self.eval = value
        else:
            raise KeyError(key)

    @property
    def total(self):
        return self.train + self.eval

@dataclass
class GlobalCounters:
    steps:Counter = Counter()
    epochs:Counter = Counter()

class HyperGraph:
    """HyperGraph is the container for all nodes.
    """

    def __init__(self) -> None:
        self.groups:Dict[str, ExecutableGraph] = {}
        self.groups["*/"] = ExecutableGraph()
        self.shortcuts:Dict[str, ExecutableGraph] = {}
        self.global_counters = GlobalCounters()
        self.last_executed_egraph = None
        self.run_info = iDict()
        self.num_workers = 0

    def add(self, name, node:Node, tags="*"):
        uris = []
        for tag in as_list(tags):
            uris.append(as_group_name(tag) + name)
        self[uris] = node.clone(deepcopy=True)

        if isa(node, Configurable) and not frozen(node) and "num_workers" in node:
            self.num_workers = max(self.num_workers, node["num_workers"])

    def print_forward_output(self, *nodenames, every=1, total=None, tags:List[str] = "*", train_only=True, localrank0_only=True):

        def probe_fn(n:Node, x:GraphOutputCache):
            if train_only and not n.training: return
            if localrank0_only and n.launcher.local_rank != 0: return
            if total is not None and n.global_steps // every > total: return  # total
            if n.global_steps > 1 and n.global_steps % every: return # every
            prefix = f"E{self.global_counters.epochs.train}S{self.global_counters.steps.train}:"

            for nodename in as_list(nodenames):
                _print(x[nodename], prefix=prefix, uri=nodename)

        self.add(f"print_output_of({','.join(as_list(nodenames))})", Node(forward=probe_fn), tags=tags)

    @overload
    def run(self, tasks, devices="auto", run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

    @overload
    def run(self, tasks, launcher:ElasticLauncher=None, run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0): ...

    @overload
    def run(
        self, tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
        node_rank=0, master_addr="127.0.0.1", master_port=None,
        redirects="0", tee="0", out_dir=None, resume_from=None, seed=0,
        role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
    ): ...

    @overload
    def run(
        self, tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
        rdzv_endpoint="", rdzv_backend="static", rdzv_configs="", standalone=False,
        redirects="0", tee="0", out_dir=None, resume_from=None, seed=0,
        role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
    ): ...

    def run(self, tasks, launcher:ElasticLauncher=None, run_id:str="unnamed", out_dir:str=None, resume_from:str=None, seed=0, start_method="spawn", **kwds):
        
        self._set_initial_rng_state(seed)

        if in_main_process():
            self._prepare_out_dir(run_id=run_id, out_dir=out_dir)
            
            kwds["rdzv_id"] = run_id
            kwds["log_dir"] = self.run_info.log_dir
            kwds["start_method"] = start_method
            kwds["events"] = Events(start_method)
            if "omp_num_threads" not in kwds:
                kwds["omp_num_threads"] = self.num_workers + 1

            if launcher is None:
                launcher = ElasticLauncher(**kwds)
            else:
                launcher.update(kwds)
            launcher.freeze()
            launcher(self._run_impl, as_list(tasks), launcher, resume_from)

    def _run_impl(self, tasks, launcher:ElasticLauncher, resume_from):
        self.run_info.tasks = tasks
        self.run_info.launcher = launcher
        global_shared_events["debugger_start"] = launcher.events.debugger_start
        global_shared_events["debugger_end"] = launcher.events.debugger_end
        try:
            self.load_checkpoint(resume_from)
            self.exec_tasks(tasks, launcher)
        except StopAllTasks:
            pass
    
    @property
    def launcher(self) -> ElasticLauncher:
        return self.run_info.launcher
    
    def _set_initial_rng_state(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def _prepare_out_dir(self, run_id:str, out_dir:str=None):
        """

        ```
        - out_dir
            - run_id_{hash}
                - logs
                - ckpts
            - run_id_{new_hash}
                - logs
                - ckpts
            - ...
        ```
        """
        base_out_dir = out_dir or "out"
        os.makedirs(base_out_dir, exist_ok=True)
        run_id_dir = tempfile.mkdtemp(prefix=f"{run_id}_", dir=base_out_dir)
        ckpt_dir = os.path.join(run_id_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        log_dir = os.path.join(run_id_dir, "logs")

        self.run_info.id = run_id
        self.run_info.out_dir = run_id_dir
        self.run_info.ckpt_dir = ckpt_dir
        self.run_info.log_dir = log_dir
        
    def _get_activated_groups(self):

        tags = ["*"]
        for task in self.run_info.tasks:
            if isa(task, _Task):
                tags.extend(task.freeze().tags)
        groups = list(set([as_group_name(tag) for tag in tags]))
        
        return groups


    def save_checkpoint(self, save_to=None):

        if save_to is None:
            fname = f"E{self.global_counters.epochs.train}S{self.global_counters.steps.train}.pth"
            save_to = os.path.join(self.run_info.ckpt_dir, fname)
        
        _last_save_to = getattr(self, "_last_ckpt_save_to", None)
        setattr(self, "_last_ckpt_save_to", save_to)
        if _last_save_to == save_to: return  # in case of repeat saving.
        
        activated_groups = self._get_activated_groups()
        
        _checkpoint = {
            "tasks":[task.state_dict() for task in self.run_info.tasks if isa(task, _Task)],
            "groups":{name:self.groups[name].state_dict() for name in activated_groups},
            "counters": self.global_counters,
        }
        
        if self.launcher.rank == 0:
        
            print(f"Saving checkpoint to \"{save_to}\".")
            torch.save(_checkpoint, save_to)

    def load_checkpoint(self, resume_from, strict=False):
        self.run_info.resume_from = resume_from
        if resume_from is None: return
        _checkpoint = torch.load(resume_from, map_location=self.launcher.assigned_device)
        
        try: # resuming task progress
            _tasks:List[Task] = [task for task in self.run_info.tasks if isa(task, _Task)]
            if len(_tasks) != len(_checkpoint["tasks"]): raise ResumeTaskFailed()
            for t, s in zip(_tasks, _checkpoint["tasks"]):
                t.freeze().load_state_dict(s, dry_run=True)
        except ResumeTaskFailed:
            if self.launcher.local_rank == 0:
                if strict:
                    get_logger().error("Resuming tasks failed, exiting due to the strict policy.")
                    raise StopAllTasks()
                else:
                    get_logger().warn("Resuming tasks failed, will run from start.")
        else:
            for t, s in zip(_tasks, _checkpoint["tasks"]):
                t.load_state_dict(s, dry_run=False)

        dummy_task = Namespace(launcher=self.run_info.launcher, training=True)
        activated_groups = self._get_activated_groups()
        for name, group_states in _checkpoint["groups"].items():
            if name not in activated_groups: continue
            egraph = self.groups[name]
            egraph.task = dummy_task
            egraph.prepare_nodes()
            egraph.load_state_dict(group_states, strict=strict)
            
        self.global_counters = _checkpoint["counters"]
        
        
    def exec_tasks(self, tasks, launcher:ElasticLauncher):

        for task in as_list(tasks):
            if is_configurable(task):
                task.freeze()
            if isa(task, _Task):
                task(self, launcher)
            elif isa(task, callable):
                args = [x for x, _ in zip(
                    [self, launcher],
                    signature(task).parameters
                )]
                task(*args)
            else:
                get_logger().warning(f"A custom task `{task}` is not callable, skipping.")

    def __setitem__(self, key, value):
        # assume ``key`` is a (list of) valid uri, and ``value`` is a node.
        try:
            for uri in as_list(key):
                group_name, node_name = self._parse_uri(uri)
                if not group_name in self.groups:
                    self.groups[group_name] = ExecutableGraph()
                self.groups[group_name].add_node(node_name, value, group_name)
            return value
        except InvalidURIError: pass

        # assume ``key`` is group_name and ``value`` is an ExecutableGraph.
        assert isa(key, str) and isa(value, ExecutableGraph)
        self.groups[as_group_name(key)] = value

    def __getitem__(self, key):
        # assume ``key`` is a valid uri.
        try:
            return self._get_node_by_uri(key)
        except InvalidURIError: pass

        # assume ``key`` is a (list of) group_name.
        group_names = ['*/'] + [n.rstrip('/') + '/' for n in as_list(key)]
        shortcut_key = hash(tuple(group_names))
        if shortcut_key not in self.shortcuts:
            egraph = ExecutableGraph()
            for group_name in group_names:
                if group_name not in self.groups:
                    raise KeyError(f"`{group_name}` is not a valid group name or a valid node uri.")
                group = self.groups[group_name]
                for node_name, node in group.items():
                    egraph.add_node(node_name, node, group_name)
            self.shortcuts[shortcut_key] = egraph
        else:
            egraph = self.shortcuts[shortcut_key]
        return egraph

    def __contains__(self, name):
        name = name.rstrip("/")
        as_group_name = name + '/'
        try:
            return self._has_node_by_uri(name)
        except InvalidURIError:
            pass
        for group_name in self.groups.keys():
            if group_name.startswith(as_group_name):
                return True
        for group in self.groups.values():
            if name in group:
                return True
        return False

    @staticmethod
    def _parse_uri(uri):
        """Parse a uri and return group_name and node_name.

        Args:
            uri (str): "{group_name}/{node_name}"

        Returns:
            (str, str): (group_name + "/", node_name)
        """
        if not isa(uri, str): raise InvalidURIError(uri)
        idns = uri.rfind("/") + 1
        if 0 == idns:
            group_name, node_name = '*/', uri
        else:
            group_name, node_name = uri[:idns], uri[idns:]
        if 0 == len(node_name): raise InvalidURIError(uri)
        return group_name, node_name

    def _get_node_by_uri(self, uri):
        group_name, node_name = self._parse_uri(uri)
        try: return self.groups[group_name][node_name]
        except KeyError: pass
        raise KeyError(group_name + node_name)

    def _has_node_by_uri(self, uri):
        try:
            self._get_node_by_uri(uri)
            return True
        except KeyError:
            return False


def as_group_name(tag):
    return tag.rstrip('/') + '/'
