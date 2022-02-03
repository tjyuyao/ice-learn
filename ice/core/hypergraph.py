from argparse import Namespace
from copy import deepcopy
from inspect import signature
from typing import Dict, List, overload

import torch.cuda
from ice.core.graph import ExecutableGraph, GraphOutputCache, InvalidURIError, StopTask, Node
from ice.llutil.argparser import as_list, isa
from ice.llutil.collections import Counter
from ice.llutil.config import Configurable, configurable, is_configurable
from ice.llutil.launcher import ElasticLauncher
from ice.llutil.logging import get_logger
from ice.llutil.multiprocessing import called_from_main
from torch.autograd.grad_mode import set_grad_enabled

from ice.llutil.print import _print


class _Task(Configurable):

    def __freeze__(self, *, steps: int = 0, epochs: int = 0):
        self.total_steps = steps
        self.total_epochs = epochs


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
        self.tags = tags
        self.step_mode = self.total_epochs == 0
        self.task_steps = 0
        self.task_epochs = 0

    def __call__(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
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

        # run epochs: assert self.epochs != 0 and self.steps == 0
        for task_epochs in range(self.total_epochs):
            self.task_epochs = task_epochs
            self.egraph.apply("epoch_start")
            while True:
                try: self.egraph.iterate(hyper_graph)
                except StopIteration: break
                except StopTask: return
            self.egraph.apply("epoch_end")
            self.global_epochs += 1

        # run steps: assert self.epochs == 0 and self.steps != 0
        for task_steps in range(self.total_steps):
            self.task_steps = task_steps
            try:
                self.egraph.iterate(hyper_graph)
            except StopTask: return
            
        self.global_tasks += 1

    @property
    def _train_str(self):
        return "train" if self.training else "eval"
    
    @property
    def global_tasks(self):
        return self.hyper_graph.global_counters.tasks[self._train_str]
        
    @global_tasks.setter
    def global_tasks(self, value):
        self.hyper_graph.global_counters.tasks[self._train_str] = value
        return value
    
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
        self.tasks = tasks
        self.repeat = times
        self.etasks = [deepcopy(t) for _ in range(times) for t in self.tasks]
        super().__freeze__(
            steps=sum(t.total_steps for t in self.etasks if isa(t, _Task)),
            epochs=sum(t.total_steps for t in self.etasks if isa(t, _Task)),
        )

    def __call__(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
        hyper_graph._run_impl(self.etasks, launcher)

    def __repr__(self) -> str:
        reprstr = ",".join([repr(subtask) for subtask in self.tasks])
        reprstr = f"[{reprstr}] * {self.repeat}"
        return reprstr


class HyperGraph:
    """HyperGraph is the container for all nodes.
    """

    def __init__(self) -> None:
        self.groups:Dict[str, ExecutableGraph] = {}
        self.groups["*/"] = ExecutableGraph()
        self.shortcuts:Dict[str, ExecutableGraph] = {}
        self.global_counters = Namespace(
            tasks = Counter(),
            steps = Counter(),
            epochs = Counter(),
        )
        self.resumed_counters = self.global_counters
        self.last_executed_egraph = None

    def add(self, name, node, tags="*"):
        uris = []
        for tag in as_list(tags):
            as_group_name = tag.rstrip('/') + '/'
            uris.append(as_group_name + name)
        self[uris] = node
    
    def print_output_of(self, *nodes, every=1, total=None, tags:List[str] = "*"):

        def probe_fn(n:Node, x:GraphOutputCache):
            if n.current_launcher.local_rank != 0: return
            if total is not None and n.global_steps // every > total: return  # total
            if n.global_steps and (n.global_steps+1) % every: return # every
            prefix = f"E{n.global_epochs+1}S{n.global_steps+1}:"

            for nodename in as_list(nodes):
                _print(x[nodename], prefix=prefix, uri=nodename)

        self.add(f"print_output_of({','.join(as_list(nodes))})", Node(forward=probe_fn), tags=tags)

    @overload
    def run(self, tasks, devices="auto"): ...
    
    @overload
    def run(self, tasks, launcher:ElasticLauncher=None): ...
        
    @overload
    def run(
        self,
        tasks,
        devices="auto",
        nnodes="1:1",
        dist_backend="auto",
        monitor_interval=5,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=None,
        redirects="0",
        tee="0",
        log_dir=None,
        role="default",
        max_restarts=0,
        omp_num_threads=1,
        start_method="spawn",
    ):
        ...

    @overload
    def run(
        self,
        tasks,
        devices="auto",
        nnodes="1:1",
        dist_backend="auto",
        monitor_interval=5,
        rdzv_id="none",
        rdzv_endpoint="",
        rdzv_backend="static",
        rdzv_configs="",
        standalone=False,
        redirects="0",
        tee="0",
        log_dir=None,
        role="default",
        max_restarts=0,
        omp_num_threads=1,
        start_method="spawn",
    ):
        ...

    def run(self, tasks, launcher:ElasticLauncher=None, **kwds):
        if called_from_main():
            if launcher is None:
                launcher = ElasticLauncher(**kwds)
            else:
                launcher.update(kwds)
            launcher.freeze()
            launcher(self._run_impl, tasks, launcher)

    def _run_impl(self, tasks, launcher:ElasticLauncher):

        for task in as_list(tasks):
            if is_configurable(task):
                task.freeze()
            if isa(task, _Task):
                with set_grad_enabled(task.training):
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
        as_group_name = key.rstrip('/') + '/'
        self.groups[as_group_name] = value

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
        if 0 == idns: raise InvalidURIError(uri)
        group_name, node_name = uri[:idns], uri[idns:]
        if 0 == len(node_name): raise InvalidURIError(uri)
        return group_name, node_name

    def _get_node_by_uri(self, uri):
        group_name, node_name = self._parse_uri(uri)
        return self.groups[group_name][node_name]

    def _has_node_by_uri(self, uri):
        try:
            self._get_node_by_uri(uri)
            return True
        except KeyError:
            return False

    def _training_steps_resumed(self) -> bool:
        return self.global_counters.steps["train"] >= self.resumed_counters.steps["train"]

    def _training_tasks_resumed(self) -> bool:
        return self.global_counters.tasks["train"] >= self.resumed_counters.tasks["train"]
