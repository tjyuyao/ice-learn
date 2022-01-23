from argparse import Namespace
from copy import deepcopy
from inspect import signature
from typing import Dict, List, overload

import torch.cuda
from ice.core.graph import Counter, ExecutableGraph, InvalidURIError, StopTask
from ice.llutil.argparser import as_list, isa
from ice.llutil.config import configurable
from ice.llutil.logging import get_logger
from ice.llutil.multiprocessing import called_from_main
from ice.llutil.multiprocessing.launcher import ElasticLauncher
from torch.autograd.grad_mode import set_grad_enabled


class _Task:

    def __init__(self, *, steps: int = 0, epochs: int = 0):
        self.steps = steps
        self.epochs = epochs


@configurable
class Task(_Task):

    @overload
    def __init__(self, *, train: bool, steps: int, tags="*"): ...

    @overload
    def __init__(self, *, train: bool, epochs: int, tags="*"): ...

    def __init__(self, *, train: bool, tags="*", **kwds):
        super().__init__(**kwds)
        assert self.epochs == 0 or self.steps == 0
        self.training = train
        self.tags = tags
        self.step_mode = self.epochs == 0

    def __repr__(self) -> str:
        return repr(self._config)

    def __call__(self, hyper_graph: "HyperGraph", launcher: ElasticLauncher):
        # maintain running progress.
        hyper_graph.global_counters.tasks[self._train_str] += 1
        hyper_graph.global_counters.epochs[self._train_str] += self.epochs

        # prepare states.
        self.launcher = launcher
        self.egraph: ExecutableGraph = hyper_graph[self.tags]
        self.egraph.task = self

        if self.egraph is not hyper_graph.last_executed_egraph:
            hyper_graph.last_executed_egraph.clean_up()
            self.egraph.prepare()
            if self.launcher.assigned_device.type == "cuda":
                torch.cuda.empty_cache() # result in more precise value in `nvidia-smi`.
        hyper_graph.last_executed_egraph = self.egraph

        # run epochs: assert self.epochs == 0 or self.steps == 0
        for _ in range(self.epochs):
            self.egraph.apply("epoch_start")
            while True:
                try: self.egraph.iterate(hyper_graph)
                except StopIteration: break
                except StopTask: return
            self.egraph.apply("epoch_end")

        # run steps: assert self.epochs == 0 or self.steps == 0
        for _ in range(self.steps):
            try:
                self.egraph.iterate(hyper_graph)
            except StopTask: return

    @property
    def _train_str(self):
        return "train" if self.training else "eval"


@configurable
class Repeat(_Task):

    def __init__(self, tasks:List[_Task], times:int) -> None:
        self.tasks = tasks
        self.repeat = times
        self.etasks = [deepcopy(t) for _ in range(times) for t in self.tasks]
        super().__init__(
            steps=sum(t.steps for t in self.etasks if isa(t, _Task)),
            epochs=sum(t.steps for t in self.etasks if isa(t, _Task)),
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

    def add_node(self, name, node, tags="*"):
        uris = []
        for tag in tags:
            as_group_name = tag.rstrip('/') + '/'
            uris.append(as_group_name + name)
        self[uris] = node

    @overload
    def run(self, tasks, device="auto"):
        ...

    @overload
    def run(self, tasks, launcher:ElasticLauncher):
        ...

    def run(self, tasks, launcher:ElasticLauncher="auto"):
        if called_from_main():
            if isa(launcher, str):
                launcher = ElasticLauncher(devices=launcher)
            launcher(self._run_impl, self, tasks, launcher)

    def _run_impl(self, tasks, launcher:ElasticLauncher):
        for task in as_list(tasks):
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
                self.groups[group_name][node_name] = value
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
        shortcut_key = hash(group_names)
        if shortcut_key in self.shortcuts:
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
        if -1 == idns: raise InvalidURIError(uri)
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
