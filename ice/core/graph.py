"""An executable configuration graph.

Note:
    We describe the concept of this core module in following few lines and show some pesudo-codes. This is very close to but not the same as the real code.

An acyclic directed hypergraph $G$ consists of a set of vertices $V$ and a set of hyperarcs $H$, where a hyperarc is a pair $<X, Y>$ , $X$ and $Y$ non empty subset of $V$.

We have a tag system that split the vertices $V$ into maybe overlapping subsets $V_i$, that each of which is a degenerated hypergraph $G_i$ that only consists of vertices $V_i$ and a set of hyperarcs $H_i$ so that each hyperarc is a pair $<x, Y>$, where $x \in V_i$ and $Y \subset V_i$. We call tails $x$ as producers and heads $Y$ as consumers in each hyperarc, this states the dependencies.

User defines a vertice (`Node` in the code) by specify a computation process $f$ (`forward` in the code) and the resources $R$ (`Dataset`s, `nn.Module`s, imperatively programmed function definitions such as losses and metrics, etc.) needed by it.

```python
vertice_1 = Node(
    name = "consumer_node_name",
    resources = ...,
    forward = lambda n, x: do_something_with(n.resources, x["producer_node_name"]),
    tags = ["group1", "group2"],
)
```

A longer version of `forward` parameter that corresponds to the previous notation would be `forward = lambda self, V_i: do_something_with(self.resources, V_i["x"])`,  but we will stick to the shorter version in the code.

So at the time of configuration, we are able to define every material as a node, and the name of nodes can be duplicated, i.e. multiple $x\in V$ can have the same identifier, as long as they does not have the same tag $i$ that selects $V_i$. The tags mechanism is flexible. Every node can have multiple of them, and multiple tags can be specified so that a union of subsets will be retrieved. If no tag is specified for a node, a default tag `*` will be used and a retrival will always include the `*` group.

```python
hyper_graph = HyperGraph([
    vertice_1,
    vertice_2,
    ...,
    vertice_n,
])

activated_graph = hyper_graph["group1", "group3", "group5"]
freeze_and_execute(activated_graph)
```
"""

import logging
import os
import time
from argparse import Namespace
from collections import Counter as _Counter
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, List, overload

import torch.cuda
import torch.distributed as dist
from torch.autograd.grad_mode import set_grad_enabled
from ice.llutil import multiprocessing as mp
from ice.llutil.argparser import as_list, isa
from ice.llutil.config import configurable


class InvalidURIError(Exception):
    """An Exception raised when valid node URI is expected."""


class StopTask(Exception):
    """An Exception raised to exit current task."""


class Counter(_Counter):

    def __getattr__(self, key):
        try: return super()[key]
        except KeyError: pass
        raise AttributeError(key)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(super()):
            return super().__setattr__(__name, __value)
        else:
            return super().__setitem__(__name, __value)


class Device:

    def __init__(self, reprstr:str, world_size:int="") -> None:
        """initialize Device object from string.

        Attributes:
        - device_type(str): "cuda", "cpu".
        - device_id(int): will be 0 if not speficied or ``device_type`` is "cpu".
        - world_size(int): will be 1 for "cpu", larger than 1 often indicates in a DDP case.

        Examples:
            The following are examples for input string, the representation string is of format ``{device_type}:{device_id}:{world_size}``.

        >>> Device("cpu")
        cpu:0:1

        >>> Device("cuda")
        cuda:0:1

        >>> Device("cuda:1")
        cuda:1:1

        >>> Device("ddp", world_size=4)
        cuda:0:4

        >>> Device("ddp:0", world_size=4)
        cuda:0:4

        >>> Device("ddp:0:4")
        cuda:0:4

        Note:
            `world_size` is optional, the maximum number of cuda devices can be detected automatically using ``torch.cuda.device_count()``.
        """
        idid = reprstr.find(":")
        if -1 == idid:
            self.device_type = reprstr
            self.device_id = 0
            self.world_size = int(world_size) if world_size else 1
        else:
            self.device_type = reprstr[:idid]
            self.device_id = reprstr[idid+1:]
            self.world_size = world_size
            if self.device_id:
                idsz = reprstr.find(":")
                if -1 == idsz:
                    self.device_id = int(self.device_id)
                else:
                    if not self.world_size:
                        self.world_size = int(self.device_id[idsz+1:])
                    self.device_id = int(self.device_id[:idsz])
            else:
                self.device_id = 0
            if not self.world_size:
                if self.device_type == "ddp":
                    self.device_type = "cuda"
                    self.world_size = torch.cuda.device_count()
                else:
                    self.world_size = 1

        assert self.device_type in ("cuda", "cpu")

    def __repr__(self) -> str:
        return f"{self.device_type}:{self.device_id}:{self.world_size}"

    def as_torch_device(self) -> str:
        return f"{self.device_type}:{self.device_id}"


class Node:

    def __init__(self,
                 forward:Callable[["Node", "NodeOutputCache"], Any]=None,
                 **resources) -> None:
        self.resources = resources
        self.forward_impl = forward
        self.egraph:ExecutableGraph = None

    @property
    def name(self) -> str:
        """Returns the node name in the current activated ``ExecutableGraph``.

        Returns:
            str|None: the name specified by `ice.add_...(name=...)`.
        """
        return self.egraph.node_names[self]

    @property
    def uris(self) -> List[str]:
        """Returns the node URIs in the current ``HyperGraph``.

        Returns:
            List[str]: ["{tag}/{name}"], each as a unique identifier of this node.
        """
        return [group_name + self.name for group_name in self.egraph.group_names[self]]

    @property
    def training(self) -> bool:
        if self.egraph is None or self.egraph.task is None: return False
        return self.egraph.task.training

    @property
    def device(self):
        return self.egraph.task.device.as_torch_device()

    @property
    def step_mode(self) -> bool:
        return self.egraph.task.step_mode

    def forward(self):
        self.forward_impl(self, self.egraph.cache)

    def backward(self): ...

    def update(self): ...

    def epoch_start(self): ...

    def epoch_end(self): ...

    def prepare(self): """Prepare all resources, including moving tensors to GPU."""

    def clean_up(self): ...

    def interrupt(self): ...

    def dry_run(self): ...


class NodeOutputCache:

    def __init__(self, graph:"ExecutableGraph") -> None:
        self.graph = graph
        self.clear()

    def __getitem__(self, name):
        """Execute node with name ``name`` if not executed, return the last executed cache else."""
        if name not in self.data:
            self.data[name] = self.graph[name].forward()
        return self.data[name]

    def clear(self):
        """Clear the cache, next calls to ``__getitem__`` will recalculate."""
        self.data = {}


class ExecutableGraph:

    def __init__(self) -> None:
        self.nodes:Dict[str, Node] = {}
        self.group_names:Dict[Node, List[str]] = {}
        self.node_names:Dict[Node, str] = {}
        self.cache = NodeOutputCache(self)
        self.task = None

    def add_node(self, node_name, node, group_names):
        if node_name in self.nodes.keys():
            assert node is self.nodes[node_name], "Different node can not share node_name in one task."
            assert node_name == self.node_names[node], f"A node_name cannot have two different names: `{node_name}` and `{self.node_names[node]}`."
        else:
            self.nodes[node_name] = node
            self.node_names[node] = node_name
            self.group_names[node] = set()
        for group_name in as_list(group_names):
            self.group_namesp[node].add(group_name)

    def __getitem__(self, key):
        return self.nodes[key]

    def partial_apply(self,
                      method: str,
                      filter: Callable[[Node], bool] = lambda _: True
                      ):
        for v in self.nodes.values():
            if filter(v):
                getattr(v, method)()

    def apply(self, method:str):
        for v in self.nodes.values():
            getattr(v, method)()

    def prepare(self):
        for node in self.nodes.items():
            node.egraph = self
        self.apply("prepare")

    def iterate(self, hyper_graph):
        if self.task.training:
            if hyper_graph._training_steps_resumed():
                self.apply("forward")
                self.apply("backward")
                self.apply("update")
            else:
                self.apply("dry_run")
        else: # eval
            if self._training_tasks_resumed(hyper_graph):
                self.apply("forward")
                self.apply("update")
            else:
                raise StopTask()

    def clean_up(self):
        self.apply("clean_up")


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
        self.device = Device("cpu")
        self.tags = tags
        self.step_mode = self.epochs == 0

    def __repr__(self) -> str:
        return repr(self._config)

    def __call__(self, hyper_graph: "HyperGraph", device: Device):
        # maintain running progress.
        hyper_graph.global_counters.tasks[self._train_str] += 1
        hyper_graph.global_counters.epochs[self._train_str] += self.epochs

        # prepare states.
        self.device = device
        self.egraph: ExecutableGraph = hyper_graph[self.tags]
        self.egraph.task = self

        if self.egraph is not hyper_graph.last_executed_egraph:
            hyper_graph.last_executed_egraph.clean_up()
            self.egraph.prepare()
            if self.device.device_type == "cuda":
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

    def __call__(self, hyper_graph: "HyperGraph", device: Device):
        hyper_graph._run_impl(self.etasks, device)

    def __repr__(self) -> str:
        reprstr = ",".join([repr(subtask) for subtask in self.tasks])
        reprstr = f"[{reprstr}] * {self.repeat}"
        return reprstr


def _randport(start, end):
    return time.time_ns() % (end - start) + start


def _run_impl_ddp_wrapper(rank: int, world_size: int, master_port: str, self: "HyperGraph", tasks, extra_kwds):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)  # https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    self._run_impl(tasks, device=rank, ddp_enabled=True, world_size=world_size, **extra_kwds)
    dist.destroy_process_group()


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

    def run(self, tasks, device:Device="cpu"):
        device = Device(device) if isa(device, str) else device
        if device.device_type == "cuda" and device.world_size > 1:
            mp_ctx:mp.ProcessContext = mp.start_processes(
                _run_impl_ddp_wrapper,
                args=(device.world_size, str(_randport(16894, 17194)), self, tasks),
                nprocs=device.world_size,
                join=False,
                start_method="spawn",
            )
            while not mp_ctx.join():
                pass
        else:
            self._run_impl(tasks, device)
        time.sleep(1.)  # To avoid [Errno 104] ("Connection reset by peer") in case program switch to next call of `run()` too quickly.

    def _run_impl(self, tasks, device):
        for task in as_list(tasks):
            if isa(task, _Task):
                with set_grad_enabled(task.training):
                    task(self, device)
            elif isa(task, callable):
                args = [x for x, _ in zip(
                    [self, device],
                    signature(task).parameters
                )]
                task(*args)
            else:
                logging.getLogger(__name__).warning(f"A custom task `{task}` is not callable, skipping.")


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
