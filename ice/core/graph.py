from typing import Any, Callable, Dict, List

from ice.llutil.argparser import as_list
from ice.llutil.config import Configurable


class InvalidURIError(Exception):
    """An Exception raised when valid node URI is expected."""


class StopTask(Exception):
    """An Exception raised to exit current task."""


class Node(Configurable):

    def __freeze__(self,
                 forward:Callable[["Node", "NodeOutputCache"], Any]=None,
                 **resources) -> None:
        self.resources = resources
        self.forward_impl = forward
        self.egraph:ExecutableGraph = None

    @property
    def name(self) -> str:
        """Returns the node name in the current activated ``ExecutableGraph``."""
        return self.egraph.node_names[self]

    @property
    def uris(self) -> List[str]:
        """Returns the node URIs <{tag}/{name}> in the current ``HyperGraph``."""
        return [group_name + self.name for group_name in self.egraph.group_names[self]]

    @property
    def training(self) -> bool:
        if self.egraph is None or self.egraph.task is None: return False
        return self.egraph.task.training

    @property
    def device(self):
        return self.egraph.task.launcher.assigned_device

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