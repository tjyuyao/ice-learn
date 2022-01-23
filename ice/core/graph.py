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


from collections import Counter as _Counter

from typing import Any, Callable, Dict, List

from ice.llutil.argparser import as_list


class InvalidURIError(Exception):
    """An Exception raised when valid node URI is expected."""


class StopTask(Exception):
    """An Exception raised to exit current task."""


class Counter(_Counter):

    def __getattr__(self, key):
        try: return super()[key]
        except KeyError: return 0

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(super()):
            return super().__setattr__(__name, __value)
        else:
            return super().__setitem__(__name, __value)


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


