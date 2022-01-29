"""contains ``Node`` and ``ExecutableGraph``."""

from typing import Any, Callable, Dict, List

from ice.llutil.argparser import as_list
from ice.llutil.config import Configurable


class InvalidURIError(Exception):
    """An Exception raised when valid node URI is expected."""


class StopTask(Exception):
    """An Exception raised to exit current task."""


class Node(Configurable):

    """This class defines the executable node.

    A executable graph is defined by a collection of executable nodes and their dependency relationships. 
    
    A node is executable if it has at least following phases of execution: `forward`, `backward`, `update`. Different subclass of nodes may implement them differently.
    
    This class is designed to be executed easily in batch mode (see ``ExecutableGraph.apply()`` for details), so that a bunch of nodes can execute together, respecting several synchronization points between phases.
    
    The dependency relationship is determined at runtime by how user access the `graph` argument of `Node.forward()` function. The `graph` argument is actually a cache (a ``GraphOutputCache`` instance) of the graph nodes outputs. The results of precedent nodes will be saved in the cache, so dependents can retrieve them easily.
    """
    
    EVENTS = ("forward", "backward", "update", "epoch_start", "epoch_end", "prepare", "clean_up", "dry_run")

    def __freeze__(self,
                   forward: Callable[["Node", "GraphOutputCache"], Any] = None,
                   **resources) -> None:
        """initialize the node.

        Args:
            forward (Callable[[self, x:``GraphOutputCache``], Any], optional): if specified, will override the original forward method.
            **resources: resources will be updated into the attributes of Node.
        """
        if forward is not None:
            self.forward_impl = lambda x: forward(self, x)  # override original implementation
            
        self.egraph: ExecutableGraph = None
        
        for k, v in resources.items():
            if hasattr(self, k) and not k in self.EVENTS:
                assert False, f"{k} is preserved for other usage and can not be used as a resource name."
            if callable(v): setattr(self, k, lambda *a, **k: v(self, *a, **k))
            else: setattr(self, k, v)

    @property
    def name(self) -> str:
        """the node name in the current activated ``ExecutableGraph``."""
        return self.egraph.node_names[self]

    @property
    def uris(self) -> List[str]:
        """the node URIs `<tag/name>` in the current ``HyperGraph``."""
        return [group_name + self.name for group_name in self.egraph.group_names[self]]

    @property
    def training(self) -> bool:
        """whether current task is training."""
        if self.egraph is None or self.egraph.task is None: return False
        return self.egraph.task.training

    @property
    def device(self):
        """the assigned device by current launcher."""
        return self.egraph.task.launcher.assigned_device

    @property
    def step_mode(self) -> bool:
        """whether current task is running by step (True) or by epoch (False)."""
        return self.egraph.task.step_mode
    
    @property
    def output(self):
        return self.forward()

    def forward(self):
        """retrieves forward output in cache or calculates it using `forward_impl` and save the output to the cache. Subclasses should not override this method."""
        
        name = self.name
        cache = self.egraph.cache
        if name in cache:
            return cache[name]
        else:
            output = self.forward_impl(cache)
            cache[name] = output
            return output

    def forward_impl(self, graph:"GraphOutputCache"): """forward pass of the node, inputs of current executable graph can be directly retrieved from `graph` argument."""

    def backward(self): """calculates gradients."""

    def update(self): """update parameters or buffers, e.g. using SGD based optimizer to update parameters. """

    def epoch_start(self): """an event hook for epoch start. (only for epoch mode)"""

    def epoch_end(self): """an event hook for epoch end. (only for epoch mode)"""

    def prepare(self): """an event hook for prepare all resources at switching executable graphs, e.g. moving models to device, initialize dataloaders, etc."""
    
    def clean_up(self): """an event hook for clean up all resources at switching executable graphs, e.g. clear device memory, closing files, etc."""

    def dry_run(self): """only update states about progress."""
    
    def state_dict(self): """returns serialization of current node."""
    
    def load_state_dict(self, state_dict): """resumes node state from state_dict."""


class GraphOutputCache:

    def __init__(self, egraph:"ExecutableGraph") -> None:
        self.egraph = egraph
        self.clear()

    def __getitem__(self, name):
        """Execute node with name ``name`` if not executed, return the last executed cache else."""
        if name not in self.data:
            self.data[name] = self.egraph[name].forward()
        return self.data[name]

    def __contains__(self, key):
        return key in self.data
    
    def __setitem__(self, name, value):
        self.data[name] = value

    def clear(self):
        """Clear the cache, next calls to ``__getitem__`` will recalculate."""
        self.data = {}


class ExecutableGraph:

    def __init__(self) -> None:
        self.nodes:Dict[str, Node] = {}
        self.group_names:Dict[Node, List[str]] = {}
        self.node_names:Dict[Node, str] = {}
        self.cache = GraphOutputCache(self)
        self.task = None

    def add_node(self, node_name, node, group_names):
        if node_name in self.nodes.keys() and node is not self.nodes[node_name]:
            if self.group_names[self.nodes[node_name]] != ["*/"]:
                assert node is self.nodes[node_name], "Different node can not share node_name in one task."
                assert node_name == self.node_names[node], f"A node_name cannot have two different names: `{node_name}` and `{self.node_names[node]}`."
            elif ["*/"] == group_names:
                return
        self.nodes[node_name] = node
        self.node_names[node] = node_name
        self.group_names[node] = set()
        for group_name in as_list(group_names):
            self.group_names[node].add(group_name)

    def __getitem__(self, key):
        return self.nodes[key]

    def items(self):
        return self.nodes.items()

    def apply(self,
              method: str,
              *args,
              filter: Callable[[Node], bool] = lambda _: True,
              **kwds):
        for v in self.nodes.values():
            if filter(v):
                getattr(v, method)(*args, **kwds)

    def prepare_nodes(self):
        for node in self.nodes.values():
            node.egraph = self
        self.apply("prepare")
        
    def clean_up_nodes(self):
        self.apply("clean_up")

    def iterate(self, hyper_graph):
        self.cache.clear()
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