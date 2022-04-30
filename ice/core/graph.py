"""contains ``Node`` and ``ExecutableGraph``."""
from __future__ import annotations
from collections import UserDict

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, overload


from ice.llutil.config import Configurable
from ice.llutil.launcher.launcher import get_current_launcher
from ice.llutil.logger import get_logger

if TYPE_CHECKING:
    import torch
    from torch.cuda.amp.grad_scaler import GradScaler
    from ice.llutil.board import BoardWriter

if TYPE_CHECKING:
    from ice.core.hypergraph import Task, HyperGraph

class InvalidURIError(Exception):
    """An Exception raised when valid node URI is expected."""

class StopTask(Exception):
    """An Exception raised to exit current task."""
    
class StopAllTasks(Exception):
    """An Exception raised to exit current running."""


class Node(Configurable):

    """This class defines the executable node.

    A executable graph is defined by a collection of executable nodes and their dependency relationships. 
    
    A node is executable if it has at least following phases of execution: `forward`, `backward`, `update`. Different subclass of nodes may implement them differently.
    
    This class is designed to be executed easily in batch mode (see ``ExecutableGraph.apply()`` for details), so that a bunch of nodes can execute together, respecting several synchronization points between phases.
    
    The dependency relationship is determined at runtime by how user access the `graph` argument of `Node.forward()` function. The `graph` argument is actually a cache (a ``GraphOutputCache`` instance) of the graph nodes outputs. The results of precedent nodes will be saved in the cache, so dependents can retrieve them easily.
    """
    
    EVENTS = ("forward", "backward", "update", "epoch_start", "epoch_end", "prepare", "clean_up", "dry_run")

    @overload
    def __init__(self,
                 forward: Callable[["Node", "GraphOutputCache"], Any] = None,
                 **resources): ...
    
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.egraph: ExecutableGraph = None

    def __freeze__(self,
                   forward: Callable[["Node", "GraphOutputCache"], Any] = None,
                   **resources) -> None:
        """initialize the node.

        Args:
            forward (Callable[[self, x:``GraphOutputCache``], Any], optional): if specified, will override the original forward method.
            **resources: resources will be updated into the attributes of Node.
        """
        import torch

        if forward is not None:
            self.forward_impl = lambda x: forward(self, x)  # override original implementation
                    
        for k, v in resources.items():
            if hasattr(self, k) and not k in self.EVENTS:
                assert False, f"{k} is preserved for other usage and can not be used as a resource name."
            if isinstance(v, (torch.Tensor, torch.nn.Module)):
                v = self.move(v)
            if callable(v): setattr(self, k, lambda *a, **k: v(self, *a, **k))
            else: setattr(self, k, v)
            
        return self

    @property
    def name(self) -> str:
        """the node name in the current activated ``ExecutableGraph``."""
        return self.egraph.node_names[self]

    @property
    def training(self) -> bool:
        """whether current task is training."""
        if self.egraph is None or self.egraph.task is None: return False
        return self.egraph.task.training

    @property
    def device(self) -> torch.device:
        """the assigned device by current launcher."""
        return self.egraph.task.launcher.assigned_device

    @property
    def step_mode(self) -> bool:
        """whether current task is running by step (True) or by epoch (False)."""
        return self.egraph.task.step_mode

    @property
    def task(self):
        """the current task."""
        return self.egraph.task
    
    @property
    def launcher(self):
        """the current launcher."""
        return get_current_launcher()
    
    @property
    def global_auto_steps(self) -> int:
        """the global steps of current task."""
        return self.egraph.task.global_auto_steps
    
    @property
    def global_train_steps(self) -> int:
        """the global train steps of current task."""
        return self.egraph.hypergraph.global_counters.steps.train

    @property
    def global_train_epochs(self) -> int:
        """the global train epochs of current task."""
        return self.egraph.hypergraph.global_counters.epochs.train

    @property
    def epoch_steps(self) -> int:
        """the steps of current epoch."""
        return self.egraph.task.epoch_steps

    @property
    def epoch_size(self) -> int:
        """the size of current epoch."""
        return self.egraph.task.epoch_size
    
    @property
    def out_dir(self) -> str:
        """the output directory of current task."""
        return self.egraph.hypergraph.run_info.out_dir
    
    @property
    def run_id(self) -> str:
        """the run id of current task."""
        return self.egraph.hypergraph.run_info.full_run_id

    @property
    def grad_scaler(self) -> GradScaler:
        """the grad scaler of current task."""
        return self.egraph.grad_scaler
    
    @property
    def grad_acc_steps(self) -> int:
        """the grad accumulator steps of current task."""
        return self.egraph.hypergraph.grad_acc_steps

    @property
    def board(self) -> BoardWriter:
        """the board writer of current task."""
        return self.egraph.hypergraph.board
    
    @property
    def all_nodes(self) -> Dict[str, Node]:
        return self.egraph.nodes

    def forward(self):
        """retrieves forward output in cache or calculates it using `forward_impl` and save the output to the cache. Subclasses should not override this method."""
        name = self.name
        cache = self.egraph.cache
        if name in cache:
            output = cache[name]
        else:
            cache.acquire(name)
            output = self.forward_impl(cache)
            cache.release(name)
            cache[name] = output
        return output

    @property
    def forward_output(self):
        return self.forward()

    def forward_impl(self, cache:"GraphOutputCache"): """forward pass of the node, inputs of current executable graph can be directly retrieved from `graph` argument."""

    def backward(self): """calculates gradients."""

    def update(self): """update parameters or buffers, e.g. using SGD based optimizer to update parameters. """

    def epoch_start(self): """an event hook for epoch start. (only for epoch mode)"""

    def epoch_end(self): """an event hook for epoch end. (only for epoch mode)"""

    def prepare(self): """an event hook for prepare all resources at switching executable graphs."""
    
    def clean_up(self): """an event hook for clean up all resources at switching executable graphs."""

    def dry_run(self): """only update states about progress."""
    
    def state_dict(self) -> Dict: """returns serialization of current node."""
    
    def load_state_dict(self, _state_dict:Dict, strict:bool): """resumes node state from state_dict."""
    
    def move(self, data, device=None):
        """Moves data to the CPU or the GPU.

        If :attr:`device` is ``None`` and the node has a ``device`` attribute, then that is the device where the data
        is moved. Otherwise, the data is moved according to the ``device.type``. If :attr:``data`` is a dict or list,
        the function is applied recursively to each of the elements.

        Args:
            data (torch.Tensor or torch.nn.Module or list or dict): the data to move.
            device (str or torch.device): the string or instance of torch.device in which to move the data.

        Returns:
            torch.Tensor or torch.nn.Module: the data in the requested device.

        Raises:
            RuntimeError: data is not one of torch.Tensor, torch.nn.module, list or dict.
        """
        import torch
        if device is None:
            device = get_current_launcher().assigned_device
        if isinstance(data, (torch.Tensor, torch.nn.Module)):
            if device.type == "cpu": return data.cpu()
            else: return data.cuda()
        elif isinstance(data, list):
            for i, v in enumerate(data): data[i] = self.move(v)
            return data
        elif isinstance(data, dict):
            for k, v in data.items(): data[k] = self.move(v)
            return data
        else:
            return data


class GraphOutputCache(UserDict):
    """a cache for storing and searching forward outputs of nodes.

    This class is used to store and search forward outputs of nodes.

    Attributes:
        cache (dict): a dict for storing forward outputs.
        egraph (ExecutableGraph): the executable graph.
        data (Dict[str, torch.Tensor]): the cache.
    """
    def __init__(self, egraph:"ExecutableGraph") -> None:
        self.egraph = egraph
        self.clear()

    def __getitem__(self, name):
        """Execute node with name ``name`` if not executed, return the last executed cache else."""
        if len(self.users):
            self.add_deps(name)
        if name not in self.data:
            self.data[name] = self.egraph[name].forward()
        return self.data[name]

    def __contains__(self, key):
        return key in self.data
    
    def __setitem__(self, name, value):
        self.data[name] = value
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def clear(self):
        """Clear the cache, next calls to ``__getitem__`` will recalculate."""
        self.data = {}
        self.users = []
        self.deps:Dict[str, Set] = {}
    
    def acquire(self, user):
        self.users.append(user)
        if user not in self.deps:
            self.deps[user] = set()
    
    def release(self, user):
        assert self.users.pop(-1) == user
    
    def add_deps(self, ancestor):
        self.deps[self.users[-1]].add(ancestor)
    
    def find_ancestors(self, user:str, filter:Callable[[Node], bool] = None):
        out = []
        q = []
        q.append(user)
        iter_count = 0
        while len(q) > 0:
            name = q.pop(0)
            assert name in self.deps
            for ancestor in self.deps[name]:
                q.append(ancestor)
            node = self.egraph[name]
            if node in out: continue
            if filter is None or filter(node):
                out.append(node)
            
            iter_count += 1
            if iter_count == 1000:
                get_logger().warn(f"Possible dependency loop: {self.deps}")
        return out
    

class ExecutableGraph:
    """an executable graph.

    This class is used to execute nodes in a graph.

    Attributes:
        hypergraph (HyperGraph): the hypergraph.
        nodes (Dict[str, Node]): a dict for storing nodes.
        nodes_tags (Dict[str, str]): a dict for storing tags of nodes.
        nodes_names (Dict[str, str]): a dict for storing names of nodes.
        cache (GraphOutputCache): a cache for storing and searching forward outputs of nodes.
        task: the task of the graph.
        losses: the losses of the graph.
        total_loss: the total loss of the graph.
    """

    def __init__(self, hypergraph) -> None:
        self.hypergraph: HyperGraph = hypergraph
        self.nodes:Dict[str, Node] = {}
        self.node_tags:Dict[Node, List[str]] = {}
        self.node_names:Dict[Node, str] = {}
        self.cache = GraphOutputCache(self)
        self.task: Task = None
        self.losses_counter = 0
        self.total_loss = 0

    def add_node(self, node_name, node, tags):
        """add a node to the graph.

        Args:
            node_name (str): the name of the node.
            node (Node): the node.
            tags (List[str]): the tags of the node.

        Raises:
            RuntimeError: the node is not a node of the hypergraph.
        """
        if node_name in self.nodes.keys() and node is not self.nodes[node_name]:
            if self.node_tags[self.nodes[node_name]] != ["*"]:
                assert node is self.nodes[node_name], f"Different node can not share node_name `{node_name}` in one task."
                assert node_name == self.node_names[node], f"A node_name cannot have two different names: `{node_name}` and `{self.node_names[node]}`."
            elif ["*"] == tags:
                return
        self.nodes[node_name] = node
        self.node_names[node] = node_name
        self.node_tags[node] = tags

    def __getitem__(self, key):
        return self.nodes[key]

    def items(self):
        return self.nodes.items()

    def apply(self,
              method: str,
              *args,
              filter: Callable[[Node], bool] = lambda _: True,
              **kwds):
        """apply ``method`` to all nodes in the graph.

        Args:
            method (str): the method name.
            *args: the arguments of the method.
            filter (Callable[[Node], bool]): the filter function.
            **kwds: the keyword arguments of the method.

        Returns:
            List[Any]: the return values of the method.

        Raises:
            RuntimeError: the method is not found.
        """
        for v in self.nodes.values():
            if filter(v):
                try:
                    getattr(v, method)(*args, **kwds)
                except Exception as e:
                    if not isinstance(e, (StopIteration)):
                        get_logger().error(f">>>>> ERROR: Node '{v.name}' raised '{type(e).__name__}' during '{method}' stage. <<<<<<")
                    raise

    def prepare_nodes(self):
        """prepare all nodes in the graph.

        This method is called before the graph is executed.
        """
        for node in self.nodes.values():
            node.egraph = self
            node.freeze()
        self.apply("prepare")
        
    def clean_up_nodes(self):
        """clean up all nodes in the graph.

        This method is called after the graph is executed.
        """
        self.apply("clean_up")
    
    @property
    def grad_scaler(self) -> GradScaler:
        return self.hypergraph._grad_scaler

    def iterate(self):
        """iterate all nodes in the graph."""
        self.cache.clear()
        self.task.global_auto_steps += 1
        if self.task.training:
            self.apply("forward")
            self.apply("backward")
            self.apply("update")
            if self.task.global_auto_steps % self.hypergraph.grad_acc_steps == 0:
                self.grad_scaler.update()
        else: # eval
            self.apply("forward")
            self.apply("update")
