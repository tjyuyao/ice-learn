from __future__ import annotations

import logging
import os
import random
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, overload

import numpy as np
from ice.core.graph import (ExecutableGraph, GraphOutputCache, InvalidURIError,
                            Node, StopAllTasks, StopTask)
from ice.llutil.argparser import args as ice_args
from ice.llutil.argparser import as_list, is_list_of, isa
from ice.llutil.backup_src import _backup_source_files_to
from ice.llutil.collections import Dict as iDict
from ice.llutil.config import Configurable, freeze, frozen, is_configurable
from ice.llutil.ignore_me import IgnoreMe
from ice.llutil.launcher import ElasticLauncher, Events, global_shared_events
from ice.llutil.launcher.launcher import get_current_launcher
from ice.llutil.logger import get_logger
from ice.llutil.utils import enable_auto_freeze, in_main_process, init_torch_multiprocessing
from ice.llutil.print import _print, set_printoptions

from tqdm import tqdm

if TYPE_CHECKING:
    from torch.cuda.amp.grad_scaler import GradScaler


class ResumeTaskFailed(Exception):
    """raised when task structure does not match during resuming."""

class _Task(Configurable):
    """A task is a unit of computation.

    It can be a single node, or a graph. A task can be executed by a worker.
    """
    def __freeze__(self, *, steps: int = 0, epochs: int = 0):
        """Freeze the task.

        Args:
            steps: number of steps to run.
            epochs: number of epochs to run.

        Returns:
            A frozen task.
        """
        self.total_steps = steps
        self.total_epochs = epochs

    def state_dict(self):
        """Return the state of the task."""
        raise NotImplementedError()

    def load_state_dict(self, _state_dict, strict):
        """Load the state of the task."""
        raise NotImplementedError()


class Task(_Task):
    """A task is a unit of computation.

    It can be a single node, or a graph. A task can be executed by a worker.

    Args:
        node: a node or a graph.
        name: the name of the task.
        total_steps: the total number of steps to run.
        total_epochs: the total number of epochs to run.
        config: a dict of configs.
    """
    @overload
    def __init__(self, *, train: bool, steps: int, tags="*", simulate_epoch:bool=False): ...

    @overload
    def __init__(self, *, train: bool, epochs: int, tags="*"): ...

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(self, *, train: bool, tags="*", simulate_epoch=False, **kwds):
        """Freeze the task.

        Args:
            train: whether the task is for training.
            tags: a list of tags.
            steps: number of steps to run.
            epochs: number of epochs to run.

        Returns:
            self.
        """
        super().__freeze__(**kwds)
        assert self.total_epochs == 0 or self.total_steps == 0
        self.training = train
        self.tags = as_list(tags)
        self.step_mode = self.total_epochs == 0
        self.epoch_steps = 0
        self.task_epochs = 0
        self.epoch_size = 0
        self.finished = False
        self.simulate_epoch = simulate_epoch
        return self

    def __call__(self, hypergraph: "HyperGraph", launcher: ElasticLauncher):
        """Execute the task.

        Args:
            hypergraph: the hypergraph.
            launcher: the launcher.
        """
        if self.finished: return  # for resuming progress
        from torch.autograd.grad_mode import set_grad_enabled
        with set_grad_enabled(self.training):
            self.__call__impl(hypergraph, launcher)
        self.finished = True

    def __call__impl(self, hypergraph: "HyperGraph", launcher: ElasticLauncher):
        # maintain running progress.
        self.hypergraph = hypergraph

        # prepare states.
        self.launcher = launcher
        self.egraph: ExecutableGraph = hypergraph.select_egraph(self.tags)
        self.egraph.task = self

        if self.egraph is not hypergraph._last_executed_egraph:
            if hypergraph._last_executed_egraph is not None:
                hypergraph._last_executed_egraph.clean_up_nodes()
            self.egraph.prepare_nodes()
            if self.launcher.assigned_device.type == "cuda":
                import torch.cuda
                torch.cuda.empty_cache() # result in more precise value in `nvidia-smi`.
        hypergraph._last_executed_egraph = self.egraph

        # run epochs: assert self.total_epochs == 0 or self.total_steps == 0
        if self.total_epochs:
            if self.epoch_steps != 0: self.global_auto_epochs -= 1  # already started before last interruption

            epoch_size = None
            for node in self.egraph.nodes.values():
                if hasattr(node, "__len__"):
                    len_node = len(node)
                    if epoch_size is None or epoch_size > len_node:
                        epoch_size = len_node
            if epoch_size is None:
                get_logger().error("No DatasetNode selected for current task!")
                raise StopTask("No DatasetNode selected")
            else:
                self.epoch_size = epoch_size
                # update progress bar total
                if launcher.local_rank == 0:
                    launcher.events.progress_bar_total.value = epoch_size

            for self.task_epochs in range(self.task_epochs, self.total_epochs):
                self.egraph.apply("epoch_start")
                while True:
                    try:
                        self._iterate()
                        self.epoch_steps += 1
                    except StopIteration:
                        self.epoch_steps = 0
                        break
                    except StopTask: return
                self.egraph.apply("epoch_end")
                self.global_auto_epochs += 1
        else:
            if launcher.local_rank == 0:
                # update progress bar total
                launcher.events.progress_bar_total.value = self.total_steps
            if self.simulate_epoch:
                self.egraph.apply("epoch_start")
            for self.epoch_steps in range(self.epoch_steps, self.total_steps):
                try:
                    for _ in range(hypergraph.grad_acc_steps):
                        self._iterate()
                except StopTask: return
            if self.simulate_epoch:
                self.egraph.apply("epoch_end")
                self.global_auto_epochs += 1


    def _iterate(self):
        self.egraph.iterate()
        self.launcher.events.progress_bar_iter.value = self.epoch_steps  # notify progressbar in main process.
        self._process_events()

    def _process_events(self):
        events:Events = self.launcher.events
        if events.pause.is_set():
            events.paused.set()
            events.resume.wait()
            events.paused.clear()
        if events.trigger_save_checkpoint.is_set():
            self.hypergraph.save_checkpoint()
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
            "task_steps" : self.epoch_steps,
            "task_epochs" : self.task_epochs,
            "finished" : self.finished,
        }
        return _state_dict

    def load_state_dict(self, _state_dict, dry_run=False):

        if  isa(_state_dict, list) or \
            self.total_epochs != _state_dict["total_epochs"] or \
            self.total_steps != _state_dict["total_steps"]:
            raise ResumeTaskFailed()

        if not dry_run:
            self.epoch_steps = _state_dict["task_steps"]
            self.task_epochs = _state_dict["task_epochs"]
            self.finished = _state_dict["finished"]

    @property
    def _train_str(self):
        return "train" if self.training else "eval"

    @property
    def global_auto_epochs(self):
        return self.hypergraph.global_counters.epochs[self._train_str]

    @global_auto_epochs.setter
    def global_auto_epochs(self, value):
        self.hypergraph.global_counters.epochs[self._train_str] = value
        return value

    @property
    def global_auto_steps(self):
        return self.hypergraph.global_counters.steps[self._train_str]

    @global_auto_steps.setter
    def global_auto_steps(self, value):
        self.hypergraph.global_counters.steps[self._train_str] = value
        return value

class Repeat(_Task):
    """Repeat a task for a fixed number of times.

    Attributes:
        task (Task): Task to repeat.
        repeat (int): Number of times to repeat the task.
        epoch_size (int): Number of steps per epoch.
        total_steps (int): Total number of steps.
        total_epochs (int): Total number of epochs.
        launcher (Launcher): Launcher object.
        hypergraph (Hypergraph): Hypergraph object.
        events (Events): Events object.
    """
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

    def __call__(self, hypergraph: "HyperGraph", launcher: ElasticLauncher):
        hypergraph.exec_tasks(self.etasks, launcher)

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

def LoadCheckpointTask(resume_from, strict=False, tags="*"):
    """Load checkpoint from a file.

    Args:
        resume_from (str): Path to the checkpoint file.
        strict (bool): If True, raise an exception if the checkpoint file does not exist.
        tags (str): Tags to load.

    Returns:
        Task: Task to load the checkpoint.
    """
    def func(g: HyperGraph):
        g.load_checkpoint(resume_from, strict=strict, tags=tags)
    return func

def SaveCheckpointTask(save_to=None, tags="*"):
    """Save checkpoint to a file.

    Args:
        save_to (str): Path to the checkpoint file.
        tags (str): Tags to save.

    Returns:
        Task: Task to save the checkpoint.
    """
    def func(g: HyperGraph):
        g.save_checkpoint(save_to=save_to, tags=tags)
    return func

class Counter:
    """Counter object.

    Attributes:
        epochs (int): Number of epochs.
        steps (int): Number of steps.
    """
    def __init__(self) -> None:
        self.train = 0
        self.eval = 0

    def __getitem__(self, key):
        """Get the value of the counter.

        Args:
            key (str): Name of the counter.

        Returns:
            int: Value of the counter.
        """
        if key == "train":
            return self.train
        elif key == "eval":
            return self.eval
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """Set the value of the counter.

        Args:
            key (str): Name of the counter.
            value (int): Value of the counter.

        Raises:
            KeyError: If the key is not valid.
        """
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
    """Global counters object.

    Attributes:
        epochs (int): Number of epochs.
        steps (int): Number of steps.
    """
    steps:Counter = Counter()
    epochs:Counter = Counter()

def _tags_to_uid(tags, name):
    """Convert tags to a unique id.

    Args:
        tags (str): Tags.
        name (str): Name of the task.

    Returns:
        str: Unique id.

    Raises:
        ValueError: If the tags are not valid.
    """
    tags = as_list(tags)
    if len(tags) == 1 and tags[0][0] == "#":
        return f"{tags[0]}/{name}"
    if len(tags) == 1 and tags[0] == "*":
        return f"*/{name}"
    tags = [tag for tag in tags if tag[0] != "#" and tag != "*"]
    tag = ','.join(sorted(tags))
    return f"{tag}/{name}"


class HyperGraph:
    """HyperGraph is the container for all nodes.

    Attributes:
        nodes (dict): Nodes.
        edges (dict): Edges.
        tasks (dict): Tasks.
        launchers (dict): Launchers.
        global_counters (GlobalCounters): Global counters.
        resume_from (str): Path to the checkpoint file.
        resume_tags (str): Tags to load.
        save_to (str): Path to the checkpoint file.
        save_tags (str): Tags to save.
        strict (bool): If True, raise an exception if the checkpoint file does not exist.
        dry_run (bool): If True, do not save the checkpoint.
        verbose (bool): If True, print the progress.
        logger (Logger): Logger.

    Raises:
        ValueError: If the tags are not valid.
    """

    def __init__(self, autocast_enabled=False, autocast_dtype=None, grad_scaler:Union[bool, GradScaler] = None) -> None:
        self.nodes = {}
        self.global_counters = GlobalCounters()
        self.run_info = iDict()

        self._shortcuts:Dict[str, ExecutableGraph] = {}
        self._last_executed_egraph = None
        self._num_workers = 0

        self.entrypoint = None
        self.grad_acc_steps = 1
        self.init_autocast(autocast_enabled, autocast_dtype, grad_scaler)

    def init_autocast(self, autocast_enabled=True, autocast_dtype=None, grad_scaler:Union[bool, GradScaler] = None):
        """Initialize autocast.

        Args:
            autocast_enabled (bool): If True, enable autocast.
            autocast_dtype: ...
            grad_scaler (GradScaler): Gradient scaler.

        Raises:
            ValueError: If the autocast_dtype is not valid.
        """
        self.autocast_kwds = dict(enabled=autocast_enabled, dtype=autocast_dtype)
        self.init_grad_scaler(grad_scaler if grad_scaler is not None else autocast_enabled)
    
    def is_autocast_enabled(self) -> bool:
        """Check if autocast is enabled.

        Returns:
            bool: If True, autocast is enabled.
        """
        return self.autocast_kwds["enabled"]
    
    def backup_source_files(self, entrypoint:str):
        """Backup source files.

        Args:
            entrypoint (str): Entrypoint.

        Raises:
            ValueError: If the entrypoint is not valid.
        """
        self.entrypoint = entrypoint

    @overload
    def init_grad_scaler(self,
                         grad_scaler: Union[bool, GradScaler] = False,
                         *,
                         init_scale=2.**16,
                         growth_factor=2.0,
                         backoff_factor=0.5,
                         growth_interval=2000,
                         enabled=True):
        ...
        
    def init_grad_scaler(self, grad_scaler:Union[bool, GradScaler] = True, **kwds):
        """Initialize the gradient scaler.

        Args:
            grad_scaler (Union[bool, GradScaler]): Gradient scaler.
            **kwds: Keyword arguments.

        Returns:
            GradScaler: Gradient scaler.

        Raises:
            ValueError: If the grad_scaler is not valid.
        """
        from torch.cuda.amp.grad_scaler import GradScaler
        if isa(grad_scaler, bool):
            if grad_scaler:
                self._grad_scaler = GradScaler(**kwds)
            else:
                self._grad_scaler = GradScaler(enabled=False)
        elif isa(grad_scaler, GradScaler):
            if len(kwds):
                get_logger().warn("when grad_scaler is a GradScaler, you should not specify other keywords parameters for `init_grad_scaler()`")
            self._grad_scaler = grad_scaler
        else:
            get_logger().warn("not supported argument type for `init_grad_scaler()`, disabling.")
            self._grad_scaler = GradScaler(enabled=False)

    def is_grad_scaler_enabled(self) -> bool:
        """Check if the gradient scaler is enabled.

        Returns:
            bool: If True, the gradient scaler is enabled.

        Raises:
            ValueError: If the grad_scaler is not valid.
        """
        return self._grad_scaler.is_enabled()

    def set_gradient_accumulate(self, every=1):
        """Set the gradient accumulate steps.

        Args:
            every (int): Gradient accumulate steps.

        Raises:
            ValueError: If the every is not valid.
        """
        self.grad_acc_steps = every

    @property
    def launcher(self) -> ElasticLauncher:
        """Get the launcher.

        Returns:
            ElasticLauncher: Launcher.

        Raises:
            ValueError: If the launcher is not valid.
        """
        return self.run_info.launcher

    def add(self, name, node:Node, tags="*"):
        """Add a node.

        Args:
            name (str): Name.
            node (Node): Node.
            tags (str): Tags.

        Returns:
            Node: Node.

        Raises:
            ValueError: If the name is not valid.
        """
        assert isa(node, Node), f"{node.__class__.__name__} is not a Node"
        tags = as_list(tags)
        assert is_list_of(tags, str)
        uid = _tags_to_uid(tags, name)
        assert uid not in self.nodes, f"duplicate node (name={repr(name)}, tags={repr(tags)})"
        self.nodes[uid] = [name, node, tags]

        if isa(node, Configurable) and not frozen(node) and "num_workers" in node:
            self._num_workers = max(self._num_workers, node["num_workers"])

    def remove(self, query):
        """Remove a node.

        Args:
            query (str): Query.

        Raises:
            ValueError: If the query is not valid.
        """
        raise NotImplementedError()

    def select_egraph(self, query) -> ExecutableGraph:
        """Select an executable graph.

        Args:
            query (str): Query.

        Returns:
            ExecutableGraph: Executable graph.

        Raises:
            ValueError: If the query is not valid.
        """
        query = as_list(query)
        keys = self._select_keys(query)
        shortcut_query = hash(tuple(query))
        egraph = ExecutableGraph(self)
        if shortcut_query not in self._shortcuts:
            for key in keys:
                name, node, tags = self.nodes[key]
                egraph.add_node(name, node.auto_freeze(), tags)
            self._shortcuts[shortcut_query] = egraph
            if self.launcher.local_rank == 0:
                get_logger().info(f"selected nodes: {keys}")
        else:
            egraph = self._shortcuts[shortcut_query]
        return egraph

    def __getitem__(self, uid) -> Node:
        """Get a node by uid.

        Args:
            uid (str): Uid.

        Returns:
            Node: Node.

        Raises:
            ValueError: If the uid is not valid.
        """
        try:
            tagstr, name = uid.split("/")
            tags = tagstr.split(",")
            keys = self._select_keys(*tags)
            nodes = [self.nodes[key][1] for key in keys if self.nodes[key][0]==name]
        except:
            raise RuntimeError(f"fail to parse node uid '{uid}'")
        if len(nodes) != 1:
            raise RuntimeError(f"fail to parse node uid '{uid}'")
        return nodes[0]

    def select_nodes(self, *query):
        """Select nodes.

        Args:
            query (str): Query.

        Returns:
            list: Nodes.

        Raises:
            ValueError: If the query is not valid.
        """
        keys = self._select_keys(query)
        out = { key:self.nodes[key][1] for key in keys }
        return out

    def _select_keys(self, *query) -> List[str]:
        """Select keys.

        Args:
            query (str): Query.

        Returns:
            list: Keys.

        Raises:
            ValueError: If the query is not valid.
        """
        query = as_list(query)
        if "*" in query: return list(self.nodes.keys())
        out = []
        for key, (_, _, tags) in self.nodes.items():
            selected = True
            for tag in tags:
                if tag[0] == "#":
                    if tag in query:
                        selected=True
                        break
                elif tag == "*":
                    selected = True
                    break
                else:
                    if tag not in query:
                        selected = False
            if selected:
                out.append(key)
        return out

    def print_forward_output(self, *nodenames, every=1, total=None, tags:List[str] = "*", train_only=True, localrank0_only=True):
        """Print forward output.

        Args:
            nodenames (str): Node names.
            every (int): Print every.
            total (int): Total.
            tags (List[str]): Tags.
            train_only (bool): Train only.
            localrank0_only (bool): Local rank 0 only.

        Raises:
            ValueError: If the nodenames is not valid.
        """
        def probe_fn(n:Node, x:GraphOutputCache):
            if train_only and not n.training: return
            if localrank0_only and n.launcher.local_rank != 0: return
            if total is not None and n.global_auto_steps // every > total: return  # total
            if n.global_auto_steps > 1 and n.global_auto_steps % every: return # every
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
        redirects="2", tee="1", out_dir=None, resume_from=None, seed=0,
        role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
    ):        ...

    @overload
    def run(
        self, tasks, devices="auto", run_id="none", nnodes="1:1", dist_backend="auto", monitor_interval=5,
        rdzv_endpoint="", rdzv_backend="static", rdzv_configs="", standalone=False,
        redirects="2", tee="1", out_dir=None, resume_from=None, seed=0,
        role="default", max_restarts=0, omp_num_threads=1, start_method="spawn",
    ):        ...

    def _set_initial_rng_state(self, seed):
        """Set initial rng state.

        Args:
            seed (int): Seed.

        Returns:
            int: Seed.

        Raises:
            ValueError: If the seed is not valid.
        """
        import torch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _run_impl(self, tasks, launcher:ElasticLauncher, resume_from):
        tasks = freeze(tasks)
        self.run_info.tasks = tasks
        self.run_info.launcher = launcher
        self.run_info._task_resumed = True

        global_shared_events["debugger_start"] = launcher.events.debugger_start
        global_shared_events["debugger_end"] = launcher.events.debugger_end
        
        # determine default autocast_dtype
        if self.autocast_kwds["dtype"] is None:
            import torch
            if launcher.assigned_device.type == "cuda":
                self.autocast_kwds["dtype"] = torch.float16
            else:
                self.autocast_kwds["dtype"] = torch.bfloat16

        # setup tensorboard
        from ice.llutil.board import BoardWriter
        if not in_main_process() and get_current_launcher().rank == 0:
            self.board:BoardWriter = BoardWriter(self.run_info.out_dir)
        else:
            self.board:BoardWriter = IgnoreMe()
        
        try:
            get_logger().info(repr(ice_args))
            set_printoptions(threshold=4, linewidth=120, precision=2, sci_mode=False)
            self.load_checkpoint(resume_from)
            self.exec_tasks(tasks, launcher)
        except StopAllTasks:
            pass
        finally:
            self.board.add_hparams()
            self.board.close()

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
        if in_main_process():
            base_out_dir = out_dir or "out"
            os.makedirs(base_out_dir, exist_ok=True)
            today = datetime.today()
            run_id_dir = tempfile.mkdtemp(prefix=f"{run_id}_{today.month:02d}{today.day:02d}{today.hour:02d}{today.minute:02d}_", dir=base_out_dir)
            ckpt_dir = os.path.join(run_id_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            log_dir = os.path.join(run_id_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            if self.entrypoint is not None:
                src_dir = os.path.join(run_id_dir, "src")
                os.makedirs(src_dir, exist_ok=True)
                _backup_source_files_to(self.entrypoint, src_dir)
        
            self.run_info.run_id = run_id
            self.run_info.full_run_id = os.path.basename(run_id_dir)
            self.run_info.out_dir = run_id_dir
            self.run_info.ckpt_dir = ckpt_dir
            self.run_info.log_dir = log_dir 
       
        # Setup the handler for root logger. The submodule logger will automatically bubble up to it.
        if in_main_process():
            handler = logging.FileHandler(os.path.join(log_dir, "agent.log"))
        else:
            handler = logging.StreamHandler(sys.stderr)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    def run(self, tasks, *, launcher:ElasticLauncher=None, run_id:str="none", out_dir:str=None, resume_from:str=None, seed=0, start_method="spawn", **kwds):
        """Run the tasks.

        Args:
            tasks (list): Tasks to run.
            launcher (ElasticLauncher): Launcher.
            run_id (str): Run id.
            out_dir (str): Output directory.
            resume_from (str): Resume from checkpoint.
            seed (int): Seed.
            start_method (str): Start method.
            kwds: Keyword arguments.

        Raises:
            ValueError: If the seed is not valid.
        """
        self._set_initial_rng_state(seed)
        self._prepare_out_dir(run_id=run_id, out_dir=out_dir)

        init_torch_multiprocessing()

        if in_main_process():
            
            kwds["rdzv_id"] = run_id
            kwds["log_dir"] = self.run_info.log_dir
            kwds["start_method"] = start_method
            kwds["events"] = Events(start_method)
            if "omp_num_threads" not in kwds:
                kwds["omp_num_threads"] = self._num_workers + 1

            if launcher is None:
                launcher = ElasticLauncher(**kwds)
            else:
                launcher.update_params(kwds)
            launcher.freeze()
            launcher(self._run_impl, as_list(tasks), launcher, resume_from)

    def save_checkpoint(self, save_to=None, tags="*"):
        """Save the checkpoint.

        Args:
            save_to (str): Path to save the checkpoint.
            tags (str): Tags to save.

        Returns:
            str: Path to the checkpoint.

        Raises:
            ValueError: If the save_to is not valid.
        """
        if save_to is None:
            fname = f"E{self.global_counters.epochs.train}S{self.global_counters.steps.train}.pth"
            save_to = os.path.join(self.run_info.ckpt_dir, fname)

        # handle repeated saving.
        _last_save_to = getattr(self, "_last_ckpt_save_to", None)
        setattr(self, "_last_ckpt_save_to", save_to)
        if _last_save_to == save_to: return

        keys = self._select_keys(tags)

        _checkpoint = {
            "tasks":[task.state_dict() for task in self.run_info.tasks if isa(task, _Task)],
            "nodes":{key:self.nodes[key][1].state_dict() for key in keys if frozen(self.nodes[key][1])},
            "counters": self.global_counters,
        }

        if self.launcher.rank == 0:
            import torch
            tqdm.write(f"Saving checkpoint to \"{save_to}\".")
            torch.save(_checkpoint, save_to)

    def load_checkpoint(self, resume_from, strict=False, tags="*"):
        """Load the checkpoint.

        Args:
            resume_from (str): Path to the checkpoint.
            strict (bool): Whether to check the keys.
            tags (str): Tags to load.

        Raises:
            ValueError: If the resume_from is not valid.
        """
        import torch
        self.run_info.resume_from = resume_from
        if resume_from is None: return
        _checkpoint = torch.load(resume_from, map_location=self.launcher.assigned_device)

        try: # resuming task progress
            _tasks:List[Task] = [task for task in self.run_info.tasks if isa(task, _Task)]
            if strict and len(_tasks) != len(_checkpoint["tasks"]): raise ResumeTaskFailed()
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
            self.run_info._task_resumed = False

        keys = self._select_keys(tags)
        for key, node_state in _checkpoint["nodes"].items():
            if key not in keys: continue
            self.nodes[key][1].freeze().load_state_dict(node_state, strict=strict)

        self.global_counters = _checkpoint["counters"]

    def exec_tasks(self, tasks, launcher:ElasticLauncher):
        """Execute the tasks.

        Args:
            tasks (List[Task]): Tasks to execute.
            launcher (ElasticLauncher): Launcher.

        Returns:
            List[Task]: Tasks executed.

        Raises:
            ValueError: If the tasks are not valid.
        """
        for task in as_list(tasks):
            if is_configurable(task):
                task.freeze()
            if isa(task, _Task):
                if isa(task, Task) and not task.finished:
                    self.run_info._task_resumed = True
                task(self, launcher)
            elif isa(task, callable):
                if task in (SaveCheckpointTask, LoadCheckpointTask):
                    raise TypeError(f"Task {task} need to be called.")
                if self.run_info._task_resumed:
                    args = [x for x, _ in zip(
                        [self, launcher],
                        signature(task).parameters
                    )]
                    task(*args)
            else:
                get_logger().warning(f"A custom task `{task}` is not callable, skipping.")
    
    @overload
    def __call__(self, *, train=True, steps=1, tags=["*"]) -> GraphOutputCache: ...
    
    @overload
    def __call__(self, tasks) -> GraphOutputCache: ...

    def __call__(self, tasks=None, train=True, steps=1, tags=["*"]) -> GraphOutputCache:
        assert in_main_process(), "Eager Mode execution is only available in main process"
        enable_auto_freeze(True)
        if tasks is None:
            tasks = [Task(train=train, steps=steps, tags=tags)]
        self._run_impl(tasks, get_current_launcher(), None)
        enable_auto_freeze(False)
        return tasks[-1].egraph.cache