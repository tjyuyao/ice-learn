from __future__ import annotations

import re
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple, overload
from ice.llutil.config import freeze

from ice.core.graph import GraphOutputCache, Node
from ice.core.optim import Optimizer
from ice.llutil.argparser import as_dict, as_list
from ice.llutil.collections import Counter
from ice.llutil.launcher.launcher import get_current_launcher
from ice.llutil.logger import get_logger

if TYPE_CHECKING:
    import torch.nn as nn


class ModuleNode(Node):
    """a node that extends `torch.nn.Module`

    `ModuleNode` manages neural network modules (`torch.nn.Module`) and the optimizers responsible to train them. For each `ModuleNode`, multiple optimizers can be specified, each of which can be responsible for different group of parameters by filtering parameters names.

    Multiple `ModelNode` with different training configuration under differnt tags can share a same `torch.nn.Module`.
    """

    @overload
    def __init__(
        self,
        module: nn.Module,
        forward: Callable[["ModuleNode", GraphOutputCache], Any],
        optimizers: Dict[Tuple[str], Optimizer] = None,
        weight_init_fn: Callable[[nn.Module], None] = None,
        static_graph=False,
        find_unused_parameters=False,
        **resources,
    ):
        ...

    @overload
    def __init__(
        self,
        module: nn.Module,
        forward: Callable[["ModuleNode", GraphOutputCache], Any],
        optimizers: Dict[Tuple[str], Optimizer] = None,
        weight_init_fn: Callable[[nn.Module], None] = None,
        static_graph=False,
        find_unused_parameters=False,
        broadcast_buffers=True,
        bucket_cap_mb=25,
        gradient_as_bucket_view=False,
        **resources,
    ):
        ...

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(self,
                   module: nn.Module,
                   forward: Callable[["ModuleNode", GraphOutputCache], Any],
                   optimizers: Dict[Tuple[str], Optimizer] = None,
                   weight_init_fn: Callable[[nn.Module], None] = None,
                   static_graph=False,
                   find_unused_parameters=False,
                   broadcast_buffers=True,
                   bucket_cap_mb=25,
                   gradient_as_bucket_view=False,
                   **resources,
                   ):
        import torch
        import torch.nn as nn
        super().__freeze__(forward=None, **resources)
        module = freeze(module)
        if weight_init_fn is not None:
            with torch.no_grad():
                weight_init_fn(module)
            sleep(0.2)
        has_nan:List[str] = [k for k, w in module.named_parameters() if torch.isnan(w.data).any().item()]
        if has_nan: get_logger().warn(f"initialized weight might has nan: {has_nan}")
        
        launcher = get_current_launcher()

        if launcher.eager_mode or launcher.assigned_device.type == "cpu":
            self.module = module
        else:
            self.module:nn.Module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        self.move(self.module)

        optim_cfgs = as_dict(optimizers, ".*") if optimizers is not None else {}
        for param in self.module.parameters():
            param.requires_grad = False

        optimizers_keys = []
        optimizers:List[Optimizer] = []
        trainable_params:Set[torch.nn.parameter.Parameter] = set()
        for patterns, optimizer in optim_cfgs.items():
            optimizers_keys.append(patterns)
            matched_params = []  # ! this should not be a set(), which will cause DDP error.
            matched_param_names = []
            for pattern in as_list(patterns):
                pattern_matched = False
                matcher = re.compile(pattern)
                for param_uri, param in self.module.named_parameters():
                    if matcher.match(param_uri):
                        matched_params.append(param)
                        matched_param_names.append(param_uri)
                        param.requires_grad = True
                        trainable_params.add(param)
                        pattern_matched = True
                if not pattern_matched:
                    get_logger().warning(f"pattern `{pattern}` does not match any parameters in `{module.__class__.__name__}`.")
            if pattern != ".*" and self.launcher.rank == 0:
                get_logger().info(f"matched_parameters for {optimizer} in {module.__class__.__name__}:\n{matched_param_names}")
            if matched_params:
                optimizer = optimizer(params=matched_params).freeze()
                optimizers.append(optimizer)

        untrainable_params:Set[torch.nn.parameter.Parameter] = set()
        for param in self.module.parameters():
            if param.requires_grad == False:
                untrainable_params.add(param)

        self.optimizers_keys = optimizers_keys
        self.optimizers = optimizers
        self.optimizable = len(trainable_params) > 0
        self.trainable_params = trainable_params
        self.untrainable_params = untrainable_params

        self.optim_counter = Counter()

        class _ModuleProxy(nn.Module):

            def __init__(self, node:"ModuleNode", module: nn.Module, forward: Callable[["ModuleNode", GraphOutputCache], Any]) -> None:
                super().__init__()
                self.node = node
                self._module = module
                self.forward_override = forward

            def forward(self, cache):
                from torch import autocast
                with autocast(self.node.launcher.assigned_device.type, **self.node.egraph.hypergraph.autocast_kwds):
                    return self.forward_override(self.node, cache)

        if self.optimizable and not launcher.eager_mode:
            from torch.nn.parallel import DistributedDataParallel
            self._ddp_module = DistributedDataParallel(
                _ModuleProxy(self, self.module, forward),
                broadcast_buffers=broadcast_buffers,
                bucket_cap_mb=bucket_cap_mb,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view)
            if static_graph:
                self._ddp_module._set_static_graph()
        else:
            self._ddp_module = _ModuleProxy(self, self.module, forward)

        return self

    def prepare(self):
        if self.training:
            for param in self.trainable_params:
                param.requires_grad = True
            for param in self.untrainable_params:
                param.requires_grad = False
                param.grad = None
            self._ddp_module.train()
        else:
            for param in self.module.parameters():
                param.requires_grad = False
                param.grad = None
            self._ddp_module.eval()

    def epoch_start(self):
        if self.training:
            self.optim_counter.epoch_steps = 0
    
    def epoch_end(self):
        if self.training:
            self.optim_counter.epochs += 1

    def forward_impl(self, cache: "GraphOutputCache"):
        return self._ddp_module(cache)

    def update(self):
        if not self.training: return
        for optimizer in self.optimizers:
            optimizer.update(
                self.grad_scaler,
                self.grad_acc_steps,
                optim_steps=self.optim_counter.steps,
                module_node=self,
            )
            self.optim_counter.steps += 1
            self.optim_counter.epoch_steps += 1

    def state_dict(self):
        _state_dict = {
            "module": self.module.state_dict(),
            "optim_counter": {k:v for k, v in self.optim_counter.items()},
            "optimizers_keys": {key:i for i, key in enumerate(self.optimizers_keys)},
            "optimizers": [optimizer.state_dict(self.launcher.rank) for optimizer in self.optimizers],
            "optimizers_backup_states": getattr(self, "optimizers_backup_states", {"optimizers_keys":{}})
        }
        return _state_dict

    def load_state_dict(self, _state_dict:Dict, strict:bool):
        import torch
        with torch.no_grad():
            # load module
            result = self.module.load_state_dict(_state_dict["module"], strict=strict)
            if not strict:
                if len(result.unexpected_keys) > 0:
                    get_logger().warn(
                        'Unexpected key(s) when loading a {}: {}. '.format(
                            self.module.__class__.__name__,
                            ', '.join('"{}"'.format(k) for k in result.unexpected_keys)))
                if len(result.missing_keys) > 0:
                    get_logger().warn(
                        'Missing key(s) when loading a {}: {}. '.format(
                            self.module.__class__.__name__,
                            ', '.join('"{}"'.format(k) for k in result.missing_keys)))
            # load optim_counter
            for k, v in _state_dict["optim_counter"].items():
                self.optim_counter[k] = v
            # load optimizers
            _backup_states = _state_dict["optimizers_backup_states"]
            if self.optimizers_keys == _state_dict["optimizers_keys"]:
                for optimizer, optim_states in zip(self.optimizers, _state_dict["optimizers"]):
                    optimizer.load_state_dict(optim_states)
            else:
                missing_keys:Set[str] = []
                unexpected_keys_in_state: Set[str] = set(_state_dict["optimizers_keys"].keys())
                unexpected_keys_in_backup: Set[str] = set(_backup_states["optimizers_keys"].keys())
                loaded_from_backup = set()
                for i, key in enumerate(self.optimizers_keys):
                    optimizer = self.optimizers[i]
                    if key in _state_dict["optimizers_keys"]:
                        j = _state_dict["optimizers_keys"][key]
                        optimizer.load_state_dict(_state_dict["optimizers"][j])
                        unexpected_keys_in_state.remove(key)
                    elif key in _backup_states["optimizers_keys"]:
                        j = _backup_states["optimizers_keys"][key]
                        optimizer.load_state_dict(_backup_states["optimizers"][j])
                        unexpected_keys_in_backup.remove(key)
                        loaded_from_backup.add(key)
                    else:
                        missing_keys.add(key)

                warn_flag = False
                warn_msgs = [f"optimizers_keys does not match for {self.module.__class__.__name__}.\n"]
                if len(missing_keys) > 0:
                    warn_msgs.append(
                        "Missing key(s) ({}) are not load.\n".format(
                            ', '.join('"{}"'.format(k) for k in missing_keys)
                        ))
                    warn_flag = True
                unexpected_keys = unexpected_keys_in_state.union(unexpected_keys_in_backup)
                if len(unexpected_keys) > 0:
                    warn_msgs.append(
                        "Unexpected key(s) ({}) are saved as a backup for future use.\n".format(
                            ', '.join('"{}"'.format(k) for k in unexpected_keys)
                        ))
                    warn_flag = True
                if len(loaded_from_backup) > 0:
                    warn_msgs.append(
                        "Following keys are loaded from a backup version: {}.\n".format(
                            ', '.join('"{}"'.format(k) for k in loaded_from_backup)
                        ))
                    warn_flag = True
                if warn_flag:
                    get_logger().warn(''.join(warn_msgs))

                # update backup states
                _new_backup_states = {
                    "optimizers_keys": {},
                    "optimizers": {},
                }
                u = 0
                for key in unexpected_keys_in_backup:
                    if key in unexpected_keys_in_state: continue
                    j = _backup_states["optimizers_keys"][key]
                    _new_backup_states["optimizers_keys"][key] = u
                    _new_backup_states["optimizers"][key] = _backup_states["optimizers"][j]
                    u += 1
                for key in unexpected_keys_in_state:
                    i = _state_dict["optimizers_keys"][key]
                    _new_backup_states["optimizers_keys"][key] = u
                    _new_backup_states["optimizers"][key] = _state_dict["optimizers"][i]
                    u += 1
                _backup_states = _new_backup_states
            # save as a backup
            setattr(self, "optimizers_backup_states", _backup_states)