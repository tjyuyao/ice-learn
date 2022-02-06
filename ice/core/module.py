from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Set, Tuple, overload

import torch
import torch.nn as nn
from ice.core.graph import GraphOutputCache, Node
from ice.core.optim import Optimizer
from ice.llutil.argparser import as_dict, as_list
from ice.llutil.collections import Counter
from ice.llutil.logging import get_logger
from torch.nn.parallel import DistributedDataParallel


class _ModuleProxy(nn.Module):
    
    def __init__(self, node:"ModuleNode", module: nn.Module, forward: Callable[["ModuleNode", GraphOutputCache], Any]) -> None:
        super().__init__()
        self.node = node
        self._module = module
        self.forward_override = forward
    
    def forward(self, cache):
        return self.forward_override(self.node, cache)


class ModuleNode(Node):
    """a node that extends `torch.nn.Module`
    
    `ModuleNode` manages neural network modules (`torch.nn.Module`) and the optimizers responsible to train them. For each `ModuleNode`, multiple optimizers can be specified, each of which can be responsible for different group of parameters by filtering parameters names.
    
    Multiple `ModelNode` with different training configuration under differnt tags can share a same `torch.nn.Module`.
    """
    
    @overload
    def __init__(self,
                   module: nn.Module,
                   forward: Callable[["ModuleNode", GraphOutputCache], Any],
                   optimizers: Dict[Tuple[str], Optimizer] = None,
                   find_unused_parameters=False,
                   ): ...
    
    @overload
    def __init__(self,
                   module: nn.Module,
                   forward: Callable[["ModuleNode", GraphOutputCache], Any],
                   optimizers: Dict[Tuple[str], Optimizer] = None,
                   find_unused_parameters=False,
                   broadcast_buffers=True,
                   bucket_cap_mb=25,
                   gradient_as_bucket_view=False,
                   ): ...
    
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
    
    def __freeze__(self,
                   module: nn.Module,
                   forward: Callable[["ModuleNode", GraphOutputCache], Any],
                   optimizers: Dict[Tuple[str], Optimizer] = None,
                   find_unused_parameters=False,
                   broadcast_buffers=True,
                   bucket_cap_mb=25,
                   gradient_as_bucket_view=False,
                   ):
        super().__freeze__()
        self.module:nn.Module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module.freeze())
        self.move(self.module)
        self._ddp_module = DistributedDataParallel(
            _ModuleProxy(self, module, forward),
            broadcast_buffers=broadcast_buffers,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view)
        
        optim_cfgs = as_dict(optimizers, "*") if optimizers is not None else {}
        for param in self.module.parameters():
            param.requires_grad = False
        
        optimizers:List[Optimizer] = []
        trainable_params:Set[torch.nn.parameter.Parameter] = set()
        for patterns, optimizer in optim_cfgs.items():
            matched_params = []  # ! this should not be a set(), which will cause DDP error.
            for param_uri, param in self.module.named_parameters():
                for pattern in as_list(patterns):
                    if fnmatch(param_uri, pattern):
                        matched_params.append(param)
                        param.requires_grad = True
                        trainable_params.add(param)
                        break
                else:
                    get_logger().warning(f"pattern `{pattern}` does not match any parameters in `{module}`.")
            optimizer['params'] = matched_params
            optimizers.append(optimizer.freeze())
            
        untrainable_params:Set[torch.nn.parameter.Parameter] = set()
        for param in self.module.parameters():
            if param.requires_grad == False:
                untrainable_params.add(param)
        
        self.optimizers = optimizers
        self.optimizable = len(trainable_params) > 0
        self.trainable_params = trainable_params
        self.untrainable_params = untrainable_params
        
        self.optim_counter = Counter()

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
        for optimizer in self.optimizers:
            optimizer.epoch_start(self.optim_counter.epochs,
                                  self.optim_counter.steps)
            
    def epoch_end(self):
        if self.training:
            self.optim_counter.epochs += 1
    
    def forward_impl(self, cache: "GraphOutputCache"):
        return self._ddp_module(cache)
    
    def update(self):
        if not self.training: return
        for optimizer in self.optimizers:
            optimizer.update(self.optim_counter.epochs,
                             self.optim_counter.steps)
            self.optim_counter.steps += 1