from __future__ import annotations

from ice.llutil.argparser import as_list
from typing import TYPE_CHECKING, List, Type, Dict, overload

from ice.llutil.config import Configurable
from ice.llutil.dictprocess import DictProcessor
from ice.llutil.launcher.launcher import get_current_launcher

if TYPE_CHECKING:
    from torch import optim
    from torch.cuda.amp.grad_scaler import GradScaler


class Optimizer(Configurable):
    """Optimizer configuration API for ice-learn.

    This is an extension of `torch.optim.Optimizer` that:
    - allows the user to update the optimizer states using ``ice.DictProcessor``,
    - leverages `torch.ZeroRedundancyOptimizer` inside for memory efficient distributed training,
    - is able to accumulate gradients for simulating large batch size,
    - etc.

    **Inspired by:**
    - https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
    - https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    - https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
    """
    
    @overload
    def __init__(
        self,
        optimizer: Type[optim.Optimizer],
        updators: List[DictProcessor] = [],
    ): ...
    
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        optimizer: Type[optim.Optimizer],
        updators: List[DictProcessor] = [],
        *,
        params
    ):
        if get_current_launcher().eager_mode:
            self.optimizer = optimizer(params)
        else:
            from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
            self.optimizer = ZeroRedundancyOptimizer(
                params, optimizer_class=optimizer)
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.updators = as_list(updators)
        
    @overload
    def update(self, grad_scaler:GradScaler, grad_acc_steps:int, *, current_epoch, epoch_steps, global_steps, epoch_size): ...

    def update(self, grad_scaler:GradScaler, grad_acc_steps:int, optim_steps, module_node):
        # gradient accumulation
        if optim_steps % grad_acc_steps: return
        import torch

        # update the learning rate
        for updator in self.updators:
            for group in self.optimizer.param_groups:
                updator(dict(n=module_node, x=group, module_node=module_node, param_group=group))
                # scale gradients according to gradient accumulate steps. (assuming mean reduce of loss in batch dimension.)
                if grad_acc_steps != 1:
                    with torch.no_grad():
                        for p in group['params']:
                            p.grad = p.grad / grad_acc_steps

        # update the network parameters and clear gradients
        grad_scaler.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        
    def state_dict(self, rank):
        self.optimizer.consolidate_state_dict()
        if rank == 0:
            _state_dict = self.optimizer.state_dict()
            return _state_dict
    
    def load_state_dict(self, _state_dict):
        self.optimizer.load_state_dict(_state_dict)