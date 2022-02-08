from torch import optim
from typing import List, Type, Dict, overload
from ice.llutil.config import Configurable
from ice.llutil.dictprocess import DictProcessor
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer


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
        optimizer_kwds: Dict,
        updators_per_step: List[DictProcessor] = [],
        updators_at_epoch_start: List[DictProcessor] = [],
        gradient_accumulation_steps: int = 1,
    ): ...
    
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        optimizer_class: Type[optim.Optimizer],
        optimizer_kwds: Dict,
        updators_per_step: List[DictProcessor] = [],
        updators_at_epoch_start: List[DictProcessor] = [],
        gradient_accumulation_steps: int = 1,
        *,
        params
    ):
        self.optimizer = ZeroRedundancyOptimizer(
            params, optimizer_class=optimizer_class, **optimizer_kwds)
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.every = gradient_accumulation_steps
        self.updators_at_epoch_start = updators_at_epoch_start
        self.updators_per_step = updators_per_step

    def epoch_start(self, epochs, steps):
        for updator in self.updators_at_epoch_start:
            for group in self.optimizer.param_groups:
                updator(dict(param_group=group,
                             trigger="epoch_start",
                             steps=steps, epochs=epochs))

    def update(self, epochs, steps):
        # gradient accumulation
        if steps % self.every: return

        # update the learning rate
        for updator in self.updators_per_step:
            for group in self.optimizer.param_groups:
                updator(dict(param_group=group,
                             trigger="step",
                             steps=steps, epochs=epochs))

        # update the network parameters and clear gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def state_dict(self, rank):
        self.optimizer.consolidate_state_dict()
        if rank == 0:
            _state_dict = self.optimizer.state_dict()
            return _state_dict
    
    def load_state_dict(self, _state_dict):
        self.optimizer.load_state_dict(_state_dict)