from enum import Enum, auto
from typing import Any, Callable, Type, overload

from ice.core.graph import GraphOutputCache, Node
from ice.core.dataset import DatasetNode
from ice.llutil.argparser import isa


class LossMode(Enum):
    MANUAL = auto()
    MGDAUB = auto()


class LossNode(Node):

    @overload
    def __init__(
        self,
        forward: Callable[["Node", "GraphOutputCache"], Any],
        weight: float = None,
        loss_mode: LossMode = LossMode.MANUAL,
        **resources,
    ):
        ...

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        forward: Callable[["Node", "GraphOutputCache"], Any],
        weight: float = None,
        loss_mode: LossMode = LossMode.MANUAL,
        **resources,
    ):
        super().__freeze__(forward=None, **resources)
        self.loss_fn = forward
        self.loss_mode = loss_mode
        self.ddp_var_batch_reduce_weight = None
        if loss_mode == LossMode.MANUAL:
            if weight is None:
                self.weight = 1.0
            elif isa(weight, float):
                self.weight = weight
            else:
                raise TypeError(f"need a float, get `{weight}`.")
        else:
            raise NotImplementedError(f"Unimplemented for LossMode `{loss_mode}`")

    def forward_impl(self, cache: "GraphOutputCache"):
        from torch import autocast
        with autocast(self.launcher.assigned_device.type, **self.egraph.hypergraph.autocast_kwds):
            loss = self.loss_fn(self, cache)
            if self.ddp_var_batch_reduce_weight is None:
                datasets = cache.find_ancestors(self.name, lambda n: hasattr(n, "batch_size_in_total"))
                if datasets:
                    Bi = datasets[0].batch_size_on_this_device
                    Bt  = datasets[0].batch_size_in_total
                    self.ddp_var_batch_reduce_weight = self.launcher.world_size * Bi / Bt
                else:
                    self.ddp_var_batch_reduce_weight = 1.
            loss = loss * self.ddp_var_batch_reduce_weight
        if self.training:
            self.egraph.losses_counter += 1
            self.board.add_scalar(self.name, loss.item(), global_step=self.global_train_steps)
        return loss

    def backward(self):
        assert self.training
        self.egraph.losses_counter -= 1
        if self.loss_mode == LossMode.MANUAL:
            loss = self.forward() * self.weight
            self.egraph.total_loss = self.egraph.total_loss + loss
            if self.egraph.losses_counter == 0:
                scaled_total_loss = self.grad_scaler.scale(self.egraph.total_loss)
                scaled_total_loss.backward()
                self.egraph.total_loss = 0
        else:
            raise NotImplementedError(f"Unimplemented for LossMode `{self.loss_mode}`")