from enum import Enum, auto
from typing import Any, Callable, Type, overload
from ice.core.graph import GraphOutputCache, Node
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
    ):
        ...

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        forward: Callable[["Node", "GraphOutputCache"], Any],
        weight: float = None,
        loss_mode: LossMode = LossMode.MANUAL,
    ):
        super().__freeze__()
        self.loss_fn = forward
        self.loss_mode = loss_mode
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
        self.egraph.losses_counter += 1
        return self.loss_fn(self, cache)

    def backward(self):
        self.egraph.losses_counter -= 1
        if self.loss_mode == LossMode.MANUAL:
            loss = self.forward() * self.weight
            if self.egraph.losses_counter > 0:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
        else:
            raise NotImplementedError(f"Unimplemented for LossMode `{self.loss_mode}`")