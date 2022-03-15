from __future__ import annotations

from collections import deque, abc
from copy import deepcopy
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union, overload

from ice.core.graph import GraphOutputCache, Node
from ice.llutil.argparser import as_dict, as_list, isa, parse_scalar
from ice.llutil.logger import get_logger

if TYPE_CHECKING:
    import torch

class Meter:
    """value reducer that works recursively."""

    def reset(self):
        raise NotImplementedError()

    def update(self, value, *args):
        raise NotImplementedError()

    def evaluate(self, *args, **kwds):
        raise NotImplementedError()
    
    def sync(self):
        if not hasattr(self, "warn_flag"):
            self.warn_flag = True
            get_logger().warning(f"{self.__class__} not implementing `sync` method for multi-gpu synchronization.")


class DictMetric(Meter):  # metric is a container of meters but is also a meter itself

    @overload
    def __init__(self, meters: Dict[str, Meter]): ...

    @overload
    def __init__(self, meter_prototype: Meter): ...

    def __init__(self, meters = None, meter_prototype = None):
        if isa(meters, Meter) and meter_prototype is None:
            meter_prototype = meters
            meters = {}
        elif isa(meters, dict) and meter_prototype is None:
            meter_prototype = None
            meters = meters
        elif meters is None and isa(meter_prototype, Meter):
            meter_prototype = meter_prototype
            meters = {}
        else:
            raise TypeError()
        self.meters:Dict[Meter] = {}
        self.meter_prototype = meter_prototype
        self.update_argnames = list(signature(meter_prototype.update).parameters.keys())
    
    def __len__(self):
        return len(self.meters)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, explicit={}, *shared_args, **kwds):
        implicit = {}
        shared_kwds = {}
        for k, v in kwds.items():
            if k in self.update_argnames:
                shared_kwds[k] = v
            else:
                implicit[k] = v
        if isa(explicit, abc.Mapping):
            explicit.update(implicit)
        else:
            explicit = as_dict(explicit, "__only_meter__")
        for k, item in explicit.items():
            if k not in self.meters:
                self.meters[k] = deepcopy(self.meter_prototype)
                self.meters[k].reset()
            self.meters[k].update(item, *shared_args, **shared_kwds)
            
    def sync(self):
        for meter in self.meters.values():
            meter.sync()        

    def evaluate(self, *args, **kwds):
        self.sync()
        if "__only_meter__" in self.meters:
            return self.meters['__only_meter__'].evaluate(*args, **kwds)
        else:
            return {k:m.evaluate(*args, **kwds) for k, m in self.meters.items()}


class MetricNode(Node):

    @overload
    def __init__(
        self,
        metric: Union[DictMetric, Meter],
        forward: Callable[["MetricNode", "GraphOutputCache"], Any],
        epoch_end: Callable[["MetricNode"], Any] = None,
        higher_better: bool = True,
        trigger_saving_best_ckpt: bool = False,
        trigger_saving_passing_score=None,
    ):
        ...

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        metric: Union[DictMetric, Meter],
        forward: Callable[["MetricNode", "GraphOutputCache"], Any],
        epoch_end: Callable[["MetricNode"], Any] = None,
        higher_better: bool = True,
        trigger_saving_best_ckpt: bool = False,
        trigger_saving_passing_score=None,
    ):
        super().__freeze__(forward)

        self.metric = metric if isa(metric, DictMetric) else DictMetric(metric)
        self.user_epoch_end_hook = epoch_end
        self.higher_better=higher_better
        self.trigger_saving_best_ckpt=trigger_saving_best_ckpt
        self.best_record=trigger_saving_passing_score

        return self

    def better(self, new_value) -> bool:
        if not self.trigger_saving_best_ckpt: return False
        if self.best_record is None: return True
        if self.higher_better:
            return new_value > self.best_record
        else:
            return new_value < self.best_record

    def epoch_start(self):
        self.metric.reset()
        
    def update(self):
        self.metric.update(*as_list(self.forward()))
        
    def epoch_end(self):
        if not self.training:
            self.tensorboard_log_metric(postfix="")
               
        if self.user_epoch_end_hook:
            self.user_epoch_end_hook(self)
    
    def tensorboard_log_metric(self, postfix=""):
        metric_value = self.metric.evaluate()
        if isinstance(metric_value, dict):
            for k, v in metric_value.items():
                self.board.add_scalar(f"{self.name}/{k}{postfix}", parse_scalar(v), global_step=self.global_train_steps)
        else:
            self.board.add_scalar(f"{self.name}{postfix}", parse_scalar(metric_value), global_step=self.global_train_steps)


class ValueMeter(Meter):

    def reset(self):
        self.value = 0

    def update(self, value: torch.Tensor):
        self.value = value.detach()

    def evaluate(self):
        return self.value

    def sync(self):
        pass  # effectively do nothing


class SummationMeter(Meter):

    def reset(self):
        self.unsync_summation: torch.Tensor = 0
        self.summation: torch.Tensor = 0

    def update(self, batch_sum: torch.Tensor):
        self.unsync_summation += batch_sum.detach()

    def evaluate(self):
        return self.summation
    
    def sync(self):
        if isa(self.unsync_summation, int) and self.unsync_summation == 0: return
        import torch
        import torch.distributed as dist
        dist.all_reduce(self.unsync_summation, op=dist.ReduceOp.SUM)
        self.summation += self.unsync_summation
        self.unsync_summation = torch.zeros_like(self.summation)


class AverageMeter(Meter):

    def reset(self):
        self.summation: torch.Tensor = 0
        self.count:int = 0
        self.unsync_summation: torch.Tensor = 0
        self.unsync_count:int = 0

    def update(self, batch_avg: torch.Tensor, count:int=1):
        batch_avg = batch_avg.detach()
        self.unsync_summation += batch_avg * count
        self.unsync_count += count

    def evaluate(self):
        return self.summation / self.count
    
    def sync(self):
        if isa(self.unsync_summation, int) and self.unsync_summation == 0: return
        import torch
        import torch.distributed as dist
        dist.all_reduce(self.unsync_summation, op=dist.ReduceOp.SUM)
        self.summation += self.unsync_summation
        self.unsync_summation = torch.zeros_like(self.summation)
        self.count += self.unsync_count * dist.get_world_size()
        self.unsync_count = 0

class MovingAverageMeter(Meter):

    def __init__(self, window_size:int) -> None:
        self.window_size:int = window_size

    def reset(self):
        self.values = deque(maxlen=self.window_size)

    def update(self, *values):
        for v in values:
            v = v.detach()
            self.values.append(v)

    def evaluate(self):
        return sum(self.values) / len(self.values)
