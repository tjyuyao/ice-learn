from __future__ import annotations

from collections import deque, abc
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union, overload

from ice.core.graph import GraphOutputCache, Node
from ice.llutil.argparser import as_dict, as_list, isa, parse_scalar
from ice.llutil.launcher.launcher import get_current_launcher
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
    
    def _sync_api(self):
        if get_current_launcher().eager_mode: return
        self.sync()


class DictMetric(Meter):  # metric is a container of meters but is also a meter itself

    @overload
    def __init__(self, meters: Dict[str, Meter]): ...

    @overload
    def __init__(self, meter: Meter): ...

    def __init__(self, meters):
        self.meters:Dict[str, Meter] = as_dict(meters, "__only_meter__")

    @property
    def meter(self):
        return self.meters["__only_meter__"]

    def __len__(self):
        return len(self.meters)
    
    def __getitem__(self, name):
        return self.meters[name]

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
            
    def _sync_api(self):
        for meter in self.meters.values():
            meter._sync_api()        

    def evaluate(self, *args, **kwds):
        self._sync_api()
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
        **resources,
    ):
        ...

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __freeze__(
        self,
        metric: Union[DictMetric, Meter],
        forward: Callable[["MetricNode", "GraphOutputCache"], Any],
        epoch_end: Callable[["MetricNode"], Any] = None,
        **resources,
    ):
        self.metric = metric if isa(metric, DictMetric) else DictMetric(metric)
        self.user_epoch_end_hook = epoch_end
        self.best_record = None
        super().__freeze__(forward, **resources)

        return self

    def save_best_ckpt(self, new_value, higher_better, save_to=None, tags="*"):
        import torch
        if isinstance(new_value, torch.Tensor):
            new_value = new_value.item()
        if self.best_record is None:
            new_best = True
        elif higher_better:
            new_best = new_value > self.best_record
        else:
            new_best = new_value < self.best_record
        if new_best:
            self.best_record = new_value
            self.egraph.hypergraph.save_checkpoint(save_to=save_to, tags=tags)

    def epoch_start(self):
        self.metric.reset()
        
    def epoch_end(self):
        if not self.training:
            self.tensorboard_log_metric()
               
        if self.user_epoch_end_hook:
            self.user_epoch_end_hook(self)
    
    def tensorboard_log_metric(self, metric_value=None, namespace=None):
        if namespace is None:
            namespace = self.name

        if metric_value is None:
            metric_value = self.metric.evaluate()
        
        if isinstance(metric_value, dict):
            for k, v in metric_value.items():
                self.tensorboard_log_metric(metric_value=v, namespace=f"{namespace}/{k}")
        else:
            self.board.add_scalar(namespace, parse_scalar(metric_value), global_step=self.global_train_steps)


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

    def sync(self):
        pass  # effectively do nothing
