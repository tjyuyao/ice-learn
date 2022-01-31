#export

from random import seed
import numpy as np
import torch
import re
import collections
from torch._six import string_classes
from typing import List, Callable, Optional, Dict, Any, Union, overload
from copy import copy

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler, RandomSampler, WeightedRandomSampler

from ice.llutil.argparser import as_dict, as_list
from ice.core.graph import Node
from ice.llutil.dictprocess import Compose, DictProcessor


_NP_STR_OBJ_ARRAY_PATTERN = re.compile(r'[SaUO]')


_FAILSAFE_COLLATE_ERR_MSG_FORMAT = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def failsafe_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if _NP_STR_OBJ_ARRAY_PATTERN.search(elem.dtype.str) is not None:
                raise TypeError(_FAILSAFE_COLLATE_ERR_MSG_FORMAT.format(elem.dtype))

            return failsafe_collate([torch.as_tensor(np.ascontiguousarray(b)) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: failsafe_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(failsafe_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [failsafe_collate(samples) for samples in transposed]

    raise TypeError(_FAILSAFE_COLLATE_ERR_MSG_FORMAT.format(elem_type))


class DatasetNode(Node):
    """Automating DataLoader and DataSampler creation and maintainance."""
    
    @overload
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 num_workers: int = 0,
                 steps_per_epoch: int = None,
                 prefetch_factor: int = 2,
                 pin_memory: bool = False,
                 worker_init_fn: Optional[Callable] = None,
                 persistent_workers: bool = False,
                 collate_fn: Optional[Callable] = failsafe_collate,
                 pipeline: Union[DictProcessor, List[DictProcessor]] = None,
                 ) -> None:
        ...
    
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
    
    def __freeze__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 num_workers: int = 0,
                 steps_per_epoch: int = None,
                 prefetch_factor: int = 2,
                 pin_memory: bool = False,
                 worker_init_fn: Optional[Callable] = None,
                 persistent_workers: bool = False,
                 collate_fn: Optional[Callable] = failsafe_collate,
                 pipeline: Union[DictProcessor, List[DictProcessor]] = None,
                 ) -> None:
        
        super().__freeze__()
        
        pipeline = Compose(as_list(pipeline)) if isinstance(pipeline, list) else pipeline
        
        self.sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        self.loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        self.steps_per_epoch = steps_per_epoch
        self.internal_epoch = 0
        self.internal_steps = 0
        self.iterator = None
    
    def forward_impl(self, _):
        
        if self.iterator is None:
            self.iterator = iter(self.loader)
        
        try:
            sample = next(self.iterator)
        except StopIteration as e:
            self.internal_epoch += 1
            self.sampler.set_epoch(epoch=self.internal_epoch)
            if self.step_mode:
                sample = next(self.iterator)
            else:
                raise e
        
        return self.move(sample)
