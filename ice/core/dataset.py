#export

import math
from ice.llutil.launcher.launcher import get_current_launcher
import numpy as np
import torch
import re
import collections
from torch._six import string_classes
from typing import Iterator, List, Callable, Optional, Dict, Union, overload

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler  # TODO: WeightedRandomSampler

from ice.llutil.config import freeze
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


class ResumableDistributedSampler(DistributedSampler):
    
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False, num_iters:int=None) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_batch_idx = 0
        self.num_iters = num_iters
        
    def set_start_batch_idx(self, i):
        self.start_batch_idx = i

    @property
    def indices(self) -> List:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        indices = indices[self.start_batch_idx:]
        self.start_batch_idx = 0

        if self.num_iters is not None:
            indices = indices[:self.num_iters]
        
        return indices
    
    def __iter__(self) -> Iterator:
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)


class _DatasetProxy(Dataset):

    def __init__(self, node, dataset, pipeline) -> None:
        super().__init__()
        self.node = node
        self._dataset = dataset
        self.pipeline = pipeline

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        sample = self._dataset.__getitem__(index)
        if self.pipeline is not None:
            sample = as_dict(sample, "sample")
            sample = self.pipeline(sample)
        return sample


class DatasetNode(Node):
    """Automating DataLoader and DataSampler creation and maintainance."""
    
    @overload
    def __init__(self,
                 dataset: Dataset,
                 shuffle: bool = False,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 batch_size_in_total: bool = False,
                 num_iters_per_epoch: int = None,
                 prefetch_factor: int = 2,
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
                 shuffle: bool = False,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 batch_size_in_total: bool = False,
                 num_iters_per_epoch: int = None,
                 prefetch_factor: int = 2,
                 worker_init_fn: Optional[Callable] = None,
                 persistent_workers: bool = False,
                 collate_fn: Optional[Callable] = failsafe_collate,
                 pipeline: Union[DictProcessor, List[DictProcessor]] = None,
                 ) -> None:
        
        super().__freeze__()
        freeze(dataset)
        pipeline = Compose(as_list(pipeline)) if isinstance(pipeline, list) else pipeline
        dataset = _DatasetProxy(self, dataset, pipeline)
        launcher = get_current_launcher()
        
        if batch_size_in_total:
            batch_size_in_total = batch_size
            batch_size_on_this_device = batch_size // launcher.world_size + (1 if launcher.rank <  batch_size % launcher.world_size else 0)
        else:
            batch_size_in_total = batch_size * launcher.world_size
            batch_size_on_this_device = batch_size
            
        # print(launcher.rank, batch_size_in_total, batch_size_on_this_device)

        self.sampler = ResumableDistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, num_iters=num_iters_per_epoch, seed=torch.random.initial_seed())
        self.loader = DataLoader(
                dataset,
                batch_size=batch_size_on_this_device,
                sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        self.internal_epoch = 0
        self.internal_steps = 0
        self.iterator = None
        
        self.actual_num_iters_per_epoch = math.ceil(len(self.sampler) / batch_size_on_this_device)
        self.batch_size_in_total = batch_size_in_total
        self.batch_size_on_this_device = batch_size_on_this_device
    
    def prepare(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

    def forward_impl(self, _):
        
        if self.iterator is None:
            self.iterator = iter(self.loader)
        
        try:
            sample = next(self.iterator)
            self.internal_steps += 1
        except StopIteration as e:
            # update states and iterator
            self.internal_steps = 0
            self.internal_epoch += 1
            self.sampler.set_epoch(epoch=self.internal_epoch)
            self.iterator = None
            # continue working
            if self.step_mode:
                self.iterator = iter(self.loader)
                sample = next(self.iterator)
            else:
                raise e

        return self.move(sample)
    
    def state_dict(self) -> Dict:
        _state_dict = {
            "internal_epoch": self.internal_epoch,
            "internal_steps": self.internal_steps,
        }
        return _state_dict
    
    def load_state_dict(self, _state_dict: Dict, strict: bool=None):
        self.internal_epoch = _state_dict["internal_epoch"]
        self.internal_steps = _state_dict["internal_steps"]
        self.sampler.set_epoch(epoch=self.internal_epoch)
        self.sampler.set_start_batch_idx(self.internal_steps)
    
    def __len__(self):
        return self.actual_num_iters_per_epoch