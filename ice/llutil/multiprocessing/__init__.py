""" a drop-in replacement for `torch.multiprocessing`.

ice.llutil.multiprocessing is a modified version of ``torch.multiprocessing``. It's designed to change
``import torch.multiprocessing as mp`` to ``from ice import multiprocessing as mp`` to have all the lambda functions, 
closures as well as pytorch tensors sent through processes in Data Distributed Parallel paradigm.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""
import torch
import sys
from .reductions import init_reductions
import multiprocess

__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocess import *  # noqa: F403


__all__ += multiprocess.__all__  # type: ignore[attr-defined]


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
torch._C._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them. This depends `mp.get_context` which was added in Python 3.4."""
from .spawn import (
    spawn,
    SpawnContext,
    start_processes,
    ProcessContext,
    ProcessRaisedException,
    ProcessExitedException,
)

from torch.multiprocessing import set_sharing_strategy, get_sharing_strategy, get_all_sharing_strategies
init_reductions()