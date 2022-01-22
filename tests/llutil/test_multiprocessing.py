# https://pytorch.org/tutorials/intermediate/dist_tuto.html

import os
from numpy import dtype
import torch
import torch.distributed as dist
from ice import multiprocessing as mp

BACKEND = "nccl" if torch.cuda.device_count() > 1 else "gloo"

def run(rank, size):
    """ Simple collective communication. """
    tensor = torch.Tensor([rank])
    if BACKEND == "nccl":
        tensor = tensor.to(rank)
    dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
    assert tensor.item() == 1

def init_process(rank, size, fn, backend):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def test_multiprocessing_reduce():
    size = 2
    processes = []
    ctx = mp.get_context('spawn')
    for rank in range(size):
        p = ctx.Process(target=init_process, args=(rank, size, run, BACKEND))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert 0 == p.exitcode

if __name__ == "__main__":
    test_multiprocessing_reduce()