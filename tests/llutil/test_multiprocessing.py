# https://pytorch.org/tutorials/intermediate/dist_tuto.html

import os
import torch
import torch.distributed as dist
from ice import multiprocessing as mp

def run(rank, size):
    """ Simple collective communication. """
    tensor = torch.rand(1)
    print('Rank ', rank, ' has data ', tensor[0])
    dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()