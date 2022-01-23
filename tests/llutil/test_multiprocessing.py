# # https://pytorch.org/tutorials/intermediate/dist_tuto.html

# import os

# import pytest
# import torch
# import torch.distributed as dist
# from ice import multiprocessing as mp

# BACKEND = "nccl" if torch.cuda.device_count() > 1 else "gloo"

# # tests

# def reduce(rank, size):
#     """ Simple collective communication. """
#     tensor = torch.Tensor([rank]).to(rank)
#     dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
#     assert tensor.item() == 1

# def all_gather_tensor(rank, size):
#     tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(size)]
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank

# def all_gather_object(rank, size):
#     pass

# # launcher

# def init_process(rank, size, fn, backend):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)

# def launch(fn, size=2):
#     processes = []
#     ctx = mp.get_context('spawn')
#     for rank in range(size):
#         p = ctx.Process(target=init_process, args=(rank, size, fn, BACKEND))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()
#         assert 0 == p.exitcode

# @pytest.mark.slow
# @pytest.mark.cuda
# def test_reduce():
#     launch(reduce, size=2)

# @pytest.mark.slow
# @pytest.mark.cuda
# def test_gather():
#     pass


import os
from ice.llutil.multiprocessing.launcher import ElasticLauncher


def run(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    print(local_rank, args)


if __name__ == "__main__":
    launch = ElasticLauncher()
    launch['nproc_per_node'] = 2
    launch.freeze()
    launch(run, lambda: 1)