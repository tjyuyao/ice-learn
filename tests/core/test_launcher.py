# https://pytorch.org/tutorials/intermediate/dist_tuto.html


# import pytest
import torch
import torch.distributed as dist
from ice.llutil.multiprocessing.launcher import ElasticLauncher


def sum_rank(launcher:ElasticLauncher):
    # print(launcher.rank)
    tensor = torch.Tensor([launcher.rank]).to(launcher.assigned_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    assert tensor.item() == (0 + launcher.world_size - 1) * launcher.world_size // 2
    

def test_launcher():
    launcher = ElasticLauncher(devices="cpu:0-3")
    launcher(sum_rank, launcher)