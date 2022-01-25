import pytest
import torch
import torch.distributed as dist
from ice.llutil.launcher import ElasticLauncher
import ice


def sum_rank(launcher: ElasticLauncher):
    # print(launcher.rank)
    tensor = torch.Tensor([launcher.rank]).to(launcher.assigned_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    assert tensor.item() == (0 + launcher.world_size - 1) * launcher.world_size // 2


def test_sum_rank():
    launcher = ElasticLauncher(devices="cpu:0-3").freeze()
    launcher(sum_rank, launcher)


@ice.test.requires_n_gpus(2)
def test_multi_gpu():
    launcher = ElasticLauncher(devices="cuda:0,1").freeze()
    launcher(sum_rank, launcher)


@ice.test.requires_cuda
def test_single_gpu():
    launcher = ElasticLauncher(devices="cuda:0,0").freeze()
    launcher(sum_rank, launcher)