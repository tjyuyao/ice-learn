from typing import Type
import ice
import torch
import torch.nn as nn
import os.path as osp
from hrnet import GuidedUpsampleConv1x1, HRNet18


def convert_pretrained(net: nn.Module, path: str, postfix="cvt"):
    """convert 3rd party pretrained weights into target module assuming the state order is same.

    find the mmcv pretrained weights here:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json

    Args:
        net_type (Type): ice-learn module type.
        path (str): saved module state_dict file.
    """
    ice_hrnet = net.state_dict()
    mmcv_hrnet = torch.load(path)
    # check compitability
    check_passed = True
    for (ki, vi), (km, vm) in zip(ice_hrnet.items(), mmcv_hrnet.items()):
        if vi.shape != vm.shape:
            print(ki, tuple(vi.shape), km, tuple(vm.shape))
            check_passed = False
    # convert state_dict
    if check_passed:
        state_dict = {ki:vm for ki, vm in zip(ice_hrnet.keys(), mmcv_hrnet.values())}
        name, ext = osp.splitext(path)
        torch.save(state_dict, f"{name}_{postfix}{ext}")
    else:
        print("I will not save the converted state_dict due to above mismatches.")


ice.run(
    lambda : convert_pretrained(
        HRNet18(GuidedUpsampleConv1x1()),
        "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006.pth",
        postfix="crela"
    )
)

# ice.run(
#     lambda : convert_pretrained(
#         HRNet18,
#         "/home/hyuyao/.cache/torch/hub/checkpoints/hrnetv2_w18-00eb2006.pth"
#     )
# )