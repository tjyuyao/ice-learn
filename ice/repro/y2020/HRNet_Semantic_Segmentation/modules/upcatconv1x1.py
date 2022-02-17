import ice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def upcatconv1x1(input_coarser, input_finer, weight, bias=None):
    xv, xq = input_coarser, input_finer
    BS = xq.size(0)
    Cy, Cv, Ck = weight.size(0), xv.size(1), xq.size(1)
    Hv, Wv, Hq, Wq = xv.size(2), xv.size(3), xq.size(2), xq.size(3)
    dey, dex = Hq // Hv, Wq // Wv

    Wxv = torch.matmul(weight[:, :Cv].view((1, Cy, Cv)),
                       xv.view((BS, Cv, Hv * Wv)))  # (BS, Cy, Hv*Wv)
    Wxv = Wxv.view((BS * Cy * Hv, 1, Wv, 1))
    Wxv = Wxv.expand((-1, dey, Wv, dex))
    Wxv = Wxv.reshape((BS, Cy, Hq * Wq))
    y = torch.baddbmm(Wxv, weight[:, Cv:].view((1, Cy, Ck)).expand(
        (BS, Cy, Ck)), xq.view((BS, Ck, Hq * Wq))).view((BS, Cy, Hq, Wq))
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y


@ice.configurable
class UpCatConv1x1(nn.Conv2d):
    
    def __init__(self,
                 in_coarser_channels,
                 in_finer_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        """Upsample coarser map, concatentate it with a finer map, then apply 1x1 convolution on it, in an memory-efficient way.

        Args:
            in_coarser_channels (int): #channels of spatially coaser but has richer feature. (in_coarser_channels >> in_finer_channels)
            in_finer_channels (int): #channels of spatially finer but has less feature.
            out_channels (int): #channels produced by the convolution
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
        
        Forward Args:
            input_coarser (Tensor): (BS, Cv, Hv, Wv)
            input_finer (Tensor): (BS, Cq, Hq, Wq)

        Returns:
            Tensor: (BS, Cy, Hy, Wy)
        
        Notes:
            The spatial size of the finer map should be divisible by that of the coarser map.
        """
        
        super().__init__(in_channels=in_coarser_channels + in_finer_channels,
                         out_channels=out_channels,
                         kernel_size=1,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        self.in_coarser_channels = in_coarser_channels
        self.in_finer_channels = in_finer_channels
    
    def forward(self, input_coarser: Tensor, input_finer: Tensor) -> Tensor:
        assert input_coarser.size(1) == self.in_coarser_channels, f"Expecting {self.in_coarser_channels}, got {input_coarser.size(1)} for coarser map channels"
        assert input_finer.size(1) == self.in_finer_channels, f"Expecting {self.in_finer_channels}, got {input_finer.size(1)} for finer map channels"
        return upcatconv1x1(input_coarser, input_finer, self.weight, self.bias)


if __name__ == "__main__":
    
    def make_test_data(
        new_tensor=torch.randn,
        seed=42,
        dtype=torch.float,
        **kwds
    ):
        torch.manual_seed(seed)
        d = ice.ConfigDict(**kwds)

        d.setdefault("BS", 2)
        d.setdefault("Hq", 8)
        d.setdefault("Wq", 8)
        d.setdefault("dey", 2)
        d.setdefault("dex", 2)
        d.setdefault("ray", 1)
        d.setdefault("rax", 1)
        d.setdefault("diy", 1)
        d.setdefault("dix", 1)
        d.setdefault("Ck", 3)
        d.setdefault("Cv", 16)
        d.setdefault("Cf", d.Ck + d.Cv)
        d.setdefault("Cy", 8)
        d.setdefault("Cs", (d.ray*2+1)*(d.rax*2+1))
        d.setdefault("Hv", d.Hq // d.dey)
        d.setdefault("Wv", d.Wq // d.dex)

        d.xq = new_tensor((d.BS, d.Ck, d.Hq, d.Wq), requires_grad=True, dtype=dtype, device="cuda")
        d.xk = new_tensor((d.BS, d.Ck, d.Hq, d.Wq), requires_grad=True, dtype=dtype, device="cuda")
        d.xv = new_tensor((d.BS, d.Cv, d.Hv, d.Wv), requires_grad=True, dtype=dtype, device="cuda")
        d.W = new_tensor((d.Cy, d.Cv + d.Ck, 1, 1), requires_grad=True, dtype=dtype, device="cuda")
        d.b = new_tensor((d.Cy, ), requires_grad=True, dtype=dtype, device="cuda")

        return d

    def test_upcatconv1x1():
        kwds = dict(BS=2, Hq=256, Wq=512, dey=4, dex=4, Ck=3, Cv=128, Cy=8, dtype=torch.double)
        # kwds = dict(BS=1, Hq=1024, Wq=2048, dey=4, dex=4, Ck=3, Cv=512, Cy=64, dtype=torch.float16)
        # kwds = dict(BS=1, Hq=1024, Wq=2048, dey=4, dex=4, Ck=3, Cv=256, Cy=32, dtype=torch.float64)
        def reference():
            d = make_test_data(**kwds)
            d.xv1 = F.interpolate(d.xv, (d.Hq, d.Wq), mode="nearest")
            d.xf = torch.cat((d.xv1, d.xq), dim=1)
            d.y = F.conv2d(d.xf, d.W, d.b)
            return d.y

        def experiment():
            d = make_test_data(**kwds)
            d.y = upcatconv1x1(d.xv, d.xq, d.W, d.b)
            return d.y

        results = {}
        for expfunc in (
            reference,
            experiment,
            ):
            name = expfunc.__name__
            # ice.print(ice.profile(expfunc, repeat=10), uri=name)
            results[name] = expfunc()
            if name == "reference": continue
            if not torch.allclose(results[name], results["reference"]):
                diff_y = torch.abs(results[name]-results["reference"]) / (torch.abs(results["reference"]) + 1e-6)
                ice.print(diff_y, uri=f"relative_error_{name}", sci_mode=False, precision=2)
            else:
                print(f"{name} check passed")
            del results[name]

    test_upcatconv1x1()
