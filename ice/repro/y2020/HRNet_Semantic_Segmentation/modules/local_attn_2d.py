import math

from pkg_resources import require

import ice
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

FLOAT_BITS = 32

KERNEL_SOURCE = """
//cuda
#define FOREACH_PK(pkx, pky, pk) for (int pky = -ray*diy, pk = 0; pky <= ray*diy; pky += diy) for (int pkx = -rax*dix; pkx <= rax*dix; pkx += dix, ++pk)
#define FOREACH(i, n) for(int i = 0; i < n; ++i)
#define CHECK_RANGE(y, x, H, W) ((y) < H && (x) < W && (y) >= 0 && (x) >=0)
#define GET_POSITION_NOCHECK(px, py)    int py = blockIdx.y * blockDim.y + threadIdx.y; int px = blockIdx.x * blockDim.x + threadIdx.x;
#define GET_POSITION(px, py)     GET_POSITION_NOCHECK(px, py); if (!CHECK_RANGE(py, px, Hq, Wq)) return;
#define COMMON_CTX_PARAMS float sqrt_d_k, const int rax, const int ray, const int dix, const int diy, const int BS, const int Ck, const int Cv, const int Cs, const int Hq, const int Wq
#define EPS 1e-6f

// #define USE_MEAN_A 1

__global__ void local_attn_2d(
    const Tensor<float, 4> *xq,
    const Tensor<float, 4> *xk,
    const Tensor<float, 4> *xv,
    Tensor<float, 4> *s,
    Tensor<float, 4> *y,
    COMMON_CTX_PARAMS
){
    GET_POSITION(px, py);
    int pny, pnx;

    FOREACH(bs, BS) {
        // We calculate raw affinity scores (inner product between query and key) in current position,
        // and simultaineously find the maximum among them.
        float a;
        float norm_a = 0.f;
        FOREACH_PK(pkx, pky, pk) {
            pny = (py+pky); pnx = (px+pkx);
            a = 0.f;
            if (CHECK_RANGE(pny, pnx, Hq, Wq)) {
                FOREACH(ck, Ck) {
                    a += (*xq)[bs][ck][py][px] * (*xk)[bs][ck][pny][pnx];
                }
                a /= sqrt_d_k;
#ifdef USE_MEAN_A
                norm_a += a;                  // pytorch implements mean_a;
#else
                if (a>norm_a) {norm_a = a;}   // we will go for max_a for better numerical stability
#endif
            }
            (*s)[bs][pk][py][px] = a;
        }
#ifdef USE_MEAN_A
        norm_a /= Cs;
#endif
        // We calculate numerically stable softmax in following two loops.
        float sum_exp_a = EPS;
        FOREACH_PK(pkx, pky, pk) {
            a = exp((*s)[bs][pk][py][px] - norm_a);
            sum_exp_a += a;
            (*s)[bs][pk][py][px] = a;
        }
        FOREACH_PK(pkx, pky, pk) {
            (*s)[bs][pk][py][px] /= sum_exp_a;
        }
        FOREACH(cy, Cv) {
            float tmp_y = 0.f;
            FOREACH_PK(pkx, pky, pk) {
                pny = (py+pky); pnx = (px+pkx);
                if (!CHECK_RANGE(pny, pnx, Hq, Wq)) continue;
                tmp_y += (*s)[bs][pk][py][px] * (*xv)[bs][cy][pny][pnx];
            }
            (*y)[bs][cy][py][px] = tmp_y;
        }
    } // FOREACH(bs, BS)
} // KERNEL local_attn_2d

__global__ void local_attn_2d__gd_a_xv(
    const Tensor<float, 4> *gd_y,
    const Tensor<float, 4> *xv,
    const Tensor<float, 4> *s,
    Tensor<float, 3> *gd_s, // temp
    Tensor<float, 4> *gd_a, // out
    Tensor<float, 4> *gdxv, // out
    COMMON_CTX_PARAMS
){
    GET_POSITION(px, py);
    int pnx, pny;
    float tmp;
    
    FOREACH(bs, BS) {

        // <- gd_y, xv(shifted)
        FOREACH_PK(pkx, pky, pk) {
            pny = (py+pky); pnx = (px+pkx);
            tmp = 0.f;
            if (CHECK_RANGE(pny, pnx, Hq, Wq)) {
                FOREACH(cy, Cv) {
                    tmp += (*gd_y)[bs][cy][py][px] * (*xv)[bs][cy][pny][pnx];
                }
            }
            (*gd_s)[pk][py][px] = tmp;
        }// -> gd_s (temp)

        // <- gd_s, s
        FOREACH(j, Cs) {
            tmp = 0.f;
            FOREACH(i, Cs) {
                tmp += (*gd_s)[i][py][px] * (*s)[bs][i][py][px] * (static_cast<float>((i==j)?1.0f:0.0f) - (*s)[bs][j][py][px]);
            }
            (*gd_a)[bs][j][py][px] = tmp;
        }// -> gd_a (out)

        // <- gd_y(shifted), s(shifted)
        FOREACH(cv, Cv) {
            tmp = 0.f;
            FOREACH_PK(pkx, pky, pk) {
                pny = (py-pky); pnx = (px-pkx);
                if (!CHECK_RANGE(pny, pnx, Hq, Wq)) continue;
                tmp += (*gd_y)[bs][cv][pny][pnx] * (*s)[bs][pk][pny][pnx];
            }
            (*gdxv)[bs][cv][py][px] = tmp;
        }// -> gdxv (out)

    }// FOREACH(bs, BS)
}// KERNEL local_attn_2d__gd_a_xv

__global__ void local_attn_2d_gd_xq_xk(
    const Tensor<float, 4> *xq,
    const Tensor<float, 4> *xk,
    const Tensor<float, 4> *gd_a,
    Tensor<float, 4> *gdxq, // out
    Tensor<float, 4> *gdxk, // out
    COMMON_CTX_PARAMS
){
    GET_POSITION(px, py);
    float tmp_xq, tmp_xk;
    FOREACH(bs, BS) {

        // <- gd_a(shifted), xk(shifted), xq(shifted)
        FOREACH(ck, Ck) {
            tmp_xq = 0.f; tmp_xk = 0.f;
            FOREACH_PK(pkx, pky, pk){
                if (CHECK_RANGE(py+pky, px+pkx, Hq, Wq)){
                    tmp_xq += (*gd_a)[bs][pk][py][px] * (*xk)[bs][ck][py+pky][px+pkx];
                }
                if (CHECK_RANGE(py-pky, px-pkx, Hq, Wq)){
                    tmp_xk += (*gd_a)[bs][pk][py-pky][px-pkx] * (*xq)[bs][ck][py-pky][px-pkx];
                }
            }
            (*gdxq)[bs][ck][py][px] = tmp_xq / sqrt_d_k;
            (*gdxk)[bs][ck][py][px] = tmp_xk / sqrt_d_k;            
        }// -> gdxq (out), gdxk (out)

    }// FOREACH(bs, BS)
}// local_attn_2d_gd_xq_xk

//!cuda
"""

cuda_kernel = ice.CUDAModule(KERNEL_SOURCE, int_bits=32, float_bits=FLOAT_BITS, boundscheck=False)


class _local_attn_2d(Function):

    @staticmethod
    def forward(ctx, xq, xk, xv, kernel_size, dilation):
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        if type(dilation) == int:
            dilation = (dilation, dilation)
        assert xq.shape == xk.shape and xq.shape[2:] == xv.shape[2:]
        ray, rax = kernel_size[0]//2, kernel_size[1]//2
        diy, dix = dilation[0], dilation[1]
        Cs = (ray * 2 + 1) * (rax * 2 + 1)
        BS, Ck, Cv, Hq, Wq = xq.size(0), xk.size(1), xv.size(1), xq.size(2), xq.size(3)
        sqrt_d_k = math.sqrt(Ck)
        s = torch.zeros((BS, Cs, Hq, Wq), device=xq.device, dtype=xq.dtype)
        y = torch.zeros((BS, Cv, Hq, Wq), device=xq.device, dtype=xq.dtype)
        ctx.params = (sqrt_d_k, rax, ray, dix, diy, BS, Ck, Cv, Cs, Hq, Wq)
        torch.cuda.synchronize()
        cuda_kernel.local_attn_2d(xq, xk, xv, s, y, *ctx.params, block=(16, 16, 1), grid=(Wq//16+1, Hq//16+1))
        torch.cuda.synchronize()
        ctx.save_for_backward(xq, xk, xv, s)
        return y

    @staticmethod
    def backward(ctx, gd_y):
        xq, xk, xv, s = ctx.saved_tensors
        sqrt_d_k, rax, ray, dix, diy, BS, Ck, Cv, Cs, Hq, Wq = ctx.params
        gd_s = torch.zeros([Cs, Hq, Wq], device=xq.device, dtype=xq.dtype)
        gdxq, gdxk, gdxv, gd_a = torch.zeros_like(xq), torch.zeros_like(xk), torch.zeros_like(xv), torch.zeros_like(s)
        torch.cuda.synchronize()
        cuda_kernel.local_attn_2d__gd_a_xv(gd_y, xv, s, gd_s, gd_a, gdxv, *ctx.params, block=(16, 16, 1), grid=(Wq//16+1, Hq//16+1))
        torch.cuda.synchronize()
        cuda_kernel.local_attn_2d_gd_xq_xk(xq, xk, gd_a, gdxq, gdxk, *ctx.params, block=(16, 16, 1), grid=(Wq//16+1, Hq//16+1))
        torch.cuda.synchronize()
        return gdxq, gdxk, gdxv, None, None


def local_attn_2d(xq, xk, xv, kernel_size=3, dilation=1):
    return _local_attn_2d.apply(xq, xk, xv, kernel_size, dilation)


if __name__ == "__main__":

    import ice

    def make_test_data(
        new_tensor=torch.rand,
        seed=45,
        dtype=torch.float,
        **kwds
    ):
        torch.manual_seed(seed)
        d = ice.ConfigDict(**kwds)

        d.setdefault("BS", 2)
        d.setdefault("Hq", 8)
        d.setdefault("Wq", 8)
        d.setdefault("ray", 1)
        d.setdefault("rax", 1)
        d.setdefault("diy", 1)
        d.setdefault("dix", 1)
        d.setdefault("Ck", 3)
        d.setdefault("Cv", 16)
        d.setdefault("Cs", (d.ray*2+1)*(d.rax*2+1))

        d.xq = new_tensor((d.BS, d.Ck, d.Hq, d.Wq), requires_grad=True, dtype=dtype, device="cuda")
        # d.xk = new_tensor((d.BS, d.Ck, d.Hq, d.Wq), requires_grad=True, dtype=dtype, device="cuda")
        d.xk = d.xq
        d.xv = new_tensor((d.BS, d.Cv, d.Hq, d.Wq), requires_grad=True, dtype=dtype, device="cuda")
        d.gd_y = new_tensor((d.BS, d.Cv, d.Hq, d.Wq), requires_grad=False, dtype=dtype, device="cuda")
        
        return d

    def test_local_attn_2d():

        def reference(**kwds):
            d = make_test_data(**kwds)
            unfold = nn.Unfold(kernel_size=(d.ray*2+1, d.rax*2+1), padding=(d.ray*d.diy, d.rax*d.dix), dilation=(d.diy, d.dix))
            xq_unf = d.xq.reshape(d.BS, d.Ck, d.Hq * d.Wq)
            xk_unf = unfold(d.xk).reshape(d.BS, d.Ck, d.Cs, d.Hq * d.Wq)
            a = torch.einsum("bcl,bcml->bml", xq_unf, xk_unf) / math.sqrt(d.Ck)
            s = F.softmax(a, dim=1)
            xv_unf = unfold(d.xv).reshape(d.BS, d.Cv, d.Cs, d.Hq * d.Wq)
            d.y = torch.einsum("bsl,bcsl->bcl", s, xv_unf).reshape(d.BS, d.Cv, d.Hq, d.Wq)
            return d

        def experiment(**kwds):
            d = make_test_data(**kwds)
            d.y = local_attn_2d(d.xq, d.xk, d.xv, kernel_size=(d.ray*2+1, d.rax*2+1), dilation=(d.diy, d.dix))
            return d
        
        PRINT_OPTS = dict(threshold=10, sci_mode=False, precision=2)
        DTYPE = {16: torch.float16, 32: torch.float32, 64: torch.float64}[FLOAT_BITS]
        ATOL = {16: 1e-3, 32: 1e-4, 64: 1e-8}[FLOAT_BITS]

        # KWDS = dict(BS=1, Hq=512, Wq=1024, Ck=64, Cv=64, diy=1, dix=1, dtype=DTYPE, new_tensor=torch.randn)
        KWDS = dict(BS=1, Hq=128, Wq=192, Ck=64, Cv=64, diy=4, dix=8, dtype=DTYPE, new_tensor=torch.randn)
        # KWDS = dict(BS=1, Hq=1024, Wq=2048, Ck=3, Cv=64, dtype=DTYPE)
        # KWDS = dict(BS=2, Hq=256, Wq=512, Ck=3, Cv=64, dtype=DTYPE)

        ref = None
        seed = 125

        expfuncs = (
            reference,
            experiment,
        )

        # profile memory
        for expfunc in expfuncs:
            name = expfunc.__name__
            # ice.print(ice.profile(lambda :expfunc(**KWDS).y, repeat=1000), uri=name)
            

        # test accuracy
        for _ in range(10):
            seed = seed * 2 + 1
            for expfunc in expfuncs:
                name = expfunc.__name__            
                d = expfunc(**KWDS, seed=seed)
                d.y.backward(d.gd_y)
                if name == "reference":
                    ref = d
                    continue
                
                for nv in ["y", "xv.grad", "xk.grad", "xq.grad"]:
                    v = eval(f"d.{nv}")
                    rv = eval(f"ref.{nv}")
                    if not torch.allclose(v, rv, atol=ATOL):
                        dv = torch.abs(v-rv)
                        ice.print(dv, uri=f"diff_abs_{name}_{nv}", **PRINT_OPTS)
                        break
                else:
                    print(f"{name} check passed")
                
                del d
        
    test_local_attn_2d()