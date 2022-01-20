<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/cuda.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.cuda`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/cuda.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CUDAModule`
Just-In-Time compilation of a set of CUDA kernel functions and device functions from source. 

``ice.CUDAModule`` works differently compared to pycuda's ``SourceModule`` in following ways: 


- Support efficient multi-dimensional torch.Tensor access with optional boundary check. 
- Automatically handle scalar data type conversion from python/pytorch to c++. 
- Compile error message will report the original source code. 
- Easier API. 
- Configurable in the ice-learn eco-system. 



**Example:**
 

```python
M, N, K = 4, 4, 1
a = torch.rand((M, K), dtype=torch.float32).cuda()
b = torch.rand((K, N), dtype=torch.float32).cuda()
c = torch.empty((M, N), dtype=torch.float32).cuda()

kernels = ice.CUDAModule(r"""
__global__ void matmul(Tensor<float, 2> *a, Tensor<float, 2> *b, Tensor<float, 2> *c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return;
    for (int k = 0; k < K; ++k) {
        v += (*a)[m][k] * (*b)[k][n];
    }
    (*c)[m][n] = v;
}
""", float_bits=32).freeze()

kernels.matmul(a, b, c, M, N, K, grid=(N // 32 + 1, M // 32 + 1), block=(32, 32, 1))

torch.cuda.synchronize()
assert torch.allclose(c, torch.mm(a, b))
```
 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/cuda.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    source,
    float_bits,
    int_bits=32,
    include_dirs=[],
    boundscheck=True,
    **kwds
)
```

Setup the parameters for compiling a CUDA source. 



**Args:**
 
 - <b>`source`</b> (str):  CUDA C++ source string. 
 - <b>`float_bits`</b> (int):  bit width of float values used as tensor scalar. 
 - <b>`int_bits`</b> (int, optional):  bit width of default int scalar. Defaults to 32. 
 - <b>`include_dirs`</b> (list, optional):  paths of extra include dirs. Defaults to []. 
 - <b>`boundscheck`</b> (bool, optional):  enable out of bound check for tensors. Defaults to True. 
 - <b>`**kwds`</b>:  other keyword args you would like to pass to pycuda's ``SourceModule``. 



**Note:**

> Direct written `float` and `int` token in the source string will be substituted to ensure the default scalar data type matches the tensors. If you do not want this to happen, use more specific CUDA typename such as `__half`, `double`, `int16_t`, etc. 
>


