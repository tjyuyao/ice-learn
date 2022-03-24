<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/random.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `api.transforms.random`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomIntegers`

```python
RandomIntegers(
    low=0,
    high=None,
    size=None,
    dtype=<class 'numpy.int64'>,
    endpoint=False,
    tolist=False,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomFloats`

```python
RandomFloats(
    low=0.0,
    high=1.0,
    size=None,
    dtype=<class 'numpy.float64'>,
    tolist=False,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomProbabilities`

```python
RandomProbabilities(
    size=None,
    dtype=<class 'numpy.float64'>,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomChoice`

```python
RandomChoice(
    candidates: List = None,
    return_index=False,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomROI`

```python
RandomROI(
    img=None,
    img_w: int = None,
    img_h: int = None,
    roi_h: int,
    roi_w: int,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```

generate random region of interest.




**Args:**


 - <b>`roi_h`</b> (int):  roi height.

 - <b>`roi_w`</b> (int):  roi width.

 - <b>`img`</b> (np.ndarray, optional):  if `img_w` or `img_h` is not specified, infer from `img`. Defaults to None.

 - <b>`img_w`</b> (int, optional):  source image width. Defaults to None.

 - <b>`img_h`</b> (int, optional):  source image height. Defaults to None.
rng (np.random.Generator, optional).




**Returns:**


 - <b>`Tuple[int]`</b>:  (top, bottom, left, right)





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/random.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomDo`

```python
RandomDo(
    translist: List[Callable],
    prob: float = 0.5,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomImage`

```python
RandomImage(
    img_w=640,
    img_h=480,
    color=True,
    low=0,
    high=255,
    dtype=<class 'numpy.uint8'>,
    rng=Generator(PCG64) at 0x7FE7B5806040
)
```

generate a random image for testing purpose




**Args:**


 - <b>`img_w`</b> (int, optional):  generated image width. Defaults to 640.

 - <b>`img_h`</b> (int, optional):  generated image height. Defaults to 480.

 - <b>`color`</b> (bool, optional):  if True, return (H, W, 3) else (H, W)-sized iamge. Defaults to True.

 - <b>`low`</b> (int, optional):  minimum pixel value. Defaults to 0.

 - <b>`high`</b> (int, optional):  maximum pixel value. Defaults to 255.

 - <b>`dtype`</b> (optional):  Defaults to np.uint8.

 - <b>`rng`</b> (optional):  np.random.Generator. Defaults to _rng.




**Returns:**


 - <b>`np.ndarray`</b>:  image array (H, W) or (H, W, 3)





