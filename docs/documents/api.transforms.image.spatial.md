<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/spatial.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `api.transforms.image.spatial`






**Global Variables**
---------------
- **interp_codes**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Resize`

```python
Resize(
    src,
    dsize_w: int = None,
    dsize_h: int = None,
    scale: float = None,
    interpolation: str = 'nearest',
    keep_ratio: bool = False,
    return_scale: bool = False
)
```



**Args:**


 - <b>`src`</b> (np.ndarray):  The input image to be resized.

 - <b>`dsize_w`</b> (int, optional):  Target width. Higher priority than `scale`. Defaults to None.

 - <b>`dsize_h`</b> (int, optional):  Target height. Should be assigned if dsize_w is specified. Defaults to None.

 - <b>`scale`</b> (float, optional):  Calculate target size using this scale. Defaults to None.

 - <b>`interpolation`</b> (str, optional):  Interpolation method, accepted values are "nearest", "bilinear", "bicubic", "area", "lanczos".. Defaults to "nearest".

 - <b>`keep_ratio`</b> (bool, optional):  If dsize is specified, the image will be rescaled as large as possible and keep its aspect ratio, and padding redundant areas with zeros. Defaults to False.

 - <b>`return_scale`</b> (bool, optional):  Whether to return actual `w_scale` and `h_scale`. Defaults to False.




**Raises:**


 - <b>`TypeError`</b>:  when only one of dsize_w and dsize_h is specified.

 - <b>`TypeError`</b>:  when scale is negative.




**Returns:**


 - <b>`tuple | ndarray`</b>:  (`resized_img`, `w_scale`, `h_scale`) or `resized_img`





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Crop`

```python
Crop(src: ndarray, roi: Tuple)
```

crop a region of interest from the `src` array.




**Args:**


 - <b>`src`</b> (np.ndarray):  source array (H, W) or (H, W, C)

 - <b>`roi`</b> (Tuple):  region of interst (top, bottom, left, right)




**Returns:**


 - <b>`np.ndarray`</b>:  `src[top:bottom, left:right, ...]`





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Flip`

```python
Flip(src: ndarray, direction: str = 'horizontal')
```

flip the `src` image.




**Args:**


 - <b>`src`</b> (np.ndarray):  image (H, W) or (H, W, C)

 - <b>`direction`</b> (str, optional):  choose in "horizontal" and "vertical". Defaults to "horizontal".




**Raises:**


 - <b>`ValueError`</b>:  bad direction value




**Returns:**


 - <b>`np.ndarray`</b>:  flipped image.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Pad`

```python
Pad(img, dst_w, dst_h, pad_val=0, padding_mode='constant')
```

Padding a image to target size.




**Args:**


 - <b>`src`</b> (np.ndarray):  Source image.

 - <b>`dst_w`</b> (int):  target width.

 - <b>`dst_h`</b> (int):  target height.

 - <b>`pad_val`</b> (int, optional):  padding value to be filled in. Defaults to 0. User should set the value to some value meant to be ignored.

 - <b>`padding_mode`</b> (str):  Type of padding. Should be: constant, edge,

 - <b>`reflect or symmetric. Default`</b>:  constant.



        - constant: pads with a constant value, this value is specified
 with pad_val.

        - edge: pads with the last value at the edge of the image.

        - reflect: pads with reflection of image without repeating the
 last value on the edge. For example, padding [1, 2, 3, 4]
 with 2 elements on both sides in reflect mode will result
 in [3, 2, 1, 2, 3, 4, 3, 2].

        - symmetric: pads with reflection of image repeating the last
 value on the edge. For example, padding [1, 2, 3, 4] with
 2 elements on both sides in symmetric mode will result in
 [2, 1, 1, 2, 3, 4, 4, 3]




**Returns:**


 - <b>`np.ndarray`</b>:  padded image.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SizeDivisorMultiple`

```python
SizeDivisorMultiple(img=None, w=None, h=None, divisor)
```

Returns a smallest but larger shape to ensure each edge to be multiple to some number.




**Args:**


 - <b>`img`</b> (np.ndarray, optional):  (H, W) or (H, W, 3), will extract image size if w or h not specified.

 - <b>`w`</b> (int, optional):  original width

 - <b>`h`</b> (int, optional):  original height

 - <b>`divisor`</b> (int, optional):  the returned edge is a multiple of this value.




**Returns:**


 - <b>`dict`</b>:  {"h": new_h, "w": new_w}





