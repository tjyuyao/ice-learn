<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/semseg.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `api.transforms.semseg`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `LoadAnnotation`

```python
LoadAnnotation(
    seg_path,
    prefix=None,
    label_map=None,
    reduce_zero_label=False,
    imdecode_backend='pillow',
    file_client: FileClient = <ice.llutil.file_client.FileClient object at 0x7f9507610730>
)
```

Load annotations for semantic segmentation.


**Args**:

    - reduce_zero_label (bool): Whether reduce all label value by 1.
 Usually used for datasets where 0 is background label.
 Default: False.

    - imdecode_backend (str): Backend for `mmcv.imdecode`. Default: 'pillow'

    - file_client : See [mmcv.fileio.FileClient](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient) for details.


**References**:

    - [mmseg.datasets.pipelines.LoadAnnotations](https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.LoadAnnotations)





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `RandomCrop`

```python
RandomCrop(
    img=None,
    seg=None,
    dst_h: int,
    dst_w: int,
    cat_max_ratio=1.0,
    ignore_index=255,
    rng=Generator(PCG64) at 0x7F95076083C0
)
```

Crop a pair of images and segmentation labels such that the class is relatively balanced for training.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `LabelToTensor`

```python
LabelToTensor(src: ndarray)
```

Convert to tensor (as int64).




**Args:**


 - <b>`src`</b> (np.ndarray):  (H, W)-shaped integer map.




**Returns:**

a torch.Tensor





