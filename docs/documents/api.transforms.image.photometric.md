<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/photometric.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `api.transforms.image.photometric`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Normalize`

```python
Normalize(img, mean, std, to_rgb=True)
```

Normalize the image and convert BGR2RGB.




**Args:**


 - <b>`img`</b> (np.ndarray):  original image.

 - <b>`mean`</b> (sequence):  Mean values of 3 channels.

 - <b>`std`</b> (sequence):  Std values of 3 channels.

 - <b>`to_rgb`</b> (bool):  Whether to convert the image from BGR to RGB,
 default is true.




**Returns:**


 - <b>`dict`</b>:  {'img': Normalized results, 'img_norm_cfg': {'mean': ..., 'std': ..., 'to_rgb':...}}





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ToTensor`

```python
ToTensor(img: ndarray)
```



**Args:**


 - <b>`img`</b> (np.ndarray):  (1) transpose (HWC->CHW), (2)to tensor


**Returns:**

a torch.Tensor





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `PhotoMetricDistortion`

```python
PhotoMetricDistortion(
    img,
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18
)
```

Apply photometric distortion to image sequentially, every dictprocessation
is applied with a probability of 0.5. The position of random contrast is in
second or second to last.


1. random brightness


2. random contrast (mode 0)


3. convert color from BGR to HSV


4. random saturation


5. random hue


6. convert color from HSV to BGR


7. random contrast (mode 1)




**Args:**


 - <b>`img`</b> (np.ndarray):  imput image.

 - <b>`brightness_delta`</b> (int):  delta of brightness.

 - <b>`contrast_range`</b> (tuple):  range of contrast.

 - <b>`saturation_range`</b> (tuple):  range of saturation.

 - <b>`hue_delta`</b> (int):  delta of hue.


**Returns:**


 - <b>`dict`</b>:  distorted_image





