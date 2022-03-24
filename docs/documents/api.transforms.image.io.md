<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `api.transforms.image.io`






**Global Variables**
---------------
- **IMREAD_COLOR**
- **IMREAD_GRAYSCALE**
- **IMREAD_IGNORE_ORIENTATION**
- **IMREAD_UNCHANGED**
- **TJCS_RGB**
- **TJPF_GRAY**
- **TJPF_BGR**
- **TurboJPEG**
- **tifffile**
- **jpeg**
- **supported_backends**
- **imread_flags**
- **imread_backend**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/io.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `use_backend`

```python
use_backend(backend)
```

Select a backend for image decoding.




**Args:**


 - <b>`backend`</b> (str):  The image decoding backend type. Options are `cv2`,

 - <b>``pillow`, `turbojpeg` (see https`</b>: //github.com/lilohuang/PyTurboJPEG)
and `tifffile`. `turbojpeg` is faster but it only supports `.jpeg`
file format.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/io.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `imread`

```python
imread(
    img_or_path,
    flag='color',
    channel_order='bgr',
    backend=None,
    file_client_args=None
)
```

Read an image.




**Note:**

> In v1.4.1 and later, add `file_client_args` parameters.




**Args:**


 - <b>`img_or_path`</b> (ndarray or str or Path):  Either a numpy array or str or
 pathlib.Path. If it is a numpy array (loaded image), then
 it will be returned as is.

 - <b>`flag`</b> (str):  Flags specifying the color type of a loaded image,
 candidates are `color`, `grayscale`, `unchanged`,
 `color_ignore_orientation` and `grayscale_ignore_orientation`.
 By default, `cv2` and `pillow` backend would rotate the image
 according to its EXIF info unless called with `unchanged` or
 `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
 always ignore image's EXIF info regardless of the flag.
 The `turbojpeg` backend only supports `color` and `grayscale`.

 - <b>`channel_order`</b> (str):  Order of channel, candidates are `bgr` and `rgb`.

 - <b>`backend`</b> (str | None):  The image decoding backend type. Options are
 `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
 If backend is None, the global imread_backend specified by

 - <b>``mmcv.use_backend()` will be used. Default`</b>:  None.

 - <b>`file_client_args`</b> (dict | None):  Arguments to instantiate a

 - <b>`FileClient. See `</b>: class:`mmcv.fileio.FileClient` for details.

 - <b>`Default`</b>:  None.




**Returns:**


 - <b>`ndarray`</b>:  Loaded image array.




**Examples:**

```python
import mmcv
    img_path = '/path/to/img.jpg'
    img = mmcv.imread(img_path)
    img = mmcv.imread(img_path, flag='color', channel_order='rgb',
#     ...     backend='cv2')
    img = mmcv.imread(img_path, flag='color', channel_order='bgr',
#     ...     backend='pillow')
    s3_img_path = 's3://bucket/img.jpg'
    # infer the file backend by the prefix s3
    img = mmcv.imread(s3_img_path)
    # manually set the file backend petrel
    img = mmcv.imread(s3_img_path, file_client_args={
#     ...     'backend': 'petrel'})
    http_img_path = 'http://path/to/img.jpg'
    img = mmcv.imread(http_img_path)
    img = mmcv.imread(http_img_path, file_client_args={
#     ...     'backend': 'http'})
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/io.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `imfrombytes`

```python
imfrombytes(content, flag='color', channel_order='bgr', backend=None)
```

Read an image from bytes.




**Args:**


 - <b>`content`</b> (bytes):  Image bytes got from files or other streams.

 - <b>`flag`</b> (str):  Same as :func:`imread`.

 - <b>`backend`</b> (str | None):  The image decoding backend type. Options are
 `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
 None, the global imread_backend specified by `mmcv.use_backend()`

 - <b>`will be used. Default`</b>:  None.




**Returns:**


 - <b>`ndarray`</b>:  Loaded image array.




**Examples:**

```python
img_path = '/path/to/img.jpg'
    with open(img_path, 'rb') as f:
        img_buff = f.read()
    img = mmcv.imfrombytes(img_buff)
    img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
    img = mmcv.imfrombytes(img_buff, backend='pillow')
    img = mmcv.imfrombytes(img_buff, backend='cv2')
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/api/transforms/image/io.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `imwrite`

```python
imwrite(img, file_path, params=None, auto_mkdir=None, file_client_args=None)
```

Write image to file.




**Note:**

> In v1.4.1 and later, add `file_client_args` parameters.


Warning:
 The parameter `auto_mkdir` will be deprecated in the future and every
 file clients will make directory automatically.




**Args:**


 - <b>`img`</b> (ndarray):  Image array to be written.

 - <b>`file_path`</b> (str):  Image file path.

 - <b>`params`</b> (None or list):  Same as opencv :func:`imwrite` interface.

 - <b>`auto_mkdir`</b> (bool):  If the parent folder of `file_path` does not exist,
 whether to create it automatically. It will be deprecated.

 - <b>`file_client_args`</b> (dict | None):  Arguments to instantiate a

 - <b>`FileClient. See `</b>: class:`mmcv.fileio.FileClient` for details.

 - <b>`Default`</b>:  None.




**Returns:**


 - <b>`bool`</b>:  Successful or not.




**Examples:**

```python
# write to hard disk client
    ret = mmcv.imwrite(img, '/path/to/img.jpg')
    # infer the file backend by the prefix s3
    ret = mmcv.imwrite(img, 's3://bucket/img.jpg')
    # manually set the file backend petrel
    ret = mmcv.imwrite(img, 's3://bucket/img.jpg', file_client_args={
#     ...     'backend': 'petrel'})
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Load`

```python
Load(
    img_path,
    prefix=None,
    to_float32=False,
    flag='color',
    channel_order='bgr',
    backend='cv2',
    file_client: FileClient = <ice.llutil.file_client.FileClient object at 0x7fe7b5817280>
)
```

**Args**:
 * flag (str): Flags specifying the color type of a loaded image,
 candidates are `color`, `grayscale`, `unchanged`,
 `color_ignore_orientation` and `grayscale_ignore_orientation`.
 By default, `cv2` and `pillow` backend would rotate the image
 according to its EXIF info unless called with `unchanged` or
 `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
 always ignore image's EXIF info regardless of the flag.
 The `turbojpeg` backend only supports `color` and `grayscale`.
 * channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
 * backend (str | None): The image decoding backend type. Options are
 `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
 If backend is None, the global imread_backend specified by
 `mmcv.use_backend()` will be used. Default: None.
 * file_client : See [mmcv.fileio.FileClient](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient) for details.


**References**:

    - [mmseg.datasets.pipelines.LoadImageFromFile](https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.LoadImageFromFile)

    - [mmcv.image.imfrombytes()](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.image.imfrombytes)

    - [mmcv.image.imread()](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.image.imread)





