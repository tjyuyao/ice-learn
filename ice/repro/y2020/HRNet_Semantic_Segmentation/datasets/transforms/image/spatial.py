
import numbers
import numpy as np
import cv2
from ice.llutil.dictprocess import dictprocess


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


@dictprocess
def Resize(src, dsize_w:int=None, dsize_h:int=None, scale:float=None, interpolation:str="nearest", keep_ratio:bool=False, return_scale:bool=False):
    """

    Args:
        src (np.ndarray): The input image to be resized.
        dsize_w (int, optional): Target width. Higher priority than `scale`. Defaults to None.
        dsize_h (int, optional): Target height. Should be assigned if dsize_w is specified. Defaults to None.
        scale (float, optional): Calculate target size using this scale. Defaults to None.
        interpolation (str, optional): Interpolation method, accepted values are "nearest", "bilinear", "bicubic", "area", "lanczos".. Defaults to "nearest".
        keep_ratio (bool, optional): If dsize is specified, the image will be rescaled as large as possible and keep its aspect ratio, and padding redundant areas with zeros. Defaults to False.
        return_scale (bool, optional): Whether to return actual `w_scale` and `h_scale`. Defaults to False.

    Raises:
        TypeError: when only one of dsize_w and dsize_h is specified.
        TypeError: when scale is negative.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or `resized_img`
    """

    if (dsize_w is None) ^ (dsize_h is None):
        raise TypeError("dsize_w and dsize_h should be specified at the sametime.")

    ssize_h, ssize_w = src.shape[:2]

    # calculates dsize (version 1)

    if dsize_w is not None and scale is not None:
        ... # TODO: log warning

    if dsize_w is None: # need to infer via scale factor
        if not (isinstance(scale, float) and scale > 0.):
            raise TypeError("When dsize is None, scale should be positive float.")
        dsize_h_v1 = int(ssize_h * scale + 0.5)
        dsize_w_v1 = int(ssize_w * scale + 0.5)
    elif keep_ratio: # dsize is definitely not None, but may need to calculate ratio.
        scale_tmp = min(dsize_h / ssize_h, dsize_w / ssize_w)
        dsize_h_v1 = int(ssize_h * scale_tmp + 0.5)
        dsize_w_v1 = int(ssize_w * scale_tmp + 0.5)
    else:
        dsize_h_v1 = dsize_h
        dsize_w_v1 = dsize_w

    resized_img = cv2.resize(
        src, (dsize_w_v1, dsize_h_v1), interpolation=interp_codes[interpolation])

    if not return_scale:
        return resized_img
    else:
        w_scale = dsize_w_v1 / ssize_w
        h_scale = dsize_h_v1 / ssize_h
        return resized_img, w_scale, h_scale


from typing import Tuple

@dictprocess
def Crop(src:np.ndarray, roi:Tuple):
    """crop a region of interest from the `src` array.

    Args:
        src (np.ndarray): source array (H, W) or (H, W, C)
        roi (Tuple): region of interst (top, bottom, left, right)

    Returns:
        np.ndarray: `src[top:bottom, left:right, ...]`
    """
    top, bot, lft, rgt = roi
    src = src[top:bot, lft:rgt, ...]
    return src


@dictprocess
def Flip(*, src:np.ndarray, direction:str="horizontal"):
    """flip the `src` image.

    Args:
        src (np.ndarray): image (H, W) or (H, W, C)
        direction (str, optional): choose in "horizontal" and "vertical". Defaults to "horizontal".

    Raises:
        ValueError: bad direction value

    Returns:
        np.ndarray: flipped image.
    """
    if direction == "horizontal":
        dst = src[:, ::-1, ...]
    elif direction == "vertical":
        dst = src[::-1, :, ...]
    else:
        raise ValueError(f"Expected direction in 'horizontal', 'vertical', got '{direction}'")
    return dst


@dictprocess
def Pad(img, dst_w, dst_h, pad_val=0, padding_mode="constant"):
    """Padding a image to target size.

    Args:
        src (np.ndarray): Source image.
        dst_w (int): target width.
        dst_h (int): target height.
        pad_val (int, optional): padding value to be filled in. Defaults to 0. User should set the value to some value meant to be ignored.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

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

    Returns:
        np.ndarray: padded image.
    """
    
    shape=(dst_h, dst_w)

    padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


@dictprocess
def SizeDivisorMultiple(*, img=None, w=None, h=None, divisor):
    """Returns a smallest but larger shape to ensure each edge to be multiple to some number.

    Args:
        img (np.ndarray, optional): (H, W) or (H, W, 3), will extract image size if w or h not specified.
        w (int, optional): original width
        h (int, optional): original height
        divisor (int, optional): the returned edge is a multiple of this value.

    Returns:
        dict: {"h": new_h, "w": new_w}
    """
    if w is None or h is None:
        h = img.shape[0]
        w = img.shape[1]

    h = int(np.ceil(h / divisor)) * divisor
    w = int(np.ceil(w / divisor)) * divisor
    return {"h": h, "w":w}