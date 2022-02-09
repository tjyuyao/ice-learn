__all__ = ['RandomIntegers', 'RandomFloats', 'RandomProbabilities', 'RandomChoice', 'RandomROI', 'RandomDo',
           'RandomImage']


from typing import Callable, Dict, List

import numpy as np
from ice.llutil.argparser import as_list
from ice.llutil.dictprocess import dictprocess
from numpy.random import default_rng

_rng = default_rng()


@dictprocess
def RandomIntegers(low=0, high=None, size=None, dtype=np.int64, endpoint=False, tolist=False, rng=_rng):
    out = rng.integers(low, high=high, size=size, dtype=dtype, endpoint=endpoint)
    if tolist: out = out.tolist()
    return out


@dictprocess
def RandomFloats(low=0., high=1., size=None, dtype=np.float64, tolist=False, rng=_rng):
    if isinstance(high, (list, tuple)): high = np.array(high)
    if isinstance(low, (list, tuple)): low = np.array(low)

    out = (high - low) * rng.random(size=size, dtype=dtype) + low

    if tolist: out = out.tolist()
    return out


@dictprocess
def RandomProbabilities(size=None, dtype=np.float64, rng=_rng):
    out = rng.random(size=size, dtype=dtype)
    if isinstance(out, np.ndarray): out = out.tolist()
    return out


@dictprocess
def RandomChoice(candidates:List=None, return_index=False, rng=_rng):
    selected_index = rng.integers(len(candidates))
    selected_entry = candidates[selected_index]
    if return_index:
        return selected_entry, selected_index
    return selected_entry


@dictprocess
def RandomROI(*, img=None, img_w:int=None, img_h:int=None, roi_h:int, roi_w:int, rng=_rng):
    """generate random region of interest.

    Args:
        roi_h (int): roi height.
        roi_w (int): roi width.
        img (np.ndarray, optional): if `img_w` or `img_h` is not specified, infer from `img`. Defaults to None.
        img_w (int, optional): source image width. Defaults to None.
        img_h (int, optional): source image height. Defaults to None.
        rng (np.random.Generator, optional).

    Returns:
        Tuple[int]: (top, bottom, left, right)
    """
    if img_w is None or img_h is None:
        img_w = img.shape[1]
        img_h = img.shape[0]

    margin_h = max(img_h - roi_h, 0)
    margin_w = max(img_w - roi_w, 0)
    offset_h = rng.integers(0, margin_h + 1)
    offset_w = rng.integers(0, margin_w + 1)
    top, bot = offset_h, offset_h + roi_h
    lft, rgt = offset_w, offset_w + roi_w

    return top, bot, lft, rgt


def RandomDo(translist: List[Callable], prob:float=0.5, rng=_rng):
    def _dictprocess(data: Dict={}):
        if rng.random() < prob:
            for trans in as_list(translist):
                data = trans(data)
        return data
    return _dictprocess


@dictprocess
def RandomImage(img_w=640, img_h=480, color=True, low=0, high=255, dtype=np.uint8, rng=_rng):
    """generate a random image for testing purpose

    Args:
        img_w (int, optional): generated image width. Defaults to 640.
        img_h (int, optional): generated image height. Defaults to 480.
        color (bool, optional): if True, return (H, W, 3) else (H, W)-sized iamge. Defaults to True.
        low (int, optional): minimum pixel value. Defaults to 0.
        high (int, optional): maximum pixel value. Defaults to 255.
        dtype (optional): Defaults to np.uint8.
        rng (optional): np.random.Generator. Defaults to _rng.

    Returns:
        np.ndarray: image array (H, W) or (H, W, 3)
    """
    size = (img_h, img_w, 3) if color else (img_h, img_w)
    out = rng.integers(0, 255, size=size, dtype=dtype, endpoint=True)
    return out