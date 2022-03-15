import os.path as osp

import numpy as np
from ice.api.transforms.random import RandomROI, _rng
from ice.llutil.dictprocess import dictprocess
from ice.llutil.file_client import FileClient

from .image.io import imfrombytes
from .image.spatial import Crop


@dictprocess
def LoadAnnotation(
        seg_path,
        prefix=None,
        label_map=None,
        reduce_zero_label=False,
        imdecode_backend='pillow',
        file_client: FileClient = FileClient(backend='disk'),
):
    """Load annotations for semantic segmentation.

    **Args**:
        - reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        - imdecode_backend (str): Backend for `mmcv.imdecode`. Default: 'pillow'
        - file_client : See [mmcv.fileio.FileClient](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient) for details.

    **References**:
        - [mmseg.datasets.pipelines.LoadAnnotations](https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.LoadAnnotations)
    """

    if prefix is not None:
        full_seg_path = osp.join(prefix, seg_path)
    else:
        full_seg_path = seg_path

    seg_bytes = file_client.get(full_seg_path)
    seg = imfrombytes(seg_bytes, flag='unchanged',
                      backend=imdecode_backend).squeeze().astype(np.uint8)

    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            seg[seg == old_id] = new_id

    # reduce zero_label
    if reduce_zero_label:
        # avoid using underflow conversion
        seg[seg == 0] = 255
        seg = seg - 1
        seg[seg == 254] = 255

    return seg


@dictprocess
def RandomCrop(*, img=None, seg=None, dst_h:int, dst_w:int, cat_max_ratio=1., ignore_index=255, rng=_rng):
    """Crop a pair of images and segmentation labels such that the class is relatively balanced for training."""

    img_h, img_w = seg.shape[:2]

    def genroi():
        roi_cfg = RandomROI(img_w=img_w, img_h=img_h, roi_h=dst_h, roi_w=dst_w, rng=rng)()
        seg_roi = Crop(src=seg, roi=roi_cfg)()
        return roi_cfg, seg_roi

    roi_cfg, seg_roi = genroi()

    if cat_max_ratio < 1.:
        # Repeat 10 times
        for _ in range(10):
            labels, cnt = np.unique(seg_roi, return_counts=True)
            cnt = cnt[labels != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio: break
            roi_cfg, seg_roi = genroi()

    return dict(
        img_roi = Crop(src=img, roi=roi_cfg)() if img is not None else None,
        seg_roi = seg_roi,
        roi_cfg = roi_cfg,
    )


@dictprocess
def LabelToTensor(src:np.ndarray):
    """Convert to tensor (as int64).

    Args:
        src (np.ndarray): (H, W)-shaped integer map.

    Returns:
        a torch.Tensor
    """
    import torch
    segmap = src.astype(np.int64)
    return torch.from_numpy(segmap)
