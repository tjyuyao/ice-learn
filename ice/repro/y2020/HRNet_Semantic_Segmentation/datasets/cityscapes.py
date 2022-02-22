import os.path as osp
import cv2
import numpy as np
from torch.utils.data import Dataset
import ice

@ice.configurable
class Cityscapes(Dataset):

    IMG_SUFFIX = '_leftImg8bit.png'
    SEG_SUFFIX = '_gtFine_labelTrainIds.png'
    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    def __init__(self, data_root, split):
        """Cityscapes Filename Dataset

        Args:
            data_root (str): root of dataset directory.
            split (str|List[str]): Choose from one or more of "train", "val", "test", "trainextra".

        **`__getitem__` Returns**:
            Dict : {'img_path': img_path(`_leftImg8bit.png`), 'seg_path': seg_map_path(`_gtFine_labelTrainIds.png`)}
        """
        self.filenames = []
        for split in ice.as_list(split):
            split_path = osp.join(data_root, f"{split}.txt")
            with open(split_path) as f:
                for line in f:
                    img_name = line.strip()
                    filename = dict(
                        img_path=img_name,
                        seg_path=img_name,
                    )
                    self.filenames.append(filename)

        self.img_prefix = osp.join(data_root, "leftImg8bit", split)
        self.seg_prefix = osp.join(data_root, "gtFine", split)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return dict(
            img_path = osp.join(self.img_prefix, self.filenames[idx]["img_path"] + self.IMG_SUFFIX),
            seg_path = osp.join(self.seg_prefix, self.filenames[idx]["img_path"] + self.SEG_SUFFIX),
        )


import ice.api.transforms as IT


def train_aug_pipeline(
    resize_ratio_low = .5,
    resize_ratio_high = 2.,
    crop_height = 512,
    crop_width = 1024,
    cat_max_ratio = 0.75,
    ignore_index = 255,
    random_flip_prob = .5,
    normalize_mean = [123.675, 116.28, 103.53],
    normalize_std = [58.395, 57.12, 57.375],
):
    return [
        # Load
        IT.image.Load(img_path="img_path", dst="img"),
        IT.semseg.LoadAnnotation(seg_path="seg_path", dst="seg"),
        # Random Resize
        IT.random.RandomFloats(low=resize_ratio_low, high=resize_ratio_high, dst="resize_ratio"),
        IT.image.spatial.Resize(scale="resize_ratio", src="img", dst="img"),
        IT.image.spatial.Resize(scale="resize_ratio", src="seg", dst="seg"),
        # Random Crop
        IT.semseg.RandomCrop(dst_h=crop_height, dst_w=crop_width, cat_max_ratio=cat_max_ratio, ignore_index=ignore_index,
                                img="img", seg="seg", dst=dict(img="img_roi", seg="seg_roi")),
        # Random Flip
        IT.random.RandomDo([IT.image.spatial.Flip(src="img", dst="img"),
                            IT.image.spatial.Flip(src="seg", dst="seg")], prob=random_flip_prob),
        # Photometric Augmentation
        IT.image.photometric.PhotoMetricDistortion(img="img", dst="img"),
        # Normalize & ToTensor
        IT.image.Normalize(
            img="img", dst="img", mean=normalize_mean,
            std=normalize_std, to_rgb=True),
        IT.image.ToTensor(img="img", dst="img"),
        IT.semseg.LabelToTensor(src="seg", dst="seg"),
        IT.Collect("img", "seg")
    ]

@ice.dictprocess
def make_edge_gt(seg):
    img_blur = cv2.GaussianBlur(seg, (3,3), 0)
    edge = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    edge = edge.astype(np.float) / 255.
    return edge

def train_aug_pipeline(
    resize_ratio_low = .5,
    resize_ratio_high = 2.,
    crop_height = 512,
    crop_width = 1024,
    cat_max_ratio = 0.75,
    ignore_index = 255,
    random_flip_prob = .5,
    normalize_mean = [123.675, 116.28, 103.53],
    normalize_std = [58.395, 57.12, 57.375],
):
    return [
        # Load
        IT.image.Load(img_path="img_path", dst="img"),
        IT.semseg.LoadAnnotation(seg_path="seg_path", dst="seg"),
        # Random Resize
        IT.random.RandomFloats(low=resize_ratio_low, high=resize_ratio_high, dst="resize_ratio"),
        IT.image.spatial.Resize(scale="resize_ratio", src="img", dst="img"),
        IT.image.spatial.Resize(scale="resize_ratio", src="seg", dst="seg"),
        # Random Crop
        IT.semseg.RandomCrop(dst_h=crop_height, dst_w=crop_width, cat_max_ratio=cat_max_ratio, ignore_index=ignore_index,
                                img="img", seg="seg", dst=dict(img="img_roi", seg="seg_roi")),
        # Random Flip
        IT.random.RandomDo([IT.image.spatial.Flip(src="img", dst="img"),
                            IT.image.spatial.Flip(src="seg", dst="seg")], prob=random_flip_prob),
        make_edge_gt(seg="seg", dst="edge"),
        # Photometric Augmentation
        IT.image.photometric.PhotoMetricDistortion(img="img", dst="img"),
        # Normalize & ToTensor
        IT.image.Normalize(
            img="img", dst="img", mean=normalize_mean,
            std=normalize_std, to_rgb=True),
        IT.image.ToTensor(img="img", dst="img"),
        IT.semseg.LabelToTensor(src="seg", dst="seg"),
        IT.Collect("img", "seg", "edge")
    ]

def eval_pipeline(
    normalize_mean = [123.675, 116.28, 103.53],
    normalize_std = [58.395, 57.12, 57.375],
):
    return [
        # Load
        IT.image.Load(img_path="img_path", dst="raw_img"),
        IT.semseg.LoadAnnotation(seg_path="seg_path", dst="seg"),
        make_edge_gt(seg="seg", dst="edge"),
        # Normalize & ToTensor
        IT.image.Normalize(
            img="raw_img", dst="img", mean=normalize_mean,
            std=normalize_std, to_rgb=True),
        IT.image.ToTensor(img="img", dst="img"),
        IT.semseg.LabelToTensor(src="seg", dst="seg"),
        IT.Collect("img", "seg", "raw_img", "img_path", "edge")
    ]