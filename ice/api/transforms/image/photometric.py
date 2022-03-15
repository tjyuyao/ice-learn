import cv2
import numpy as np
import numpy.random as random

from ice.llutil.dictprocess import dictprocess


@dictprocess
def Normalize(img, mean, std, to_rgb=True):
    """Normalize the image and convert BGR2RGB.

    Args:
        img (np.ndarray): original image.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.

    Returns:
        dict: {'img': Normalized results, 'img_norm_cfg': {'mean': ..., 'std': ..., 'to_rgb':...}}
    """

    img = img.copy().astype(np.float32)
    mean=np.array(mean, dtype=np.float32)
    std=np.array(std, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return dict(
        img=img,
        img_norm_cfg=dict(mean=mean, std=std, to_rgb=to_rgb),
    )


@dictprocess
def ToTensor(img:np.ndarray):
    """
    Args:
        img (np.ndarray): (1) transpose (HWC->CHW), (2)to tensor
    Returns:
        a torch.Tensor
    """
    import torch
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return {"img": torch.from_numpy(img)}


@dictprocess
def PhotoMetricDistortion(
        img,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18):
    """Apply photometric distortion to image sequentially, every dictprocessation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness

    2. random contrast (mode 0)

    3. convert color from BGR to HSV

    4. random saturation

    5. random hue

    6. convert color from HSV to BGR

    7. random contrast (mode 1)

    Args:
        img (np.ndarray): imput image.
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    Returns:
        dict: distorted_image
    """

    contrast_lower, contrast_upper = contrast_range
    saturation_lower, saturation_upper = saturation_range

    def bgr2hsv(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def hsv2bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def convert(img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(img):
        """Brightness distortion."""
        if random.randint(2):
            return convert(
                img,
                beta=random.uniform(-brightness_delta,
                                    brightness_delta))
        return img

    def contrast(img):
        """Contrast distortion."""
        if random.randint(2):
            return convert(
                img,
                alpha=random.uniform(contrast_lower, contrast_upper))
        return img

    def saturation(img):
        """Saturation distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 1] = convert(
                img[:, :, 1],
                alpha=random.uniform(saturation_lower,
                                     saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(img):
        """Hue distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-hue_delta, hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def distorted(img):
        """Call function to perform photometric distortion on images.

        Args:
            img (np.ndarray): imput image.

        Returns:
            dict: Result dict with images distorted.
        """

        # random brightness
        img = brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = contrast(img)

        # random saturation
        img = saturation(img)

        # random hue
        img = hue(img)

        # random contrast
        if mode == 0:
            img = contrast(img)

        return img

    return distorted(img)