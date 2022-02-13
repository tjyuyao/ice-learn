import torch
import torch.nn.functional as F
import ice

def area_intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        ):
    """Calculate intersection and Union by area.

    Modified from https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/core/evaluation/metrics.html#intersect_and_union

    Args:
        pred_label (torch.Tensor): Prediction segmentation map.
        label (torch.Tensor): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    assert pred_label.shape == label.shape

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


@ice.configurable
class SemsegIoUMetric(ice.DictMetric):

    def __init__(self, num_classes:int, ignore_index:int=255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        super().__init__(ice.SummationMeter())

    def update(self, seg_logit, seg_label):
        seg_logit = seg_logit
        seg_logit = F.interpolate(
            seg_logit,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        seg_pred = torch.argmax(seg_logit, dim=1).flatten()
        seg_label = seg_label.flatten()
        intersect, union, _, _ = area_intersect_and_union(
            pred_label = seg_pred,
            label      = seg_label,
            num_classes = self.num_classes,
            ignore_index = self.ignore_index,
        )
        super().update(intersect=intersect, union=union)

    def evaluate(self):
        value = super().evaluate()
        value = value["intersect"] / (value["union"] + 1e-8)
        return value