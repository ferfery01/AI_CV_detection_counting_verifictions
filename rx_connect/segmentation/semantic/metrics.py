from typing import Dict, List

import numpy as np
import segmentation_models_pytorch as smp
import torch
from determined.pytorch import MetricReducer


class SegmentMetricReducer(MetricReducer):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.results: Dict[str, List[torch.Tensor]] = {key: [] for key in ("tp", "fp", "fn", "tn")}

    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> None:
        """Update the metric with the given predictions and ground truth. This method is called
        once for each batch.
        """
        # We will compute IoU metric by two ways: dataset-wise and image-wise
        # For now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class.
        # these values will be aggregated in the end of an epoch.
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt_mask.long(), mode="binary")

        # Save step outputs
        self.results["tp"].append(tp)
        self.results["fp"].append(fp)
        self.results["fn"].append(fn)
        self.results["tn"].append(tn)

    def per_slot_reduce(self) -> Dict[str, torch.Tensor]:
        """Reduce the metrics across each slot to some intermediate value. The return value of
        this method from each slot is the passed to cross_slot_reduce().

        NOTE: All the tensor are moved to CPU before returning, otherwise the tensors will be
        on different cuda devices during the distributed training and the cross_slot_reduce() will
        fail.
        """
        metrics: Dict[str, torch.Tensor] = {}
        metrics["tp"] = torch.cat(self.results["tp"]).cpu()
        metrics["fp"] = torch.cat(self.results["fp"]).cpu()
        metrics["fn"] = torch.cat(self.results["fn"]).cpu()
        metrics["tn"] = torch.cat(self.results["tn"]).cpu()

        return metrics

    def cross_slot_reduce(self, per_slot_metrics: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """Reduce the intermediate values from each slot to the final metric value. This method is
        called once per epoch after per_slot_reduce() has been called on each slot. The return value
        of this method is the final metric value for the epoch.
        """
        # per_slot_metrics is a list of dicts returned by the self.pre_slot_reduce() on each slot
        tp = torch.concat([metric["tp"] for metric in per_slot_metrics])
        fp = torch.concat([metric["fp"] for metric in per_slot_metrics])
        fn = torch.concat([metric["fn"] for metric in per_slot_metrics])
        tn = torch.concat([metric["tn"] for metric in per_slot_metrics])

        metrics_dict: Dict[str, np.ndarray] = {}

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        metrics_dict["per_image_iou"] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # Dataset IoU means that we aggregate intersection and union over whole dataset and then compute
        # IoU score. The difference between `dataset_iou` and `per_image_iou` scores can be large, if dataset
        # contains images with empty masks. Empty images influence a lot on `per_image_iou` and much less on
        # `dataset_iou`.
        metrics_dict["dataset_iou"] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Compute Dice score
        metrics_dict["dice_coef"] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        # Compute accuracy
        metrics_dict["accuracy"] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        return metrics_dict
