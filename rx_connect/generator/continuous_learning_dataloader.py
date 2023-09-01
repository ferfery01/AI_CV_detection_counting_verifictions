from typing import List, cast

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from rx_connect.pipelines.generator import RxImageGenerator

UNIFIED_TRANSFORM = A.Compose(
    [
        # Resize the longest side to 224, maintaining the aspect ratio
        A.LongestMaxSize(224, always_apply=True),
        # Pad the image on the sides to make it square
        A.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),
        # Normalize the image
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Convert to PyTorch tensor
        ToTensorV2(),
    ]
)


class ContinuousLearningDataset(Dataset):
    """
    Dataset for continuous learning.

    Args:
    - num_samples (int):
        Number of samples in the dataset. This is only virtually used to comply with the dataset definition.
    - new_sample_rate (float): Rate at which to generate new samples.
        '0.0' means it will only generate new samples when the first time it is called.
        '1.0' means it will always generate new samples.
    """

    def __init__(self, num_samples: int = 1000, new_sample_rate: float = 0.0):
        self.generator = RxImageGenerator(num_pills_type=2, num_pills=(20, 20), apply_noise=False)
        self.new_sample_rate = new_sample_rate
        self.samples: List[dict | None] = [None] * num_samples
        self.last_idx = -1

        # Define the pre-processing transforms
        self._transforms = UNIFIED_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate the new sample if needed, and returns the batch created from the sample at 'idx'.
        """
        if self.samples[idx] is None or np.random.rand() < self.new_sample_rate:
            image, _, _, gt_bbox, gt_counts = self.generator()

            query_img_list = [image[xmin:xmax, ymin:ymax] for (xmin, xmax, ymin, ymax) in gt_bbox]
            ref_img_list = self.generator.reference_pills

            query_tensor = self._pad_into_tensor(query_img_list)
            ref_tensor = self._pad_into_tensor(ref_img_list)
            self.samples[idx] = {"queries": query_tensor, "reference": ref_tensor, "ref_counts": gt_counts}
        return cast(dict, self.samples[idx])

    def _pad_into_tensor(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Pad and transform the images (np.ndarray) into a single torch Tensor.
        """
        list_padded_image_tensor = [self._transforms(image=image)["image"] for image in images]
        stack_padded_image_tensor = torch.stack(list_padded_image_tensor).float()
        return stack_padded_image_tensor


class ContinuousLearningDataLoader(DataLoader):
    def __init__(self, mode: str = "train"):
        num_samples, new_sample_rate = (100, 1.0) if mode == "train" else (1000, 0.0)
        super().__init__(
            dataset=ContinuousLearningDataset(num_samples=num_samples, new_sample_rate=new_sample_rate),
        )

    def __iter__(self):
        return iter(self.dataset)
