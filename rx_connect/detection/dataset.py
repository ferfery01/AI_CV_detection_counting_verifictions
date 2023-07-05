from typing import Dict, List, Tuple

from super_gradients.common.object_names import Dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)


def create_dataloaders(
    data_dir: str,
    classes: List[str],
    input_dim: List[int],
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[Dataloaders, Dataloaders]:
    """Loads the dataset and returns the train, val, and test dataloaders."""
    dataloader_params: Dict[str, int] = {"batch_size": batch_size, "num_workers": num_workers}
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": data_dir,
            "images_dir": "train/images",
            "labels_dir": "train/labels",
            "classes": classes,
            "input_dim": input_dim,
        },
        dataloader_params=dataloader_params,
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": data_dir,
            "images_dir": "val/images",
            "labels_dir": "val/labels",
            "classes": classes,
            "input_dim": input_dim,
        },
        dataloader_params=dataloader_params,
    )

    return train_data, val_data
