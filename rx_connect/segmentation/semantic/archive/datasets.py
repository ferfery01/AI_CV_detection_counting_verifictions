from pathlib import Path
from typing import Any, Optional, Tuple, Union

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from rx_connect.segmentation.semantic.augments import SegmentTransform
from rx_connect.segmentation.semantic.datasets import DatasetSplit, SegDataset


class SegDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Union[str, Path],
        image_size: Tuple[int, int],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

        # Initialize transforms
        aspect_ratio = image_size[1] / image_size[0]
        self.train_transforms = SegmentTransform(
            train=True, normalize=True, image_size=self.image_size, aspect_ratio=aspect_ratio, **self.kwargs
        )
        self.val_transforms = SegmentTransform(train=False, normalize=True, image_size=self.image_size)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SegDataset(
                root_dir=self.root_dir,
                data_split=DatasetSplit.TRAIN,
                image_size=self.image_size,
                transforms=self.train_transforms,
            )
            self.val_dataset = SegDataset(
                root_dir=self.root_dir,
                data_split=DatasetSplit.VAL,
                image_size=self.image_size,
                transforms=self.val_transforms,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SegDataset(
                root_dir=self.root_dir,
                data_split=DatasetSplit.TEST,
                image_size=self.image_size,
                transforms=self.val_transforms,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
