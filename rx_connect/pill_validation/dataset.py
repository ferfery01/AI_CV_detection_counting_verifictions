from pathlib import Path
from typing import Optional

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from skimage import io
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from rx_connect.pill_validation.augments import RefConsTransform
from rx_connect.types.dataset import ePillIDDataset


class SingleImagePillID(Dataset):
    def __init__(
        self,
        root: Path,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        train: bool,
        transforms: A.Compose,
        rotate_aug: Optional[int] = None,
    ):
        self.root = root
        self.label_encoder = label_encoder
        self.train = train
        self.rotate_aug = rotate_aug
        self.transforms = transforms
        self.df = self.rotate_df(df, 360 // self.rotate_aug) if self.rotate_aug is not None else df

    def rotate_df(self, df: pd.DataFrame, n_rotations: int = 24) -> pd.DataFrame:
        """Generate a new DataFrame that represents various rotation states of the original data.

        This method should be used only for evaluation, not during training.
        The method adds a new column 'rot_degree' to the DataFrame which represents the rotation angle.

        Args:
            df: DataFrame with columns ['image_path', 'is_ref', 'is_front', 'pilltype_id']
            n_rotations: Number of rotations to apply to each image.

        Returns:
            A new DataFrame with additional column representing the rotation angle.

        Raises:
            AssertionError: if the method is called during training, or if `rotate_aug` is None.
        """
        assert not self.train, "`rotate_aug` should only be used for eval"
        assert self.rotate_aug is not None, "`rotate_aug` should be an integer"

        new_df = df.loc[df.index.repeat(n_rotations)].reset_index(drop=True)
        new_df["rot_degree"] = np.tile(np.arange(n_rotations) * self.rotate_aug, len(df))

        return new_df

    def load_img(self, df_row: pd.Series) -> torch.Tensor:
        """Load image and apply transforms"""
        img_path, is_ref = df_row.image_path, df_row.is_ref
        image: np.ndarray = io.imread(self.root / img_path)
        rot_degree: int = df_row.rot_degree if self.rotate_aug is not None else 0

        return self.transforms(image, is_ref=is_ref, rot_degree=rot_degree)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ePillIDDataset:
        df_row = self.df.iloc[idx]

        # Load image and apply transforms
        image: torch.Tensor = self.load_img(df_row)
        ndc_code: str = df_row.pilltype_id

        return {
            "image": image,
            "label": int(self.label_encoder.transform([ndc_code])[0]),
            "image_name": str(df_row.image_path),
            "is_ref": bool(df_row.is_ref),
            "is_front": bool(df_row.is_front),
        }


class PillIDDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Path,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        pin_memory_device: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.df = df
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.kwargs = kwargs

        # Initialize transforms
        self.train_transforms = RefConsTransform(train=True, normalize=True, **kwargs)
        self.val_transforms = RefConsTransform(train=False, normalize=True, **kwargs)

        self.init_dataframes()

    def init_dataframes(self) -> None:
        train_df = self.df[self.df.split == "train"]
        val_df = self.df[self.df.split == "val"]

        ref_only_df, cons_train_df = train_df[train_df.is_ref], train_df[~train_df.is_ref]
        cons_val_df = val_df[~val_df.is_ref]

        self.train_df = pd.concat([ref_only_df, cons_train_df], sort=False)
        self.val_df = pd.concat([ref_only_df, cons_val_df])

        labels_df = pd.DataFrame({"pilltype_id": self.label_encoder.classes_})
        self.eval_df = pd.merge(cons_val_df, labels_df, on=["pilltype_id"], how="inner")
        self.ref_df = pd.merge(ref_only_df, labels_df, on=["pilltype_id"], how="inner")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SingleImagePillID(
                self.root,
                self.train_df,
                self.label_encoder,
                train=True,
                transforms=self.train_transforms,
                **self.kwargs,
            )
            self.val_dataset = SingleImagePillID(
                self.root,
                self.val_df,
                self.label_encoder,
                train=False,
                transforms=self.val_transforms,
            )
            self.eval_dataset = SingleImagePillID(
                self.root,
                self.eval_df,
                self.label_encoder,
                train=False,
                transforms=self.val_transforms,
                rotate_aug=24,
            )
            self.ref_dataset = SingleImagePillID(
                self.root,
                self.ref_df,
                self.label_encoder,
                train=False,
                transforms=self.val_transforms,
                rotate_aug=24,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            pin_memory_device=self.pin_memory_device,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return a list of test dataloaders, one for reference images and one for
        consumer images."""
        test_dl = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            pin_memory_device=self.pin_memory_device,
        )
        ref_dl = DataLoader(
            self.ref_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            pin_memory_device=self.pin_memory_device,
        )
        return [test_dl, ref_dl]
