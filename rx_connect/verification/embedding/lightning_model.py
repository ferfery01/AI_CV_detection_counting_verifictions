from typing import List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from rx_connect.generator.continuous_learning_dataloader import (
    ContinuousLearningDataLoader,
)
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.embedding.base import ResNetEmbeddingModel

logger = setup_logger()


class EmbeddingLightningModel(L.LightningModule):
    """
    Lightning model for embedding model training.
    """

    def __init__(
        self,
        model: ResNetEmbeddingModel,
    ):
        super().__init__()

        self.model = model
        self.similar_fn = F.cosine_similarity
        self.relu = nn.ReLU()
        self.criterion = nn.BCELoss()
        self.epoch_margin: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FOrward call with normalization."""
        return F.normalize(self.model(x))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step includes:
        1. Forward pass for both references.
        2. Forward passes for querties.
        3. Positive and negative similarities.
        4. Loss function for maximizing the error margin (pos_sim - neg_sim).
        5. Log metrics.
        """
        (query_img, ref_img, ref0_count) = batch["queries"], batch["reference"], batch["ref_counts"][0]

        ref_vec = self(ref_img)
        query_vec = self(query_img)
        similarities = self.relu(self.similar_fn(query_vec.unsqueeze(1), ref_vec.unsqueeze(0), dim=2))
        pos_sim = torch.min(
            torch.cat([similarities[:ref0_count, 0], similarities[ref0_count:, 1]])
        ).unsqueeze(0)
        neg_sim = torch.max(
            torch.cat([similarities[:ref0_count, 1], similarities[ref0_count:, 0]])
        ).unsqueeze(0)

        assert hasattr(self.logger, "log_metrics")
        margin = pos_sim - neg_sim
        margin_loss = self.criterion(pos_sim, torch.Tensor([1]).to(self.device)) + self.criterion(
            neg_sim, torch.Tensor([0]).to(self.device)
        )

        self.log_dict(
            {
                "pos_sim": pos_sim.item(),
                "neg_sim": neg_sim.item(),
                "margin": margin.item(),
                "margin_loss": margin_loss.item(),
            }
        )
        self.epoch_margin.append(margin.item())
        return margin_loss

    def on_train_epoch_end(self) -> None:
        """
        Log epoch average margin.
        """
        epoch_avg_margin = np.mean(self.epoch_margin)
        self.epoch_margin.clear()
        self.log("epoch_avg_margin", float(epoch_avg_margin))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return ContinuousLearningDataLoader(mode="train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
