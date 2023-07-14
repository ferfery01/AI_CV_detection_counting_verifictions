from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rx_connect.tools.logging import setup_logger
from rx_connect.verification.classification.base import configure_and_create_model

logger = setup_logger()


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        arch: str,
        pooling: str,
        dropout_rate: float = 0.5,
        emb_size: int = 2048,
        middle: int = 1000,
        pretrained: bool = True,
        skip_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.out_features = emb_size

        base_model = configure_and_create_model(arch, pooling, emb_size, pretrained=pretrained)
        dropout = nn.Dropout(dropout_rate)
        layers = [("feature_extractor", base_model), ("dropout", dropout)]

        # Create the embedding layer if necessary
        if not skip_embedding:
            layers.append(
                (
                    "emb",
                    nn.Sequential(
                        nn.Linear(emb_size, middle),
                        nn.BatchNorm1d(middle, affine=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(middle, emb_size),
                        nn.Tanh(),
                    ),
                )
            )

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding of the input tensor. This is an alias for `self.forward(x)`."""
        return self.forward(x)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize the input tensor."""
    return F.normalize(x, p=2, dim=1)


class BinaryHead(nn.Module):
    def __init__(self, num_class: int, emb_size: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.fc = nn.Linear(emb_size, num_class)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = l2_norm(features)
        logits = self.fc(features) * self.scale

        return logits


class MultiheadModel(nn.Module):
    def __init__(
        self, embedding_model: EmbeddingModel, n_classes: int, sep_side_train: bool = True, scale: int = 64
    ) -> None:
        super().__init__()

        if sep_side_train:
            n_classes *= 2
            logger.info(f"Treating front/back side of the pill as separate classes. n_classes={n_classes}")

        self.n_classes = n_classes
        self.embedding_model = embedding_model
        self.emb_size = embedding_model.out_features

        self.binary_head = BinaryHead(self.n_classes, self.emb_size, scale)
        self.sep_side_train = sep_side_train

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model. Returns a dictionary with the embedding and logits."""
        emb = self.embedding_model(x)
        logits = self.binary_head(emb)

        return {"emb": emb, "logits": logits}

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding of the input tensor. This is an alias for `self.forward(x)["emb"]`."""
        return self.forward(x)["emb"]

    def shift_label_indexes(self, logits: torch.Tensor) -> torch.Tensor:
        """Shift the label indexes of the logits to the original classes. This is necessary because
        the model was trained on a dataset where the front and back side of the pill were treated
        as separate classes. This function takes the max of the logits for the front and back side
        of the pill to get the logits for the original classes.
        """
        logger.assertion(self.sep_side_train, "Cannot shift label indexes if sep_side_train is False")

        actual_n_classes = self.n_classes // 2
        front = logits[:, :actual_n_classes]
        back = logits[:, actual_n_classes:]
        logger.assertion(front.shape == back.shape, f"{front.shape} != {back.shape}")

        logits = torch.stack([front, back], dim=0)
        logits, _ = logits.max(dim=0)

        return logits

    def get_original_logits(self, x: torch.Tensor, softmax: bool = False) -> torch.Tensor:
        """Get the logits for the original classes."""
        logits = self.forward(x)["logits"]
        if softmax:
            logits = F.softmax(logits, dim=1)

        # Shift back the logits to the original classes
        if self.sep_side_train:
            logits = self.shift_label_indexes(logits)

        return logits

    def get_original_n_classes(self) -> int:
        """Returns the number of classes that the model was trained on. This is different from the
        number of classes that the model predicts, which is `self.n_classes`.
        """
        return self.n_classes // 2 if self.sep_side_train else self.n_classes
