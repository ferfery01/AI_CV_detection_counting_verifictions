from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rx_connect.core.types.verification.model import MultiHeadModelOutput
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.classification.base import configure_and_create_model
from rx_connect.verification.classification.margins import build_margin_head

logger = setup_logger()


class EmbeddingModel(nn.Module):
    """A PyTorch model class for embedding. This model includes an optional embedding layer that
    can be skipped.

    Attributes:
        arch (str): The name of the architecture.
        pooling (str): The name of the pooling method.
        dropout_rate (float): The dropout rate for regularization.
        emb_size (int): The size of the embeddings.
        middle (int): The size of the middle layer in the optional embedding layer.
        pretrained (bool): Whether or not to use a pretrained model.
        skip_embedding (bool): Whether or not to skip the embedding layer.
        out_features (int): The number of output features, equal to `emb_size`.
        model (nn.Module): The constructed PyTorch model.

    Methods:
        forward: Forward pass for the model.
        get_embedding: Get the embedding of an input tensor.
    """

    def __init__(
        self,
        arch: str,
        pooling: str,
        dropout_rate: float = 0.5,
        emb_size: int = 2048,
        middle: int = 1024,
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
                        nn.Linear(emb_size, middle, bias=False),
                        nn.BatchNorm1d(middle, affine=True),
                        nn.ReLU(),
                        nn.Linear(middle, emb_size, bias=False),
                        nn.Tanh(),
                    ),
                )
            )

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model."""
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding of the input tensor. This is an alias for `self.forward(x)`."""
        return self.forward(x)


class ScaledLogitClassifier(nn.Module):
    """The ScaledLogitClassifier is a PyTorch nn.Module designed to serve as the final
    classification layer of a model. It is engineered for diverse classification tasks.

    Inheriting functionalities from PyTorch's nn.Module, this class leverages a fully
    connected linear layer (nn.Linear) for the classification task. The input features are
    first normalized using L2 normalization and then scaled. The scaling factor, which can be
    tuned as a hyperparameter, assists in the convergence of the softmax loss function
    by controlling the magnitude of the logits prior to softmax operation, as described
    in the paper https://arxiv.org/abs/1704.06369.

    Attributes:
        n_classes (int): Specifies the number of classes in the classification task.
        emb_size (int): Denotes the dimensionality of the input feature vector.
        scale (int): A tunable scaling factor applied to output logits to control their magnitude.
        fc (nn.Linear): A fully connected layer responsible for classification.

    Methods:
        forward(features: torch.Tensor) -> torch.Tensor:
            Accepts a tensor representing features, normalizes it using L2 normalization, processes
            it through the fully connected layer, scales the resulting logits, and returns them.
    """

    def __init__(self, emb_size: int, n_classes: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.fc = nn.Linear(emb_size, n_classes, bias=False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        logits = self.fc(emb) * self.scale
        return logits


class MarginHead(nn.Module):
    """General margin head class that wraps the construction of specific margin heads like ArcFaceHead and
    CosFaceHead.
    """

    def __init__(
        self, head_type: str, emb_size: int, num_class: int, scale: int = 64, m: float = 0.5
    ) -> None:
        super().__init__()
        self.fc = build_margin_head(head_type, emb_size, num_class, scale, m)

    def forward(self, emb: torch.Tensor, label: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self.fc(emb, label) if label is not None else None


class MultiheadModel(nn.Module):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        n_classes: int,
        sep_side_train: bool,
        head_type: str = "arcface",
        scale1: int = 30,
        scale2: int = 64,
        m: float = 0.5,
    ) -> None:
        super().__init__()

        if sep_side_train:
            n_classes *= 2
            logger.info(f"Treating front/back side of the pill as separate classes. n_classes={n_classes}")

        self.n_classes = n_classes
        self.embedding_model = embedding_model
        self.emb_size = embedding_model.out_features

        self.cls_head = ScaledLogitClassifier(self.emb_size, self.n_classes, scale=scale1)
        self.margin_head = MarginHead(head_type, self.emb_size, self.n_classes, scale=scale2, m=m)
        self.sep_side_train = sep_side_train

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> MultiHeadModelOutput:
        """Forward pass of the model. Returns a dictionary with the embedding and logits."""
        emb = self.embedding_model(x)
        cls_logits = self.cls_head(emb)
        angular_logits = self.margin_head(emb, target)

        return MultiHeadModelOutput(emb=emb, cls_logits=cls_logits, angular_logits=angular_logits)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding of the input tensor. This is an alias for `self.forward(x)["emb"]`."""
        return self.forward(x)["emb"]

    def shift_label_indexes(self, logits: torch.Tensor) -> torch.Tensor:
        """Shift the label indexes of the logits to the original classes. This is necessary because
        the model was trained on a dataset where the front and back side of the pill were treated
        as separate classes. This function takes the max of the logits for the front and back side
        of the pill to get the logits for the original classes.
        """
        assert self.sep_side_train, "Cannot shift label indexes if `sep_side_train` is False"

        actual_n_classes = self.n_classes // 2
        front = logits[:, :actual_n_classes]
        back = logits[:, actual_n_classes:]
        assert front.shape == back.shape, f"{front.shape} != {back.shape}"

        # Take the max of the logits for the front and back side of the pill
        logits = torch.stack([front, back], dim=-1)
        logits, _ = logits.max(dim=-1)

        return logits

    def get_original_logits(self, x: torch.Tensor, softmax: bool = False) -> torch.Tensor:
        """Get the logits for the original classes."""
        logits = self.forward(x, target=None)["cls_logits"]
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
