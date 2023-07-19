import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginHead(ABC, nn.Module):
    """Abstract base class for the margin-based loss head.

    Attributes:
        emb_size (int): Size of the embedding vector.
        n_classes (int): Number of classes.
        scale (int): Scaling factor to increase the margin.
        m (float): Margin value.
        eps (float): Small value for numerical stability.
        kernel (nn.Linear): Linear layer used for transformations.

    Methods:
        forward: Abstract method to be implemented by the subclasses.
    """

    def __init__(self, emb_size: int, n_classes: int, scale: int, m: float, eps: float = 1e-5) -> None:
        super().__init__()

        # Init the weights of the fully connected layer
        self.kernel = nn.Linear(emb_size, n_classes, bias=False)
        nn.init.xavier_normal_(self.kernel.weight)

        self.n_classes = n_classes
        self.scale = scale
        self.m = m
        self.eps = eps

    def compute_cosine(self, emb: torch.Tensor) -> torch.Tensor:
        """Computes the cosine similarity between the input embeddings and the weights of the fully
        connected layer.

        Args:
            emb (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: The cosine similarity between the input embeddings and the weights of the fully
                connected layer.
        """
        # Normalize the weights of the last fully connected layer
        with torch.no_grad():
            self.kernel.weight = F.normalize(self.kernel.weight, dim=0)

        # Normalize the input embeddings
        emb = F.normalize(emb, dim=1)

        # Compute the cosine similarity b/w the normalized embeddings and the normalized weights
        return self.kernel(emb)

    @abstractmethod
    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Abstract forward method that should be implemented by the subclasses.

        Args:
            emb (torch.Tensor): Input embeddings.
            label (torch.Tensor): Ground truth labels.

        Raises:
            NotImplementedError: If this method is not overridden in the subclasses.
        """
        raise NotImplementedError


class CosFaceHead(MarginHead):
    """Cosine Margin Softmax loss layer, proposed in "CosFace: Large Margin Cosine Loss for Deep Face
    Recognition" (https://arxiv.org/abs/1801.09414). It is similar to the ArcFace loss function, but
    it uses a cosine similarity instead of an angular similarity.
    """

    def __init__(
        self, emb_size: int, n_classes: int, scale: int = 64, m: float = 0.4, eps: float = 1e-5
    ) -> None:
        super().__init__(emb_size, n_classes, scale, m, eps)

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Compute the cosine similarity b/w the normalized embeddings and the normalized weights
        cos_theta = self.compute_cosine(emb)

        # Create one-hot vectors from the labels
        d_theta = self.m * F.one_hot(label, num_classes=self.n_classes)

        # Calculate cosθ - m
        cos_theta_m = self.scale * (cos_theta - d_theta)

        # Scale the output in order for the softmax loss to converge, as described in the paper
        # NormFace (https://arxiv.org/abs/1704.06369)
        return self.scale * cos_theta_m


class ArcFaceHead(MarginHead):
    """ArcFace Margin Softmax loss layer, proposed in "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition" (https://arxiv.org/abs/1801.07698). It enhances the discriminative power of the
    deeply learned features.
    """

    def __init__(
        self, emb_size: int, n_classes: int, scale: int = 64, m: float = 0.5, eps: float = 1e-5
    ) -> None:
        super().__init__(emb_size, n_classes, scale, m, eps)

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb (torch.Tensor): The input embeddings for which the additive margin softmax will be computed.
            label (torch.Tensor): The true labels of the input embeddings.

        Returns:
            torch.Tensor: The computed additive margin softmax for the input embeddings.
        """
        # Compute the cosine similarity b/w the normalized embeddings and the normalized weights
        cos_theta = self.compute_cosine(emb).clamp(-1 + self.eps, 1 - self.eps)  # for numerical stability

        with torch.no_grad():
            # Compute the theta value
            theta = cos_theta.acos()

            # Create one-hot vectors from the labels
            one_hot = self.m * F.one_hot(label, num_classes=self.n_classes)

            # Calculate cos(theta + m)
            theta_m = torch.clip(theta + one_hot, min=self.eps, max=math.pi - self.eps)
            cosine_m = theta_m.cos()

        # Scale the output in order for the softmax loss to converge, as described in the paper
        # NormFace (https://arxiv.org/abs/1704.06369)
        return self.scale * cosine_m


def build_margin_head(
    head_type: str, emb_size: int, n_classes: int, scale: int = 64, m: float = 0.5, eps: float = 1e-5
) -> MarginHead:
    """Builds a margin-based loss head. Currently supported types are 'arcface' and 'cosface'.

    Args:
        head_type (str): Type of margin head to construct.
        emb_size (int): Size of the input embedding vector.
        n_classes (int): Number of classes.
        scale (int, optional): Scaling factor to increase the margin. Default is 64.
        m (float, optional): Margin value. Default is 0.5.
        eps (float, optional): Small value for numerical stability. Default is 1e-4.

    Returns:
        MarginHead: A MarginHead object, which can be either an ArcFaceHead or CosFaceHead.

    Raises:
        ValueError: If an invalid margin head type is provided.
    """
    if head_type == "arcface":
        return ArcFaceHead(emb_size, n_classes, scale, m, eps)
    elif head_type == "cosface":
        return CosFaceHead(emb_size, n_classes, scale, m, eps)
    else:
        raise ValueError(f"Invalid margin head type: {head_type}. Valid types are 'arcface' and 'cosface'.")
