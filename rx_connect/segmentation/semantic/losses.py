import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        from_logits: bool = False,
        eps: float = 1e-5,
    ) -> None:
        """Initialize Focal Tversky Loss.

        Args:
            alpha: Controls the magnitude of penalties for false positives.
            beta: Controls the magnitude of penalties for false negatives.
            gamma: Focusing parameter for the Tversky index.
            from_logits: Whether to apply sigmoid activation to the predictions.
            eps: Small constant to prevent division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Tversky Loss.

        Args:
            pred: Predicted output (B, C, H, W)
            target: Ground truth (B, H, W)

        Returns:
            loss: Focal Tversky loss
        """
        # Apply sigmoid activation to constrain predictions between 0 and 1
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Calculate True Positives, False Positives, and False Negatives
        tp = torch.sum(pred * target) + self.eps
        fp = torch.sum(pred * (1 - target)) + self.eps
        fn = torch.sum((1 - pred) * target) + self.eps

        # Calculate Tversky index
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn)

        # Calculate Focal Tversky loss
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)

        return focal_tversky_loss
