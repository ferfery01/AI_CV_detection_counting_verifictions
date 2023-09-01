import timm
import torch
import torch.nn as nn
from timm.models.resnet import ResNet


class ResnetComponents(nn.Module):
    """Reconstruct a ResNet model into 2 layers:
    1. Feature Extractor
    2. Pooling Module
    """

    def __init__(self, basemodel: ResNet) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(basemodel.children())[:-2])  # w, h, dim
        self.pool = nn.AdaptiveMaxPool2d((1, 1))  # 1, 1, dim


class ResNetEmbeddingModel(nn.Module):
    """Load a backbone model and reconstruct it into 2 layers:
    1. Feature Extractor
    2. Pooling Module
    """

    def __init__(self, arch: str, pretrained: bool = True) -> None:
        """Initialize the model.

        Args:
            arch: Model architecture name (e.g. resnet18)
                Options: ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
            pretrained: Whether to use pretrained weights or not
        """
        super().__init__()
        self.arch = arch
        basemodel = timm.create_model(arch, pretrained=pretrained)

        backbone = self._reconstruct_resnet(basemodel)
        self.features = backbone.features
        self.pool = backbone.pool

    @staticmethod
    def _reconstruct_resnet(basemodel: ResNet) -> ResnetComponents:
        return ResnetComponents(basemodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
