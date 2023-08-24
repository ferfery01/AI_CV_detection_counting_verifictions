from enum import Enum, auto
from typing import Any, Dict, Optional, Type, Union

import timm
import torch
import torch.nn as nn
from timm.models.resnet import ResNet

from rx_connect.verification.classification.pooling import GlobalAvgPool


class ResnetComponents(nn.Module):
    """Reconstruct a ResNet model into three layers:
    1. Feature Extractor
    2. Pooling Module
    3. Classifier Head
    """

    def __init__(self, basemodel: ResNet) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(basemodel.children())[:-2])
        self.pool = basemodel.global_pool
        self.classifier = basemodel.fc
        self.pool_dim = basemodel.fc.weight.size(1)


class BaseModel(nn.Module):
    """Load a backbone model and reconstruct it into three layers:
    1. Feature Extractor
    2. Pooling Module
    3. Classifier Head
    """

    def __init__(self, arch: str, pretrained: bool = True) -> None:
        """Initialize the model.

        Args:
            arch: Model architecture name (e.g. resnet18)
            pretrained: Whether to use pretrained weights or not
        """
        super().__init__()
        self.arch = arch
        basemodel = timm.create_model(arch, pretrained=pretrained)

        backbone = self._reconstruct_resnet(basemodel)
        self.features = backbone.features
        self.pool = backbone.pool
        self.classifier = backbone.classifier
        self.pool_dim = backbone.pool_dim

    @staticmethod
    def _reconstruct_resnet(basemodel: ResNet) -> ResnetComponents:
        return ResnetComponents(basemodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EnhancedBaseModel(BaseModel):
    """Subclass of BaseModel that introduces customizable pooling and classifier head.

    Args:
        arch: Model architecture
        pool_module: Pooling module to use
        pool_config: Configuration for the pooling module
        num_classes: The number of classes
        freeze_layers_up_to: Index of the final layer to freeze in network
        pretrained: Whether to use pretrained weights or not
    """

    def __init__(
        self,
        arch: str,
        pool_module: Type[GlobalAvgPool],
        pool_config: Dict[str, Any],
        num_classes: int,
        freeze_layers_up_to: Optional[int] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__(arch, pretrained)

        self._configure_representation(pool_module, pool_config, num_classes)
        if freeze_layers_up_to is not None:
            self._freeze_layers(freeze_layers_up_to)

    def _update_classifier_sequential(self, fc_input_dim: int, num_classes: int) -> None:
        """Update the classifier head of the model if it is a sequential module."""
        output_dim: int = 0
        for idx, module in enumerate(self.classifier.children()):
            if isinstance(module, nn.Linear):
                output_dim = module.weight.size(0)
                self.classifier[idx] = nn.Linear(fc_input_dim, output_dim)
                break
        self.classifier[-1] = nn.Linear(output_dim, num_classes)

    def _configure_representation(
        self, pool_module: Type[GlobalAvgPool], pool_config: Optional[Dict[str, Any]], num_classes: int
    ) -> None:
        """Configure the pooling layer of the model according to given configuration."""
        if pool_config is not None:
            # Update pooling layer
            pool_config["input_dim"] = self.pool_dim
            self.pool = pool_module(**pool_config)

            # Update classifier head
            fc_input_dim = self.pool.output_dim
            if isinstance(self.classifier, nn.Sequential):
                self._update_classifier_sequential(fc_input_dim, num_classes)
            else:
                self.classifier = nn.Linear(fc_input_dim, num_classes)
        else:
            self.classifier = nn.Linear(self.pool_dim, num_classes)

    def _freeze_layers(self, freeze_layers_up_to: int) -> None:
        """Freeze the layers up to the given index."""
        for i, module in enumerate(self.features.children()):
            if i < freeze_layers_up_to:
                for param in module.parameters():
                    param.requires_grad = False


class AGG_METHOD(Enum):
    GAvP = auto()  # Global Average Pooling


def configure_and_create_model(
    arch: str,
    pooling: Union[str, AGG_METHOD],
    num_classes: int,
    *,
    dim_reduction: int = 256,
    pretrained: bool = True,
    freeze_layers_up_to: Optional[int] = None,
) -> nn.Module:
    """Configure and create a model with the given pooling aggregation method."""
    if isinstance(pooling, str):
        pooling = AGG_METHOD[pooling]

    if pooling == AGG_METHOD.GAvP:
        pool_module = GlobalAvgPool
        pool_cfg = {"dim_reduction": dim_reduction}
    else:
        raise ValueError(
            f"Unknown representation aggregation method: {pooling} (choose from {list(AGG_METHOD)})"
        )

    return EnhancedBaseModel(
        arch,
        pool_module,
        pool_cfg,
        num_classes,
        pretrained=pretrained,
        freeze_layers_up_to=freeze_layers_up_to,
    )
