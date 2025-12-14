import torch
import torch.nn as nn
from typing import Optional

from torchvision import models
from models.base_interfaces import SupervisedModelBase

class EfficientNetB0Classifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, dropout: Optional[float] = None, pretrained: bool = True):
        super().__init__()
        # download pretrained EfficientNetB0 model
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Identity()

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.2)
        self.fc = nn.Linear(in_features, num_classes)
        self.avgpool = self.backbone.avgpool

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)   # [B, 1280, H', W']
        x = self.avgpool(x)            # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)        # [B, 1280]
        x = self.dropout(x)
        return self.fc(x)
