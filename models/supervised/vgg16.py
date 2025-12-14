import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from models.base_interfaces import SupervisedModelBase

class VGG16Classifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: Optional[float] = None):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vgg16(weights=weights)
        for p in self.backbone.features.parameters(): p.requires_grad = False
        dropout_rate = dropout if dropout is not None else 0.5
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
    def forward_features(self, x): return self.backbone.features(x)
    def forward(self, x): return self.backbone(x)
