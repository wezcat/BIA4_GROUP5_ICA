import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from models.base_interfaces import SupervisedModelBase

class ResNet34Classifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: Optional[float] = None):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        if dropout is not None:
            raise ValueError("ResNet34 does not support custom dropout value.")
    def forward_features(self, x):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x); x = self.backbone.layer1(x); x = self.backbone.layer2(x)
        x = self.backbone.layer3(x); x = self.backbone.layer4(x); return x
    def forward(self, x):
        x = self.forward_features(x); x = self.backbone.avgpool(x); x = torch.flatten(x, 1)
        return self.backbone.fc(x)