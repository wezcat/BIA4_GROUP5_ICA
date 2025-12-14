import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from models.base_interfaces import SupervisedModelBase

class ResNet50(SupervisedModelBase):
    def __init__(self, num_classes=2, dropout: Optional[float] = None):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.3)
        self.fc = nn.Linear(2048, num_classes)
    def forward_features(self, x):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x); x = self.backbone.layer1(x); x = self.backbone.layer2(x)
        x = self.backbone.layer3(x); x = self.backbone.layer4(x); return x
    def forward(self, x):
        x = self.forward_features(x); x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.dropout(x); return self.fc(x)