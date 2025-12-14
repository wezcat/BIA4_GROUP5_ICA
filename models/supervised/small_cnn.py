import torch
import torch.nn as nn
from typing import Optional
from models.base_interfaces import SupervisedModelBase

class SmallCNNClassifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, dropout: Optional[float] = None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        if dropout is not None:
            raise ValueError("SmallCNN does not support custom dropout value.")
    def forward_features(self, x): return self.features(x)
    def forward(self, x):
        x = self.forward_features(x); x = self.gap(x); x = x.view(x.size(0), -1); return self.fc(x)
