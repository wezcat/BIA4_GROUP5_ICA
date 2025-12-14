import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from models.base_interfaces import SupervisedModelBase

class InceptionV3Classifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: Optional[float] = None):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.inception_v3(weights=weights, aux_logits=True,
                                          dropout=dropout if dropout is not None else 0.5)
        self.backbone.fc = nn.Linear(2048, num_classes)
        if self.backbone.AuxLogits:
            self.backbone.AuxLogits.fc = nn.Linear(self.backbone.AuxLogits.fc.in_features, num_classes)
    def forward_features(self, x): return self.backbone.Mixed_7c(x)
    def forward(self, x):
        out = self.backbone(x); return out.logits if hasattr(out, 'logits') else out[0]
