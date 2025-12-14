import torch
from typing import Optional
import timm
from models.base_interfaces import SupervisedModelBase

class ViTBase16Classifier(SupervisedModelBase):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: Optional[float] = None):
        super().__init__()
        drop_rate = dropout if dropout is not None else 0.1
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained,
                                       num_classes=num_classes, drop_rate=drop_rate)
    def forward_features(self, x):
        return self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
    def forward(self, x): return self.model(x)