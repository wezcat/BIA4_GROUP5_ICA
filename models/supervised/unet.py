import torch
import torch.nn as nn
from typing import Optional
from models.base_interfaces import SupervisedModelBase
from models.conv_blocks import DoubleConv, Down

class UNet(SupervisedModelBase):
    def __init__(self, n_channels: int = 3, n_classes: int = 2, dropout: Optional[float] = None):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256)
        self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.3)
        self.fc = nn.Linear(1024, n_classes)
    def forward_features(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4); return x5
    def forward(self, x):
        feat = self.forward_features(x); x = self.gap(feat)
        x = x.view(x.size(0), -1); x = self.dropout(x); return self.fc(x)