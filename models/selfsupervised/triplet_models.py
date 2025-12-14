import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.base_interfaces import SelfSupervisedModelBase

class TripletResNet18(SelfSupervisedModelBase):
    def __init__(self, embedding_dim=256, margin=1.0):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.margin = margin
    def get_backbone(self) -> nn.Module: return self.backbone
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x); x = x.view(x.size(0), -1); return F.normalize(self.embedding(x), dim=1)
    def forward_triplet(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor):
        a_emb = self.forward(a); p_emb = self.forward(p); n_emb = self.forward(n)
        ap_dist = F.pairwise_distance(a_emb, p_emb); an_dist = F.pairwise_distance(a_emb, n_emb)
        loss = F.relu(ap_dist - an_dist + self.margin).mean()
        return loss, a_emb, p_emb, n_emb
