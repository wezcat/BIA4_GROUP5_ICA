import torch.nn as nn
from torchvision import models
import numpy as np
import joblib
import torch
from catboost import CatBoostClassifier
from config import cfg
from models.base_interfaces import MachineLearningModelBase

class DenseNet201FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        backbone = models.densenet201(weights=weights)
        backbone.classifier = nn.Identity()
        self.backbone = backbone.eval()
    def forward(self, x): return self.backbone(x)

class DenseNet201CatBoost(MachineLearningModelBase):
    def __init__(self):
        self.feature_extractor_type = "densenet201"
        self.feature_extractor = DenseNet201FeatureExtractor().to(cfg.device)
        self.feature_extractor.eval()
        self.model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=False)
    def fit(self, X: np.ndarray, y: np.ndarray):
        with torch.no_grad():
            tensor = torch.from_numpy(X.reshape(-1, 3, cfg.input_size, cfg.input_size)).float().to(cfg.device)
            feats = self.feature_extractor(tensor).cpu().numpy()
        self.model.fit(feats, y)
    def predict(self, X: np.ndarray):
        with torch.no_grad():
            tensor = torch.from_numpy(X.reshape(-1, 3, cfg.input_size, cfg.input_size)).float().to(cfg.device)
            feats = self.feature_extractor(tensor).cpu().numpy()
        return self.model.predict(feats)
    def predict_proba(self, X: np.ndarray):
        with torch.no_grad():
            tensor = torch.from_numpy(X.reshape(-1, 3, cfg.input_size, cfg.input_size)).float().to(cfg.device)
            feats = self.feature_extractor(tensor).cpu().numpy()
        return self.model.predict_proba(feats)
    def save(self, path: str):
        joblib.dump(self.model, path)
        torch.save(self.feature_extractor.state_dict(), path.replace(".pkl", "_feat.pth"))
    def load(self, path: str):
        self.model = joblib.load(path)
        self.feature_extractor.load_state_dict(
            torch.load(path.replace(".pkl", "_feat.pth"), map_location=cfg.device))
