from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

class SupervisedModelBase(nn.Module, ABC):
    def __init__(self): super().__init__(); self.device = torch.device("cpu")
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor: pass # extract features
    def to(self, device: torch.device): self.device = device; return super().to(device) # remember the device

class SelfSupervisedModelBase(nn.Module, ABC):
    def __init__(self): super().__init__(); self.device = torch.device("cpu")
    @abstractmethod
    def forward_triplet(self, a, p, n): pass # triplet: anchor / positive / negative
    @abstractmethod
    def get_backbone(self) -> nn.Module: pass
    def to(self, device: torch.device): self.device = device; return super().to(device)

class MachineLearningModelBase(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray): pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def save(self, path: str): pass # other types of model already have function to save and load
    @abstractmethod
    def load(self, path: str): pass