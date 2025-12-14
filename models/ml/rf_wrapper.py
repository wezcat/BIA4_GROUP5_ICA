import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from models.base_interfaces import MachineLearningModelBase

class RandomForestClassifierWrapper(MachineLearningModelBase):
    def __init__(self, n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
    def fit(self, X: np.ndarray, y: np.ndarray): self.model.fit(X, y); return self
    def predict(self, X: np.ndarray) -> np.ndarray: return self.model.predict(X)
    def predict_proba(self, X: np.ndarray) -> np.ndarray: return self.model.predict_proba(X)
    def save(self, path: str): joblib.dump(self.model, path)
    def load(self, path: str): self.model = joblib.load(path); return self
