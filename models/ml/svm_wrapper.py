import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from models.base_interfaces import MachineLearningModelBase

class SVMClassifierWrapper(MachineLearningModelBase):
    def __init__(self, C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                class_weight=class_weight,
                probability=True,
                random_state=random_state
            ))
        ])
    def fit(self, X: np.ndarray, y: np.ndarray): self.model.fit(X, y); return self
    def predict(self, X: np.ndarray) -> np.ndarray: return self.model.predict(X)
    def predict_proba(self, X: np.ndarray) -> np.ndarray: return self.model.predict_proba(X)
    def save(self, path: str): joblib.dump(self.model, path)
    def load(self, path: str): self.model = joblib.load(path); return self