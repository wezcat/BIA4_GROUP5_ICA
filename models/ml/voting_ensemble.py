import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from models.base_interfaces import MachineLearningModelBase

class VotingEnsembleML(MachineLearningModelBase):
    def __init__(self):
        rf = RandomForestClassifier(n_estimators=1000, max_depth=12, random_state=42)
        lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=7)
        cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=False)
        self.model = VotingClassifier(estimators=[("rf", rf), ("lgb", lgb), ("cat", cat)], voting="soft")
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def save(self, path): joblib.dump(self.model, path)
    def load(self, path): self.model = joblib.load(path)