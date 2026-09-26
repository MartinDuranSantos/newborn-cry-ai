import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from typing import Any
from xgboost import XGBClassifier

from app.config import FULL_MODEL_PATH


def _class_balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    weights = {c: 1.0 / (len(classes) * p) for c, p in zip(classes, probs)}
    return np.array([weights[c] for c in y])


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict, dict]:
    
    encoder = LabelEncoder().fit(y)
    y_enc = encoder.transform(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    for train_idx, val_idx in skf.split(X, y_enc):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

       # ML model
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=_class_balanced_weights(y_train))
        y_pred = clf.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average="macro", zero_division=0))
        recalls.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

    final_clf = XGBClassifier(**params)
    final_clf.fit(X, y_enc, sample_weight=_class_balanced_weights(y))
    final_clf.label_classes = encoder.classes_

    metrics = {
        "accuracy_mean": float(np.mean(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "f1_mean": float(np.mean(f1s)),
    }

    return final_clf, metrics, params


def save_model(model: XGBClassifier, path: str = str(FULL_MODEL_PATH)) -> None:
    FULL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)