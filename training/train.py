import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from app.config import MODEL_PATH


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_estimators=200,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average="macro", zero_division=0))
        recalls.append(recall_score(y_val, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

    final_clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_estimators=200,
    )
    final_clf.fit(X, y)

    metrics = {
        "accuracy_mean": float(np.mean(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "f1_mean": float(np.mean(f1s)),
    }

    return final_clf, metrics


def save_model(model: RandomForestClassifier, path: str = str(MODEL_PATH)) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)