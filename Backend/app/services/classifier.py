import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.config import FULL_MODEL_PATH


_model: RandomForestClassifier | None = None


def load_model(path: str = str(FULL_MODEL_PATH)) -> None:
    global _model
    _model = joblib.load(path)


def predict(features: np.ndarray) -> tuple[str, float]:
    if _model is None:
        raise RuntimeError("El modelo no está cargado")
    features = np.asarray(features).reshape(1, -1)
    probs = _model.predict_proba(features)[0]
    best_idx = int(np.argmax(probs))
    classes = getattr(_model, "label_classes", None)
    if classes is None:
        classes = _model.classes_
    category = str(classes[best_idx])
    confidence = float(probs[best_idx])
    return category, confidence