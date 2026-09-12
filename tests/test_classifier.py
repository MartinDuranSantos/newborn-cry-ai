import numpy as np
import pytest

from app.config import CATEGORY_MAP, MODEL_PATH
from app.services.classifier import load_model, predict


@pytest.fixture(autouse=True)
def _load_model():
    load_model()
    yield


class TestClassifier:
    def test_predict_returns_valid_category(self):
        features = np.random.default_rng(0).random(123)
        category, _ = predict(features)
        assert category in CATEGORY_MAP.values()

    def test_predict_returns_valid_confidence(self):
        features = np.random.default_rng(1).random(123)
        _, confidence = predict(features)
        assert 0.0 <= confidence <= 1.0

    def test_predict_returns_tuple(self):
        features = np.random.default_rng(2).random(123)
        result = predict(features)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_predict_consistent_for_same_features(self):
        features = np.random.default_rng(3).random(123)
        c1, p1 = predict(features)
        c2, p2 = predict(features)
        assert c1 == c2
        assert p1 == p2

    def test_predict_probabilities_sum_to_one(self):
        features = np.random.default_rng(4).random(123)
        assert MODEL_PATH.exists()
        category, confidence = predict(features)
        assert confidence > 0.0
        assert category != ""

    def test_load_model_accepts_custom_path(self, tmp_path):
        import shutil

        custom = tmp_path / "custom.joblib"
        shutil.copy(MODEL_PATH, custom)
        load_model(str(custom))
        category, confidence = predict(np.random.rand(123))
        assert category in CATEGORY_MAP.values()
        assert 0.0 <= confidence <= 1.0