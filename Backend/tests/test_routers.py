import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import CATEGORY_MAP
from app.main import app

SAMPLE_WAV = list(
    Path("/home/mds/PhD/MY_PROJECTS/infant-cri-ai/Dataset/archive/cry/belly pain").glob("*.wav")
)[0]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def model_loaded():
    from app.services.classifier import load_model

    load_model()
    yield


class TestClassifyEndpoint:
    def test_classify_valid_wav_returns_200(self, client):
        with open(SAMPLE_WAV, "rb") as f:
            response = client.post("/clasificar", files={"file": (SAMPLE_WAV.name, f, "audio/wav")})
        assert response.status_code == 200

    def test_classify_response_schema(self, client):
        with open(SAMPLE_WAV, "rb") as f:
            response = client.post("/clasificar", files={"file": (SAMPLE_WAV.name, f, "audio/wav")})
        body = response.json()
        assert set(["categoria", "confianza", "advertencia", "mensaje"]) == set(body.keys())
        assert body["categoria"] in CATEGORY_MAP.values()
        assert 0.0 <= body["confianza"] <= 1.0
        assert isinstance(body["advertencia"], bool)
        assert isinstance(body["mensaje"], str)

    def test_invalid_format_returns_400(self, client):
        response = client.post(
            "/clasificar", files={"file": ("video.avi", b"data", "video/x-msvideo")}
        )
        assert response.status_code == 400
        assert "no soportado" in response.json()["detail"]

    def test_empty_file_returns_422(self, client):
        response = client.post("/clasificar", files={"file": ("empty.wav", b"", "audio/wav")})
        assert response.status_code == 422
        assert "vac" in response.json()["detail"]

    def test_corrupt_file_returns_422(self, client):
        response = client.post(
            "/clasificar", files={"file": ("fake.wav", b"RIFF not audio", "audio/wav")}
        )
        assert response.status_code == 422

    def test_low_confidence_returns_warning(self, client):
        with open(SAMPLE_WAV, "rb") as f:
            response = client.post("/clasificar", files={"file": (SAMPLE_WAV.name, f, "audio/wav")})
        body = response.json()
        if body["confianza"] < 0.70:
            assert body["advertencia"] is True
            assert "no está seguro" in body["mensaje"]
        else:
            assert body["advertencia"] is False


class TestRoot:
    def test_root_returns_message(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "mensaje" in response.json()