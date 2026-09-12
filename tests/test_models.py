import pytest
from pydantic import ValidationError

from app.models import ClassifyResponse, ErrorResponse


class TestClassifyResponse:
    def test_valid_response(self):
        r = ClassifyResponse(
            categoria="hambre",
            confianza=0.9,
            advertencia=False,
            mensaje="Clasificación exitosa",
        )
        assert r.categoria == "hambre"
        assert r.confianza == 0.9
        assert r.advertencia is False

    def test_valid_response_with_warning(self):
        r = ClassifyResponse(
            categoria="miedo",
            confianza=0.45,
            advertencia=True,
            mensaje="Confianza baja",
        )
        assert r.advertencia is True
        assert r.confianza == 0.45

    def test_rejects_non_string_categoria(self):
        with pytest.raises(ValidationError):
            ClassifyResponse(categoria=123, confianza=0.9, advertencia=False, mensaje="ok")

    def test_rejects_confianza_above_one(self):
        with pytest.raises(ValidationError):
            ClassifyResponse(categoria="hambre", confianza=1.5, advertencia=False, mensaje="ok")

    def test_rejects_confianza_below_zero(self):
        with pytest.raises(ValidationError):
            ClassifyResponse(categoria="hambre", confianza=-0.1, advertencia=False, mensaje="ok")

    def test_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            ClassifyResponse(categoria="hambre")


class TestErrorResponse:
    def test_valid_error(self):
        e = ErrorResponse(detail="Formato no soportado")
        assert e.detail == "Formato no soportado"

    def test_rejects_missing_detail(self):
        with pytest.raises(ValidationError):
            ErrorResponse()
