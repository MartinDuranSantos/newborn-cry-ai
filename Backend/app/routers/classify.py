from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import CONFIDENCE_THRESHOLD
from app.models import ClassifyResponse
from app.services.audio_processor import (
    AudioError,
    UnsupportedFormatError,
    filter_baby_cry,
    load_audio,
    reduce_noise,
)
from app.services.classifier import predict
from app.services.feature_extractor import extract_features
from app.services.logger import log_classification

router = APIRouter(prefix="/clasificar", tags=["clasificar"])


@router.post("/", response_model=ClassifyResponse)
async def classify_audio(file: UploadFile = File(...)) -> ClassifyResponse:
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            audio, sr = load_audio(tmp_path)
            audio = reduce_noise(audio, sr)
            audio = filter_baby_cry(audio, sr)
            features = extract_features(audio, sr)
            category, confidence = predict(features)

            warning = confidence < CONFIDENCE_THRESHOLD
            mensaje = (
                "Clasificación exitosa"
                if not warning
                else "El modelo no está seguro; la clasificación puede ser incorrecta"
            )

            result = ClassifyResponse(
                categoria=category,
                confianza=confidence,
                advertencia=warning,
                mensaje=mensaje,
            )
            log_classification(filename, result.categoria, result.confianza, result.advertencia)
            return result
        finally:
            tmp_path.unlink(missing_ok=True)

    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {exc}") from exc