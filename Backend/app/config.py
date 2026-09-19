from pathlib import Path

CATEGORY_MAP: dict[str, str] = {
    "belly pain": "dolor de barriga",
    "burping": "eructos",
    "cold_hot": "frío/calor",
    "discomfort": "malestar",
    "hungry": "hambre",
    "laugh": "risa",
    "lonely": "soledad",
    "noise": "ruido",
    "scared": "miedo",
    "silence": "silencio",
    "tired": "cansancio",
}

CONFIDENCE_THRESHOLD: float = 0.70

SUPPORTED_FORMATS: set[str] = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp"}

SAMPLE_RATE: int = 22050

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_PATH: Path = PROJECT_ROOT.parent / "Dataset" / "archive" / "cry"

MODEL_PATH: Path = PROJECT_ROOT / "models" / "model.joblib"

LOGS_DIR: Path = PROJECT_ROOT / "logs"
