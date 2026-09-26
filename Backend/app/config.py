from pathlib import Path

# CATEGORY_MAP: dict[str, str] = {
#     "belly pain": "dolor de barriga",
#     "burping": "eructos",
#     "cold_hot": "frío/calor",
#     "discomfort": "malestar",
#     "hungry": "hambre",
#     "laugh": "risa",
#     "lonely": "soledad",
#     "noise": "ruido",
#     "scared": "miedo",
#     "silence": "silencio",
#     "tired": "cansancio",
# }

CATEGORY_MAP: dict[str, str] = {
    "belly pain": "dolor de barriga",
    "hungry": "hambre",
    "scared": "miedo",
}

CONFIDENCE_THRESHOLD: float = 0.70

SUPPORTED_FORMATS: set[str] = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp"}

SAMPLE_RATE: int = 22050

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATASET_PATH: Path = PROJECT_ROOT.parent / "Dataset" / "archive" / "cry"

model_name: str = "model.joblib"

FULL_MODEL_PATH: Path = PROJECT_ROOT / "models" / "model.joblib"

MODEL_PATH: Path = FULL_MODEL_PATH

LOGS_DIR: Path = PROJECT_ROOT / "logs"
