import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import LOGS_DIR


def log_classification(filename: str, category: str, confidence: float, warning: bool) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "consultas.json"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "category": category,
        "confidence": confidence,
        "warning": warning,
    }

    entries: list[dict] = []
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            with open(log_path, encoding="utf-8") as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            entries = []

    entries.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)