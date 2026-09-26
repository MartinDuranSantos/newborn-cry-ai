from pathlib import Path

import numpy as np

from app.config import CATEGORY_MAP, DATASET_PATH, SAMPLE_RATE, SUPPORTED_FORMATS
from app.services.audio_processor import AudioError, load_audio
from app.services.feature_extractor import extract_features


def load_dataset(dataset_path: Path = DATASET_PATH, clean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if clean:
        from app.services.audio_processor import filter_baby_cry, reduce_noise

    X: list[np.ndarray] = []
    y: list[str] = []

    for folder_name, label in CATEGORY_MAP.items():
        folder_path = dataset_path / folder_name
        if not folder_path.exists():
            continue

        for audio_file in sorted(folder_path.iterdir()):
            if audio_file.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            try:
                audio, sr = load_audio(audio_file)
                if clean:
                    audio = reduce_noise(audio, sr)
                    #audio = filter_baby_cry(audio, sr)
                features = extract_features(audio, sr)
                X.append(features)
                y.append(label)
            except (AudioError, Exception):
                continue

    return np.array(X), np.array(y)