import librosa
import librosa.feature.rhythm
import numpy as np

N_MFCC: int = 40
FEATURE_SIZE: int = 123


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = _ensure_min_duration(audio, sr)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = float(np.mean(librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)))

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            chroma_std,
            contrast_mean,
            contrast_std,
            np.array([zcr_mean, zcr_std, rms_mean, rms_std, tempo], dtype=np.float32),
        ]
    )
    return features.astype(np.float32)


def _ensure_min_duration(audio: np.ndarray, sr: int, min_seconds: float = 0.3) -> np.ndarray:
    min_len = int(sr * min_seconds)
    if len(audio) >= min_len:
        return audio
    return np.pad(audio, (0, min_len - len(audio)), mode="constant", constant_values=0)