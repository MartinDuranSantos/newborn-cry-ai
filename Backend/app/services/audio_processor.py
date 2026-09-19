from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, istft, sosfilt, stft

from app.config import SUPPORTED_FORMATS, SAMPLE_RATE

BABY_CRY_LOW: int = 200
BABY_CRY_HIGH: int = 4000


class AudioError(Exception):
    pass


class UnsupportedFormatError(AudioError):
    pass


def validate_audio(file_path: Path) -> None:
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Formato no soportado: {file_path.suffix}. "
            f"Formatos válidos: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if not file_path.exists():
        raise AudioError(f"El archivo no existe: {file_path}")

    if file_path.stat().st_size == 0:
        raise AudioError("El archivo está vacío (0 bytes)")


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    validate_audio(file_path)

    try:
        if file_path.suffix.lower() == ".3gp":
            return _load_3gp(file_path)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioError(f"No se pudo procesar el audio: {exc}") from exc

    if len(audio) == 0:
        raise AudioError("El audio no contiene datos de señal válidos")

    return audio, sr


def _load_3gp(file_path: Path) -> tuple[np.ndarray, int]:
    segment = AudioSegment.from_file(file_path, format="3gp")
    segment = segment.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    return audio, SAMPLE_RATE


def reduce_noise(audio: np.ndarray, sr: int, frame_length: int = 2048,
                 hop_length: int = 512, noise_percentile: int = 50,
                 gate_factor: float = 1.5) -> np.ndarray:
    if len(audio) < frame_length:
        return _normalize(audio)

    _, _, zxx = stft(audio, fs=sr, nperseg=frame_length, noverlap=frame_length - hop_length)
    mag = np.abs(zxx)

    global_thresh = np.percentile(mag, noise_percentile)
    gate = mag >= (np.maximum(global_thresh, 1e-10) * gate_factor)
    zxx_clean = zxx * gate.astype(float)

    _, audio_clean = istft(zxx_clean, fs=sr, nperseg=frame_length, noverlap=frame_length - hop_length)

    return _normalize(audio_clean.astype(np.float32))[: len(audio)]


def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def filter_baby_cry(audio: np.ndarray, sr: int, order: int = 4) -> np.ndarray:
    nyq = sr / 2
    low = max(BABY_CRY_LOW / nyq, 1e-6)
    high = min(BABY_CRY_HIGH / nyq, 1.0 - 1e-6)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfilt(sos, audio)

    return _remove_silence(filtered, sr)


def _remove_silence(audio: np.ndarray, sr: int, frame_ms: int = 30,
                    energy_percentile: float = 15.0, smoothing_frames: int = 5) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    if len(audio) < frame_len * 2:
        return audio

    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames**2, axis=1)
    threshold = np.percentile(energies[energies > 0], energy_percentile) if np.any(energies > 0) else 0.0
    threshold = max(threshold, 1e-8)

    active = energies > threshold
    kernel = np.ones(smoothing_frames * 2 + 1) / (smoothing_frames * 2 + 1)
    active_smooth = np.convolve(active.astype(float), kernel, mode="same") > 0.3

    active_samples = np.repeat(active_smooth, frame_len)
    if len(active_samples) < len(audio):
        active_samples = np.pad(active_samples, (0, len(audio) - len(active_samples)), mode="edge")

    return audio[active_samples]


def validate_signal_quality(audio: np.ndarray, rms_min: float = 1e-4,
                            active_min_ratio: float = 0.05, frame_len: int = 512) -> bool:
    if len(audio) == 0:
        return False

    if np.sqrt(np.mean(audio**2)) < rms_min:
        return False

    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return bool(np.sqrt(np.mean(audio**2)) >= rms_min)

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    active_ratio = np.mean(frame_rms > rms_min)

    return bool(active_ratio >= active_min_ratio)
