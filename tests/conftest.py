import subprocess

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 22050
DURATION = 3.0


@pytest.fixture
def sine_signal() -> np.ndarray:
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def noisy_signal() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    rng = np.random.default_rng(42)
    noise = (0.2 * rng.standard_normal(len(t))).astype(np.float32)
    return clean, clean + noise


@pytest.fixture
def sine_wav(tmp_path, sine_signal):
    path = tmp_path / "test.wav"
    sf.write(path, sine_signal, SAMPLE_RATE)
    return path


@pytest.fixture
def sine_wav_in_other_formats(tmp_path, sine_signal):
    base = tmp_path / "test"
    sf.write(base.with_suffix(".wav"), sine_signal, SAMPLE_RATE)
    src = base.with_suffix(".wav")
    paths = {}
    for suffix in [".flac", ".mp3", ".ogg", ".m4a"]:
        out = base.with_suffix(suffix)
        subprocess.run(["ffmpeg", "-y", "-i", str(src), str(out)], capture_output=True, check=False)
        paths[suffix] = out
    out_3gp = base.with_suffix(".3gp")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "8000", str(out_3gp)],
        capture_output=True,
        check=False,
    )
    paths[".3gp"] = out_3gp
    return paths