import numpy as np
import pytest

from app.config import SUPPORTED_FORMATS
from app.services.audio_processor import (
    AudioError,
    filter_baby_cry,
    load_audio,
    reduce_noise,
    validate_audio,
    validate_signal_quality,
)


def _snr(signal: np.ndarray, noise: np.ndarray) -> float:
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))


def _energy_in_band(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


class TestValidateAudio:
    def test_accepts_wav(self, sine_wav):
        validate_audio(sine_wav)

    def test_rejects_invalid_extension(self, tmp_path):
        path = tmp_path / "video.avi"
        path.write_bytes(b"data")
        with pytest.raises(AudioError, match="Formato no soportado"):
            validate_audio(path)

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        with pytest.raises(AudioError, match="vac"):
            validate_audio(path)

    def test_rejects_non_existent_file(self, tmp_path):
        path = tmp_path / "missing.wav"
        with pytest.raises(AudioError, match="no existe"):
            validate_audio(path)

    def test_supported_formats_are_six(self):
        assert SUPPORTED_FORMATS == {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp"}


class TestLoadAudio:
    def test_load_wav(self, sine_wav):
        audio, sr = load_audio(sine_wav)
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert len(audio) > 0

    @pytest.mark.parametrize("suffix", [".flac", ".mp3", ".ogg", ".m4a", ".3gp"])
    def test_load_other_formats(self, sine_wav_in_other_formats, suffix):
        audio, sr = load_audio(sine_wav_in_other_formats[suffix])
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert len(audio) > 0

    def test_any_duration_accepted(self, tmp_path, sine_signal):
        import soundfile as sf

        path = tmp_path / "short.wav"
        sf.write(path, sine_signal[: int(0.3 * 22050)], 22050)
        audio, _ = load_audio(path)
        assert len(audio) > 0

    def test_rejects_corrupt_audio(self, tmp_path):
        path = tmp_path / "fake.wav"
        path.write_bytes(b"RIFF not really audio data")
        with pytest.raises(AudioError, match="No se pudo procesar"):
            load_audio(path)


class TestReduceNoise:
    def test_noise_reduction_improves_snr(self, noisy_signal):
        clean, noisy = noisy_signal
        sr = 22050
        cleaned = reduce_noise(noisy, sr)
        snr_before = _snr(clean, noisy - clean)
        snr_after = _snr(clean, cleaned - clean)
        assert snr_after > snr_before

    def test_preserves_duration(self, sine_wav):
        audio, sr = load_audio(sine_wav)
        cleaned = reduce_noise(audio, sr)
        assert len(cleaned) == len(audio)


class TestFilterBabyCry:
    def test_bandpass_attenuates_out_of_band(self):
        sr = 22050
        t = np.linspace(0, 6, sr * 6)
        signal = (
            0.3 * np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.5 * np.sin(2 * np.pi * 3000 * t)
            + 0.3 * np.sin(2 * np.pi * 8000 * t)
        )
        processed = filter_baby_cry(signal, sr)

        e50 = _energy_in_band(processed, sr, 0, 100)
        e440 = _energy_in_band(processed, sr, 400, 500)
        e3000 = _energy_in_band(processed, sr, 2800, 3200)
        e8000 = _energy_in_band(processed, sr, 7800, 8200)

        assert 10 * np.log10(e50 / e440) < -30
        assert 10 * np.log10(e8000 / e3000) < -30

    def test_vad_removes_silence(self):
        sr = 22050
        t = np.linspace(0, 6, sr * 6)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        silence_mask = ~((t >= 2) & (t < 4))
        signal = signal * silence_mask
        processed = filter_baby_cry(signal, sr)
        assert len(processed) < len(signal)


class TestValidateSignalQuality:
    def test_accepts_clean_signal(self, sine_signal):
        assert validate_signal_quality(sine_signal) is True

    def test_rejects_totally_silent(self, sine_signal):
        assert validate_signal_quality(np.zeros_like(sine_signal)) is False

    def test_rejects_empty_array(self):
        assert validate_signal_quality(np.array([])) is False

    def test_rejects_mostly_silent(self):
        sr = 22050
        signal = np.zeros(sr * 2)
        signal[sr:sr + int(0.02 * sr)] = 0.5
        assert validate_signal_quality(signal) is False