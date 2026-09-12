import numpy as np
import pytest

from app.services.feature_extractor import FEATURE_SIZE, extract_features


class TestExtractFeatures:
    def test_returns_correct_dimension(self, sine_signal):
        features = extract_features(sine_signal, 22050)
        assert features.shape == (FEATURE_SIZE,)

    def test_returns_float32(self, sine_signal):
        features = extract_features(sine_signal, 22050)
        assert features.dtype == np.float32

    def test_same_dimension_for_different_signals(self, sine_signal):
        sr = 22050
        t = np.linspace(0, 3, sr * 3)
        other = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        f1 = extract_features(sine_signal, sr)
        f2 = extract_features(other, sr)
        assert f1.shape == f2.shape

    def test_short_audio_supported(self, sine_signal):
        short = sine_signal[: int(0.1 * 22050)]
        features = extract_features(short, 22050)
        assert features.shape == (FEATURE_SIZE,)

    def test_very_short_audio_supported(self, sine_signal):
        very_short = sine_signal[:100]
        features = extract_features(very_short, 22050)
        assert features.shape == (FEATURE_SIZE,)

    def test_features_are_finite(self, sine_signal):
        features = extract_features(sine_signal, 22050)
        assert np.all(np.isfinite(features))

    def test_noisy_and_clean_signals_differ(self, sine_signal):
        sr = 22050
        rng = np.random.default_rng(0)
        feature_clean = extract_features(sine_signal, sr)
        feature_noisy = extract_features(sine_signal + 0.3 * rng.standard_normal(len(sine_signal)), sr)
        assert not np.allclose(feature_clean, feature_noisy)

    def test_real_audio_feature(self):
        from pathlib import Path

        from app.services.audio_processor import load_audio

        wav_file = list(
            Path("/home/mds/PhD/MY_PROJECTS/infant-cri-ai/Dataset/archive/cry/belly pain").glob("*.wav")
        )[0]
        audio, sr = load_audio(wav_file)
        features = extract_features(audio, sr)
        assert features.shape == (FEATURE_SIZE,)
        assert np.all(np.isfinite(features))