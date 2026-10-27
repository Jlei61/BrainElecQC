"""Numerical contracts for the pure EEG feature calculation."""

import hashlib
import json

import numpy as np
import pytest
from scipy.signal import welch

from brainelecqc.eeg.config import EEGConfig
from brainelecqc.eeg.features import _alpha_peak, _background, compute_eeg_features


def _sine(hz=10, amplitude_uv=20, fs=256, seconds=8):
    return amplitude_uv * 1e-6 * np.sin(2 * np.pi * hz * np.arange(round(fs * seconds)) / fs)


def test_voltage_scale_alpha_peak_and_psd_power():
    data = _sine()[None, :]
    result = compute_eeg_features(data, 256, ["eeg0"], ["Oz"])
    channel = result["channel_features"][0]
    assert channel["rms_uv"] == pytest.approx(20 / np.sqrt(2))
    assert channel["alpha_peak_hz"] == pytest.approx(10)
    assert channel["alpha_prominence_db"] > 20
    assert channel["band_powers_v2"]["alpha"] == pytest.approx(200e-12)
    f, psd = np.asarray(result["frequencies_hz"]), np.asarray(result["psd_v2_hz"])[0]
    assert np.sum(psd) * (f[1] - f[0]) == pytest.approx(200e-12)
    assert result["spectral_metadata"]["frequency_resolution_hz"] == 0.25
    assert result["channel_features"][0]["alpha_window_fraction"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("frequencies, expected", [((50,), "50"), ((60,), "60"), ((50, 60), "both")])
def test_distinguishes_50_60_and_both(frequencies, expected):
    data = np.random.default_rng(7).normal(0, 2e-6, 4096)
    for hz in frequencies:
        data += _sine(hz, 30, 256, 16)
    result = compute_eeg_features(data[None], 256, ["x"])
    channel = result["channel_features"][0]
    assert channel["line_frequency"] == expected
    for hz in frequencies:
        assert channel[f"line_{hz}_db"] > 20


def test_line_harmonic_evidence_keeps_fundamental_identity():
    rng = np.random.default_rng(12)
    data = rng.normal(0, 1e-6, 8 * 500) + _sine(50, 20, 500) + _sine(100, 15, 500)
    result = compute_eeg_features(data[None], 500, ["x"])
    harmonics = result["channel_features"][0]["line_harmonics"]
    item = next(x for x in harmonics if x["fundamental_hz"] == 50 and x["harmonic"] == 2)
    assert item["detected"] is True
    assert item["peak_hz"] == 100


def test_low_sampling_rate_reports_unavailable_bands():
    result = compute_eeg_features(_sine(fs=40)[None], 40, ["x"])
    channel = result["channel_features"][0]
    assert channel["alpha_peak_hz"] == 10
    assert channel["line_frequency"] == "insufficient"
    assert channel["line_50_db"] is None
    assert channel["line_60_status"] == "insufficient_bandwidth"
    assert result["window_features"]["high_frequency_fraction"][0] == [None, None]
    json.dumps(result, allow_nan=False)


def test_low_frequency_band_requires_its_upper_edge():
    result = compute_eeg_features(_sine(hz=1, fs=6)[None], 6, ["x"])
    assert result["window_features"]["low_frequency_fraction"][0] == [None, None]
    assert result["window_feature_status"]["low_frequency_fraction"][0] == ["insufficient_bandwidth"] * 2


def test_monotonic_one_over_f_is_background_and_has_no_alpha_peak():
    f = np.arange(0, 128.25, 0.25)
    psd = 1e-12 / np.maximum(f, 0.25)
    result = _alpha_peak(f, psd, EEGConfig())
    background = _background(f, psd, EEGConfig())
    assert result == {"status": "no_prominent_peak", "peak_hz": None, "prominence_db": None}
    assert background["slope"] == pytest.approx(-1)
    assert background["rmse_db"] < 1e-10


def test_duplicate_names_remain_separate_and_input_stays_identical():
    data = np.stack([_sine(), _sine(amplitude_uv=40)])
    before = data.copy()
    data.setflags(write=False)
    result = compute_eeg_features(data, 256, ["id-a", "id-b"], ["EEG", "EEG"])
    assert result["channel_ids"] == ["id-a", "id-b"]
    assert result["channel_names"] == ["EEG", "EEG"]
    assert result["window_features"]["relative_rms"][0] == pytest.approx([2/3, 2/3])
    assert result["window_features"]["relative_rms"][1] == pytest.approx([4/3, 4/3])
    np.testing.assert_array_equal(data, before)
    assert result["data_sha256"] == hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes()).hexdigest()


def test_short_tail_retains_sample_origin_and_actual_resolution():
    data = _sine(seconds=4.5)[None]
    result = compute_eeg_features(data, 256, ["x"], start_sample=6400, start_seconds=50.25)
    assert [(w["start_sample"], w["stop_sample"]) for w in result["windows"]] == [(6400, 7424), (7424, 7552)]
    assert result["windows"][1]["start_seconds"] == 54.25
    assert result["stop_seconds"] == 54.75
    assert result["window_spectral_metadata"][1]["status"] == "insufficient_duration"
    assert result["window_features"]["alpha_peak_hz"][0][1] is None
    assert result["window_features"]["rms_uv"][0][1] > 0
    assert result["spectral_metadata"]["tail_samples"] == 128
    assert result["spectral_metadata"]["periodogram_count"] == 1
    assert result["channel_features"][0]["alpha_assessed_window_count"] == 1
    assert result["channel_features"][0]["alpha_window_fraction"] == 1


def test_short_valid_record_has_its_own_frequency_resolution():
    result = compute_eeg_features(_sine(seconds=1.25)[None], 256, ["x"])
    assert result["spectral_metadata"]["nperseg"] == 320
    assert result["spectral_metadata"]["nfft"] == 320
    assert result["spectral_metadata"]["actual_welch_seconds"] == 1.25
    assert result["spectral_metadata"]["frequency_resolution_hz"] == 0.8


def test_total_psd_matches_equal_weight_complete_welch_windows():
    data = np.random.default_rng(45).normal(0, 2e-5, (2, 2300))
    result = compute_eeg_features(data, 256, ["a", "b"])
    f, psd = welch(data, fs=256, window="hann", nperseg=1024, noverlap=512, nfft=1024,
                   detrend="constant", scaling="density", average="mean", axis=-1)
    np.testing.assert_array_equal(result["frequencies_hz"], f)
    np.testing.assert_array_equal(result["psd_v2_hz"], psd)
    assert result["spectral_metadata"]["periodogram_start_offsets"] == [0, 512, 1024]
    assert result["spectral_metadata"]["periodogram_weights"] == [1/3, 1/3, 1/3]
    assert result["spectral_metadata"]["covered_samples"] == 2048
    assert result["spectral_metadata"]["tail_samples"] == 252


def test_ecg_eog_signed_correlation_and_emg_envelope():
    fs = 256
    t = np.arange(fs * 8) / fs
    signal = (2 + np.sin(2*np.pi*0.7*t)) * _sine(80, 10)
    result = compute_eeg_features(signal[None], fs, ["x"],
                                  references={"ecg": signal.copy(), "eog": -signal, "emg": signal.copy()})
    assert result["window_features"]["ecg_correlation"][0] == pytest.approx([1, 1])
    assert result["window_features"]["eog_correlation"][0] == pytest.approx([-1, -1])
    assert result["window_features"]["emg_envelope_correlation"][0] == pytest.approx([1, 1])
    assert result["window_features"]["high_frequency_fraction"][0] == pytest.approx([1, 1], abs=1e-4)


def test_zero_channels_and_missing_references_have_json_null():
    result = compute_eeg_features(np.zeros((2, 1024)), 256, ["a", "b"])
    assert result["window_features"]["rms_uv"] == [[0.0], [0.0]]
    assert result["window_features"]["flat_fraction"] == [[1.0], [1.0]]
    assert result["window_features"]["relative_rms"] == [[None], [None]]
    assert result["window_feature_status"]["high_frequency_fraction"] == [["no_power"], ["no_power"]]
    assert result["window_feature_status"]["ecg_correlation"] == [["missing_reference"], ["missing_reference"]]
    json.dumps(result, allow_nan=False)


def test_large_amplitude_is_centered_on_median_and_preserves_dc():
    data = np.full((1, 1024), 0.004)
    data[0, 400:420] += 0.001
    result = compute_eeg_features(data, 256, ["x"])
    assert result["window_features"]["large_amplitude_fraction"][0][0] == 20 / 1024
    assert data[0, 0] == 0.004


@pytest.mark.parametrize("change", [
    {"data_volts": np.array([[np.nan]])}, {"data_volts": np.array([[np.inf]])},
    {"data_volts": [[True]]}, {"data_volts": [["1"]]}, {"data_volts": [[1j]]},
    {"data_volts": np.zeros((1, 0))}, {"data_volts": np.zeros(100)},
    {"sfreq": 0}, {"sfreq": True}, {"sfreq": float("inf")},
    {"channel_ids": ["a", "a"]}, {"channel_ids": [""]}, {"channel_ids": "a"},
    {"channel_names": ["a", "b"]}, {"start_sample": 0.5}, {"start_sample": -1},
    {"start_seconds": float("nan")}, {"references": {"ecg": np.zeros(3)}},
    {"references": {"eog": np.full(256, np.nan)}}, {"references": {"unknown": np.zeros(256)}},
])
def test_invalid_inputs_raise(change):
    args = dict(data_volts=np.zeros((1, 256)), sfreq=256, channel_ids=["x"])
    args.update(change)
    with pytest.raises(ValueError):
        compute_eeg_features(**args)


@pytest.mark.parametrize("parameters", [{"window_seconds": 0}, {"welch_overlap_fraction": 1},
    {"alpha_band": (13, 8)}, {"min_spectral_seconds": 5}, {"flat_step_uv": -1},
    {"line_peak_half_width_hz": 3}, {"window_seconds": True}])
def test_invalid_config_raises(parameters):
    with pytest.raises(ValueError):
        EEGConfig(**parameters)
