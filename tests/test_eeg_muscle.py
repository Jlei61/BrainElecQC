"""Known-signal arithmetic and counterexamples; real recordings live outside Git."""

import json

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from brainelecqc.eeg.muscle import EEGMuscleConfig, compare_muscle_to_mne, compute_muscle_qc


def signals(rate=256, duration=40):
    rng = np.random.default_rng(4201)
    time = np.arange(rate * duration) / rate
    background = 8e-6 * np.sin(2 * np.pi * 10 * time) + rng.normal(0, .2e-6, len(time))
    muscle = sosfiltfilt(butter(4, [32, min(95, rate * .43)], fs=rate, btype="bandpass", output="sos"),
                         rng.normal(size=len(time)))
    amplitude = .2e-6 + 35e-6 * np.exp(-((time - 15) / .5) ** 4)
    emg = muscle * amplitude
    return np.array([background + emg * .7, background]), emg, rate


def test_native_clock_units_order_and_mne_reference_agree():
    data, emg, rate = signals()
    original = data.copy()
    result = compute_muscle_qc(data, rate, ["channel-b", "channel-a"], ["same", "same"],
        references={"emg": emg}, start_sample=87231, start_seconds=123.5,
        reference_channels={"emg": {"raw_name": "EMG chin", "channel_id": "aux-1"}})
    assert np.array_equal(data, original)
    assert result["channel_ids"] == ["channel-b", "channel-a"]
    assert result["reference"]["source_reference"]["raw_name"] == "EMG chin"
    assert result["parameters"]["effective_filter_band_hz"] == [30, 100]
    assert any(c["kind"] == "emg_supported_burst" and c["channel_index"] == 0 for c in result["candidates"])
    assert not any(c["channel_index"] == 1 for c in result["candidates"])
    for row in result["candidates"]:
        assert row["start_seconds"] == 123.5 + (row["start_sample"] - 87231) / rate
        assert row["stop_seconds"] == 123.5 + (row["stop_sample"] - 87231) / rate
    check = compare_muscle_to_mne(data, result)
    assert all(c["score_max_abs_error"] < 1e-12 for c in check["channels"])
    json.dumps(result, allow_nan=False)


def test_lower_sampling_rate_and_missing_emg_keep_general_candidate():
    data, _, rate = signals(rate=160)
    result = compute_muscle_qc(data, rate, ["a", "b"])
    assert result["parameters"]["effective_filter_band_hz"] == [30, 72]
    assert result["reference"]["status"] == "missing_reference"
    assert result["candidates"]
    assert all(c["kind"] == "high_frequency_burst" for c in result["candidates"])


def test_line_bursts_and_low_frequency_activity_are_counterexamples():
    rate = 256
    time = np.arange(rate * 40) / rate
    amplitude = 2e-6 + 90e-6 * np.exp(-((time - 15) / .5) ** 4)
    data = np.array([amplitude * np.sin(2*np.pi*f*time) for f in [10, 50, 60]])
    result = compute_muscle_qc(data, rate, ["alpha", "50Hz", "60Hz"])
    assert not result["candidates"]
    for channel in result["channels"][1:]:
        near_burst = [v for v in channel["window_metrics"] if 14 <= v["start_seconds"] <= 15]
        assert any(not v["gates"]["outside_line"] for v in near_burst)


def test_flatline_and_large_step_are_not_muscle_candidates():
    rate = 256
    time = np.arange(rate * 40) / rate
    row = 10e-6 * np.sin(2*np.pi*10*time)
    row[(time >= 15) & (time < 15.4)] = 0
    row[(time >= 25) & (time < 25.3)] = 2e-3
    result = compute_muscle_qc(row[None], rate, ["eeg"])
    assert not result["candidates"]
    assert result["channels"][0]["excluded_seconds"] >= 3
    blocked = result["sample_metrics"]["excluded"][0]
    assert all(blocked[round(14.5*rate):round(15.9*rate)])


def test_explicit_continuity_exclusion_uses_original_sample_clock_and_padding():
    data, emg, rate = signals()
    result = compute_muscle_qc(data, rate, ["a", "b"], references={"emg": emg}, start_sample=10000,
        excluded_intervals=[{"channel_id": "a", "start_sample": 10000+14*rate, "stop_sample": 10000+16*rate}])
    assert not result["candidates"]
    assert result["channels"][0]["excluded_seconds"] == 4
    assert result["channels"][0]["valid_seconds"] == 36
    assert result["channels"][1]["excluded_seconds"] == 1


def test_uncorrelated_emg_does_not_promote_general_high_frequency():
    data, emg, rate = signals()
    unrelated = np.roll(emg, 8 * rate)
    result = compute_muscle_qc(data, rate, ["a", "b"], references={"emg": unrelated})
    assert result["candidates"]
    assert all(c["kind"] == "high_frequency_burst" for c in result["candidates"])


def test_missing_bad_or_constant_reference_status_and_nonfinite_channel():
    data, emg, rate = signals()
    data[1, 5] = np.nan
    for ref, state in [(np.zeros_like(emg), "constant_reference"), (np.full_like(emg, np.nan), "nonfinite_reference")]:
        result = compute_muscle_qc(data, rate, ["a", "b"], references={"emg": ref})
        assert result["status"] == "partial_channels"
        assert result["reference"]["status"] == state
        assert result["channels"][1]["status"] == "nonfinite_input"
        assert result["sample_metrics"]["z_score"][1] is None
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rate,duration,status", [(64, 40, "insufficient_sampling_rate"), (256, 3, "insufficient_duration")])
def test_unavailable_scope_is_explicit(rate, duration, status):
    result = compute_muscle_qc(np.zeros((1, rate*duration)), rate, ["a"])
    assert result["status"] == status
    assert not result["candidates"]


@pytest.mark.parametrize("config", [{"upper_sampling_fraction": .5}, {"window_seconds": 0},
    {"z_threshold": float("nan")}, {"min_duration_seconds": 2}, {"step_seconds": 2},
    {"envelope_correlation_threshold": 2}, {"constant_tolerance_volts": True}])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        EEGMuscleConfig(**config)


def test_misaligned_metadata_and_reference_fingerprint_fail():
    data, emg, rate = signals()
    with pytest.raises(ValueError, match="sample count"):
        compute_muscle_qc(data, rate, ["a", "b"], references={"emg": emg[:-1]})
    with pytest.raises(ValueError, match="distinct"):
        compute_muscle_qc(data, rate, ["a", "a"])
    with pytest.raises(ValueError, match="unknown channel_id"):
        compute_muscle_qc(data, rate, ["a", "b"], excluded_intervals=[{"channel_id": "c", "start_sample": 1, "stop_sample": 2}])
    result = compute_muscle_qc(data, rate, ["a", "b"])
    with pytest.raises(ValueError, match="fingerprint"):
        compare_muscle_to_mne(data + 1e-6, result)
