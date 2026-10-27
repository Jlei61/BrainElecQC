"""Raw continuity intervals are local, physical-unit aware and reviewable."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from brainelecqc.eeg.continuity import ContinuityConfig, compute_continuity_qc


def metadata(ids, *, step=1e-6, lower=-800e-6, upper=800e-6):
    return [{"channel_id": key, "raw_name": key, "unit": "V", "physical_min": lower,
             "physical_max": upper, "scale_to_si": 1.0, "digital_to_physical_scale": step}
            for key in ids]


def test_three_constant_classes_keep_exact_ranges_and_input():
    x = np.sin(2 * np.pi * 10 * np.arange(1000) / 1000)[None] * 5e-6
    x[0, 100:200] = 0
    x[0, 300:450] = 800e-6
    x[0, 600:800] = 42e-6
    before = x.copy()
    result = compute_continuity_qc(x, 1000, ["a"], channels=metadata(["a"]))
    assert [(c["kind"], c["start_sample"], c["stop_sample"]) for c in result["candidates"]] == [
        ("near_zero_flatline", 100, 200), ("calibration_endpoint_plateau", 300, 450),
        ("constant_value", 600, 800)]
    assert result["candidates"][1]["evidence"]["endpoint"] == "upper"
    assert result["candidates"][1]["calibration_status"] == "available"
    assert result["measurements"][0]["constant_sample_count"] == 450
    assert result["automatic_exclusions"] == []
    assert all(c["action"] == "review" for c in result["candidates"])
    assert result["data_sha256"] == hashlib.sha256(before.astype("<f8").tobytes()).hexdigest()
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("amplitude", [0.05e-6, 1e-12, 1e-17])
def test_continuously_varying_low_amplitude_alpha_and_drift_stay_variable(amplitude):
    t = np.arange(1000) / 1000
    x = amplitude * np.vstack([np.sin(2 * np.pi * 10 * t), t])
    result = compute_continuity_qc(x, 1000, ["alpha", "drift"], channels=metadata(["alpha", "drift"]))
    assert result["candidates"] == []
    assert all(c["status"] == "no_constant_run_at_threshold" for c in result["measurements"])
    assert all(c["changing_adjacent_pair_count"] > 900 for c in result["measurements"])


def test_quantization_uses_half_step_for_zero_but_keeps_neighbor_codes_separate():
    # Each code remains for 50 ms: neighboring codes must never merge into 1 s flatline.
    values = np.tile(np.r_[np.full(50, -.5e-6), np.full(50, .5e-6)], 10)
    result = compute_continuity_qc(values[None], 1000, ["a"], channels=metadata(["a"]))
    assert result["candidates"] == []
    flat = compute_continuity_qc(np.full((1, 100), -.5e-6), 1000, ["a"], channels=metadata(["a"]))
    assert flat["candidates"][0]["kind"] == "near_zero_flatline"
    assert flat["candidates"][0]["evidence"]["calibration"]["quantization_step_volts"] == 1e-6


def test_constant_near_but_one_code_below_endpoint_has_its_own_class():
    result = compute_continuity_qc(np.full((1, 100), 799e-6), 1000, ["a"], channels=metadata(["a"]))
    assert result["candidates"][0]["kind"] == "constant_value"
    assert result["candidates"][0]["evidence"]["endpoint"] is None


def test_endpoint_precedes_near_zero_for_unipolar_file_range():
    result = compute_continuity_qc(np.zeros((1, 100)), 1000, ["a"], channels=metadata(["a"], lower=0))
    assert result["candidates"][0]["kind"] == "calibration_endpoint_plateau"
    assert result["candidates"][0]["evidence"]["endpoint"] == "lower"


def test_partial_and_absent_calibration_preserve_constant_observation():
    x = np.full((1, 100), 800e-6)
    absent = compute_continuity_qc(x, 1000, ["a"])
    partial = compute_continuity_qc(x, 1000, ["a"], channels=[{"channel_id": "a", "unit": "V"}])
    assert absent["candidates"][0]["calibration_status"] == "unavailable"
    assert partial["candidates"][0]["calibration_status"] == "partial"
    assert absent["candidates"][0]["kind"] == partial["candidates"][0]["kind"] == "constant_value"


def test_absolute_clock_is_independent_of_sample_origin_and_runs_at_boundaries():
    x = np.r_[np.zeros(26), np.arange(1, 75) * 1e-6][None]
    result = compute_continuity_qc(x, 256, ["a"], start_sample=3200, start_seconds=123.5)
    candidate = result["candidates"][0]
    assert (candidate["start_sample"], candidate["stop_sample"]) == (3200, 3226)
    assert candidate["start_seconds"] == 123.5
    assert candidate["stop_seconds"] == 123.5 + 26 / 256
    assert candidate["evidence"]["touches_input_start"]
    assert not candidate["evidence"]["touches_input_stop"]
    assert result["parameters"]["min_constant_samples"] == 26
    assert result["parameters"]["effective_min_constant_seconds"] == 26 / 256


def test_one_sample_shorter_than_threshold_is_retained():
    x = np.r_[np.zeros(25), np.arange(1, 76) * 1e-6][None]
    assert compute_continuity_qc(x, 256, ["a"])["candidates"] == []


def test_nonfinite_runs_do_not_join_flat_runs_and_json_remains_finite():
    x = np.r_[np.zeros(90), np.nan, np.zeros(90), np.inf, -np.inf][None]
    result = compute_continuity_qc(x, 1000, ["a"])
    assert [(c["kind"], c["start_sample"], c["stop_sample"]) for c in result["candidates"]] == [
        ("nonfinite_input", 90, 91), ("nonfinite_input", 181, 183)]
    assert result["measurements"][0]["nonfinite_sample_count"] == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n_channels", [1, 2, 3, 4])
def test_each_channel_is_evaluated_without_a_reference_group_gate(n_channels):
    ids = [f"ch{n}" for n in range(n_channels)]
    result = compute_continuity_qc(np.zeros((n_channels, 100)), 256, ids)
    assert len(result["candidates"]) == n_channels
    assert all(c["kind"] == "near_zero_flatline" for c in result["candidates"])


def test_calibration_order_uses_stable_ids_and_allows_duplicate_raw_names():
    channels = metadata(["a", "b"])
    for c in channels:
        c["raw_name"] = "same"
    channels[1]["physical_max"] = 900e-6
    result = compute_continuity_qc(np.full((2, 100), 800e-6), 1000, ["a", "b"], ["same", "same"],
                                   channels=channels[::-1])
    assert [c["kind"] for c in result["candidates"]] == ["calibration_endpoint_plateau", "constant_value"]


def test_negative_calibration_gain_keeps_sorted_endpoints_and_positive_step():
    channels = metadata(["a"], step=-1e-6, lower=800e-6, upper=-800e-6)
    result = compute_continuity_qc(np.full((1, 100), -800e-6), 1000, ["a"], channels=channels)
    assert result["candidates"][0]["evidence"]["endpoint"] == "lower"
    assert result["measurements"][0]["calibration"]["quantization_step_volts"] == 1e-6


@pytest.mark.parametrize("updates", [{"sfreq": 0}, {"sfreq": True}, {"start_sample": -.1},
    {"start_seconds": float("nan")}, {"channel_ids": ["a", "a"]}, {"channel_names": ["a", "b"]},
    {"channels": [{"channel_id": "b", "unit": "V"}]},
    {"channels": [{"channel_id": "a", "unit": "uV"}]},
    {"channels": [{"channel_id": "a", "raw_name": "other", "unit": "V"}]},
    {"channels": [{"channel_id": "a", "unit": "V", "scale_to_si": float("inf")}]},
    {"config": {"min_constant_seconds": 0}}, {"config": {"endpoint_tolerance_lsb": .5}}])
def test_invalid_identity_units_and_parameters_raise(updates):
    args = {"data_volts": np.ones((1, 100)), "sfreq": 256, "channel_ids": ["a"], **updates}
    with pytest.raises(ValueError):
        compute_continuity_qc(**args)


def test_config_object_and_mapping_agree():
    args = (np.zeros((1, 100)), 256, ["a"])
    assert compute_continuity_qc(*args, config=ContinuityConfig()) == compute_continuity_qc(*args, config={})


def test_real_eeg_bundle_regression():
    """Optional external public bundle; repository contains only expected coordinates."""
    location = os.environ.get("BRAINELECQC_REAL_EEG_BUNDLE")
    if not location:
        pytest.skip("Set BRAINELECQC_REAL_EEG_BUNDLE to the external real EEG validation bundle")
    bundle = Path(location)
    manifest = json.loads((bundle / "manifest.json").read_text())
    features = json.loads((bundle / "features.json").read_text())
    channels = json.loads((bundle / "channels.json").read_text())
    for filename in ("channels.json", "review_signals.npz", "features.json"):
        assert hashlib.sha256((bundle / filename).read_bytes()).hexdigest() == manifest["files"][filename]["sha256"]
    with np.load(bundle / "review_signals.npz", allow_pickle=False) as signals:
        data = signals["raw_data_volts"]
    result = compute_continuity_qc(data, features["sampling_rate_hz"], features["channel_ids"],
        features["channel_names"], channels=channels, start_sample=features["start_sample"],
        start_seconds=features["start_seconds"])
    found = {(c["channel_name"], c["kind"], c["start_sample"], c["stop_sample"]) for c in result["candidates"]}
    assert {("EEG F4-M1", "near_zero_flatline", 11536, 11660),
            ("EEG C4-M1", "near_zero_flatline", 11536, 11660),
            ("EEG O2-M1", "near_zero_flatline", 11563, 11650),
            ("EEG O2-M1", "calibration_endpoint_plateau", 10876, 11222)} <= found
    assert result["data_sha256"] == features["data_sha256"]
    assert result["automatic_exclusions"] == []
