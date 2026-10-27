"""Screening observations, counterexamples, and sample-clock integrity."""

from copy import deepcopy
import json

import numpy as np
import pytest

from brainelecqc.eeg.features import compute_eeg_features
from brainelecqc.eeg.masks import apply_eeg_decisions, retained_intervals, true_runs, validate_binding
from brainelecqc.eeg.policy import EEGPolicy, propose_eeg_decisions


def _wave(fs=100, seconds=10, channels=1):
    t = np.arange(round(fs * seconds)) / fs
    return np.stack([(20 + i * 5) * 1e-6 * np.sin(2 * np.pi * (7 + i) * t + .3)
                     for i in range(channels)])


def _features(data, fs=100, origin=0, seconds=0.0):
    return compute_eeg_features(data, fs, [f"eeg-{i}" for i in range(len(data))],
                                ["duplicate"] * len(data), start_sample=origin, start_seconds=seconds)


def _bounds(ids, lower=-100e-6, upper=100e-6, tolerance=0.0):
    return {cid: {"lower_volts": lower, "upper_volts": upper,
                  "tolerance_volts": tolerance, "source": "fixture calibration"} for cid in ids}


def _exclude(cid, a, b):
    return {"channel_id": cid, "start_sample": a, "stop_sample": b, "decision_source": "automatic"}


def _technical(result):
    assert result["automatic_exclusions"] == []
    assert result["summary"]["excluded_channel_samples"] == 0
    return [x for x in result["review_queue"] if x["kind"] in {"file_endpoint_plateau", "flatline"}]


@pytest.mark.parametrize("side,value", [("lower", -100e-6), ("upper", 100e-6)])
def test_rail_threshold_same_side_and_padding(side, value):
    data = _wave()
    data[0, 300:310] = value
    features = _features(data, origin=2500, seconds=43.25)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    assert len(_technical(result)) == 1
    item = _technical(result)[0]
    assert item["kind"] == "file_endpoint_plateau"
    assert item["decision_source"] == "review_pending"
    assert item["evidence"]["boundary"] == side
    assert (item["evidence"]["core_start_sample"], item["evidence"]["core_stop_sample"]) == (2800, 2810)
    assert (item["start_sample"], item["stop_sample"]) == (2800, 2810)
    assert (item["evidence"]["context_start_sample"], item["evidence"]["context_stop_sample"]) == (2775, 2835)
    assert (item["start_seconds"], item["stop_seconds"]) == pytest.approx((46.25, 46.35))
    assert result["channel_summary"][0]["excluded_samples"] == 0
    masked, keep = apply_eeg_decisions(data, result, channel_ids=features["channel_ids"])
    assert keep.all()
    np.testing.assert_array_equal(masked, data)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sample_count,expected", [(12, 0), (13, 1)])
def test_duration_threshold_uses_ceiling_of_fractional_sample_count(sample_count, expected):
    data = _wave(fs=128)
    data[0, 300:300 + sample_count] = 100e-6
    features = _features(data, fs=128)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    assert len(_technical(result)) == expected
    if expected:
        assert _technical(result)[0]["start_sample"] == 300
        assert _technical(result)[0]["stop_sample"] == 313


@pytest.mark.parametrize("start,stop,expected", [(0, 10, (0, 35)), (990, 1000, (965, 1000))])
def test_padding_clips_to_segment_boundaries(start, stop, expected):
    data = _wave()
    data[0, start:stop] = 100e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    item = _technical(result)[0]
    assert (item["start_sample"], item["stop_sample"]) == (start, stop)
    assert (item["evidence"]["context_start_sample"], item["evidence"]["context_stop_sample"]) == expected


@pytest.mark.parametrize("case", ["isolated", "short", "alternating", "normal", "unknown_extreme"])
def test_spikes_extrema_and_normal_waveforms_are_counterexamples(case):
    data = _wave()
    if case == "isolated":
        data[0, [250, 400, 600]] = 100e-6
    elif case == "short":
        data[0, 300:309] = 100e-6
    elif case == "alternating":
        data[0, 300:330] = np.tile([-100e-6, 100e-6], 15)
    elif case == "unknown_extreme":
        data[0, 300:350] = 80e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    assert _technical(result) == []
    masked, keep = apply_eeg_decisions(data, result, channel_ids=features["channel_ids"])
    assert keep.all()
    np.testing.assert_array_equal(masked, data)


def test_long_flat_segment_and_constant_channel():
    data = _wave(channels=2)
    data[0, 300:400] = 7e-6
    data[1] = 3e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features)
    exclusions = _technical(result)
    assert len(exclusions) == 2
    full = next(x for x in exclusions if x["channel_id"] == "eeg-1")
    part = next(x for x in exclusions if x["channel_id"] == "eeg-0")
    assert (part["start_sample"], part["stop_sample"]) == (300, 400)
    assert (full["start_sample"], full["stop_sample"]) == (0, 1000)
    assert result["summary"]["fully_excluded_channel_ids"] == []
    assert result["summary"]["excluded_channel_samples"] == 0


def test_short_flat_segment_does_not_trigger_full_second_rule():
    data = _wave()
    data[0, 300:399] = 7e-6
    result = propose_eeg_decisions(data, _features(data))
    assert _technical(result) == []


def test_clipping_owns_long_rail_plateau_without_flat_duplicate():
    data = _wave()
    data[0, 300:500] = 100e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    assert [x["kind"] for x in _technical(result)] == ["file_endpoint_plateau"]
    assert "flat_support" in _technical(result)[0]["evidence"]


def test_near_rail_transition_keeps_single_plateau_exclusion():
    data = _wave()
    data[0, 299] = 100e-6 - .005e-6
    data[0, 300:500] = 100e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    # Related observations share one candidate, each retaining its own exact support.
    assert len(_technical(result)) == 1
    assert _technical(result)[0]["start_sample"] == 300
    assert _technical(result)[0]["evidence"]["flat_support"]["core_start_sample"] == 299


def test_overlapping_review_context_preserves_all_samples():
    data = _wave(channels=2)
    data[0, 300:310] = 100e-6
    data[0, 330:340] = -100e-6
    features = _features(data)
    result = propose_eeg_decisions(data, features, calibration_bounds=_bounds(features["channel_ids"]))
    assert len(_technical(result)) == 2
    assert result["channel_summary"][0]["excluded_samples"] == 0
    assert result["summary"]["excluded_fraction"] == 0


def test_mask_preserves_order_names_original_values_and_half_open_clock():
    data = _wave(channels=2)
    before = data.copy()
    features = _features(data, origin=1000, seconds=31.25)
    decisions = propose_eeg_decisions(data, features)
    decisions["automatic_exclusions"] = [
        _exclude("eeg-1", 1000, 1010), _exclude("eeg-0", 1100, 1200), _exclude("eeg-0", 1150, 1250)]
    data.setflags(write=False)
    masked, keep = apply_eeg_decisions(data, decisions, channel_ids=["eeg-0", "eeg-1"])
    assert masked.shape == data.shape
    assert (~keep).sum() == 160
    np.testing.assert_array_equal(masked[keep], data[keep])
    assert np.isnan(masked[~keep]).all()
    np.testing.assert_array_equal(data, before)
    assert keep[0, 99] and not keep[0, 100] and not keep[0, 249] and keep[0, 250]
    intervals = list(retained_intervals(keep, decisions))
    assert intervals == [
        {"channel_id": "eeg-0", "start_sample": 1000, "stop_sample": 1100,
         "start_seconds": 31.25, "stop_seconds": 32.25, "status": "retain"},
        {"channel_id": "eeg-0", "start_sample": 1250, "stop_sample": 2000,
         "start_seconds": 33.75, "stop_seconds": 41.25, "status": "retain"},
        {"channel_id": "eeg-1", "start_sample": 1010, "stop_sample": 2000,
         "start_seconds": 31.35, "stop_seconds": 41.25, "status": "retain"}]


def test_alpha_feature_remains_observation_and_reference_evidence_stays_reviewable():
    fs = 256
    t = np.arange(fs * 8) / fs
    data = (20e-6 * np.sin(2*np.pi*10*t))[None]
    features = compute_eeg_features(data, fs, ["x"], references={"eog": -data[0]})
    result = propose_eeg_decisions(data, features)
    assert result["automatic_exclusions"] == []
    assert len(result["review_queue"]) == 2
    assert {x["kind"] for x in result["review_queue"]} == {"eog"}
    assert all(x["decision_source"] == "review_pending" for x in result["review_queue"])


def test_fingerprint_and_reordered_channels_reject_different_data():
    data = _wave(channels=2)
    features = _features(data)
    decisions = propose_eeg_decisions(data, features)
    changed = data.copy()
    changed[0, 1] += 1e-9
    for changed_data in (changed, data[::-1], data[:, :-1]):
        with pytest.raises(ValueError):
            apply_eeg_decisions(changed_data, decisions, channel_ids=features["channel_ids"])
    with pytest.raises(ValueError):
        apply_eeg_decisions(data, decisions, channel_ids=["eeg-1", "eeg-0"])


@pytest.mark.parametrize("change", [
    {"channel_id": "missing"}, {"start_sample": -1}, {"stop_sample": 1001},
    {"start_sample": 150, "stop_sample": 150}, {"start_sample": 151, "stop_sample": 150},
    {"start_sample": True}, {"start_sample": 1.0}, {"decision_source": "review_pending"},
])
def test_invalid_exclusion_ranges_raise(change):
    data = _wave()
    decisions = propose_eeg_decisions(data, _features(data))
    item = _exclude("eeg-0", 100, 150)
    item.update(change)
    decisions["automatic_exclusions"] = [item]
    with pytest.raises(ValueError):
        apply_eeg_decisions(data, decisions, channel_ids=["eeg-0"])


@pytest.mark.parametrize("change", [
    {"sampling_rate_hz": 0}, {"sampling_rate_hz": -100}, {"sampling_rate_hz": float("nan")},
    {"start_seconds": float("nan")}, {"start_sample": .5, "stop_sample": 1000.5},
    {"start_sample": True, "stop_sample": 1001},
])
def test_binding_validates_numeric_clock_metadata(change):
    data = _wave()
    features = _features(data)
    features.update(change)
    with pytest.raises(ValueError):
        validate_binding(data, features)


@pytest.mark.parametrize("mask", [np.ones((1, 999), dtype=bool), np.ones((2, 1000), dtype=bool),
                                   np.ones((1, 1000), dtype=float)])
def test_retained_intervals_rejects_inconsistent_mask_shape_or_type(mask):
    data = _wave()
    metadata = _features(data)
    with pytest.raises(ValueError):
        list(retained_intervals(mask, metadata))


@pytest.mark.parametrize("change", [
    {"clipping_min_seconds": 0}, {"clipping_min_seconds": -1}, {"padding_seconds": -1},
    {"flat_min_seconds": 0}, {"flat_range_uv": float("nan")}, {"review_correlation": 2},
    {"review_high_frequency_fraction": 1.1}, {"review_rms_uv": True},
])
def test_policy_rejects_invalid_parameters(change):
    with pytest.raises(ValueError):
        EEGPolicy(**change)


@pytest.mark.parametrize("change", [
    {"lower_volts": 100e-6}, {"upper_volts": float("inf")},
    {"tolerance_volts": -1}, {"tolerance_volts": 200e-6}, {"source": ""},
])
def test_invalid_calibration_bounds_raise(change):
    data = _wave()
    features = _features(data)
    bounds = _bounds(features["channel_ids"])
    bounds["eeg-0"].update(change)
    with pytest.raises(ValueError):
        propose_eeg_decisions(data, features, calibration_bounds=bounds)


def test_true_runs_half_open_edge_cases():
    assert true_runs(np.array([], dtype=bool)) == []
    assert true_runs(np.array([True, True, False, True])) == [(0, 2), (3, 4)]
    assert true_runs(np.array([False, False])) == []
