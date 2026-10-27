"""Algorithm contracts and counterexamples; clinical accuracy needs labelled data."""

import json

import numpy as np
import pytest

from brainelecqc.eeg.bad_channels import (
    BadChannelConfig, compare_with_pyprep, compute_bad_channels,
)


def _data(channels=16, seconds=12, sfreq=256):
    rng = np.random.default_rng(2518)
    t = np.arange(int(sfreq * seconds)) / sfreq
    gains = np.linspace(0.8, 1.2, channels)
    shared = 10e-6 * np.sin(2 * np.pi * 10 * t) + 4e-6 * np.sin(2 * np.pi * 3 * t)
    data = gains[:, None] * shared + rng.normal(size=(channels, len(t))) * 2e-6
    return data, [f"id:{i}" for i in range(channels)]


def test_reference_comparison_is_independent_and_preserves_extra_complete_window():
    pytest.importorskip("pyprep")
    data, ids = _data()
    data[0] *= 20
    result = compare_with_pyprep(data, 256, ids, groups=["same_reference"] * len(ids))
    assert result["reference_version"] == "0.5.0"
    assert result["all_compared_measurements_match"]
    group = result["groups"][0]
    assert group["reference_windows"] == 11
    assert group["output_windows"] == 12
    assert group["additional_output_windows"][0]["start_sample"] == 2816
    assert group["reference_candidate_channels"]["bad_by_deviation"] == [ids[0]]
    assert all(group["candidate_comparison"]["common_window_matches"].values())
    assert all(check["max_abs_difference"] < 1e-8 for check in group["checks"].values())


def test_high_amplitude_is_traceable_review_candidate_without_exclusion():
    data, ids = _data()
    data[0] *= 20
    original = data.copy()
    result = compute_bad_channels(data, 256, ids, ["same name"] * len(ids), groups=["reference"] * len(ids), start_sample=4096, start_seconds=100)
    candidate = next(item for item in result["candidates"] if item["kind"] == "amplitude_deviation")
    assert candidate["channel_id"] == ids[0]
    assert candidate["start_sample"] == 4096
    assert candidate["stop_sample"] == 7168
    assert candidate["start_seconds"] == 100
    assert candidate["stop_seconds"] == 112
    assert candidate["action"] == "review"
    assert candidate["evidence"]["amplitude_z"] > 5
    assert result["automatic_exclusions"] == []
    np.testing.assert_array_equal(data, original)
    json.dumps(result, allow_nan=False)


def test_hf_noise_and_low_correlation_locate_the_original_channel_and_time():
    data, ids = _data()
    rng = np.random.default_rng(4)
    data[-1, 3 * 256:11 * 256] = rng.normal(size=8 * 256) * 25e-6
    result = compute_bad_channels(data, 256, ids, groups=["reference"] * len(ids))
    channel = result["channels"][-1]
    assert "high_frequency_noise" in channel["flags"]
    assert "low_correlation" in channel["flags"]
    candidates = [x for x in result["candidates"] if x["kind"] == "low_correlation" and x["channel_id"] == ids[-1]]
    assert any(item["start_seconds"] <= 4 and item["stop_seconds"] >= 7 for item in candidates)
    assert all(item["window_indices"] for item in candidates)


def test_different_references_and_low_channel_counts_defer_group_candidates():
    data, ids = _data(channels=4)
    data[0] *= 100
    result = compute_bad_channels(data, 256, ids, groups=["M1", "M1", "M1", "M2"])
    assert [group["status"] for group in result["groups"]] == ["small_comparison_group"] * 2
    assert result["candidates"] == []
    assert result["channels"][0]["robust_amplitude_uv"] > result["channels"][1]["robust_amplitude_uv"]


def test_reference_must_be_explicit_for_threshold_candidates():
    data, ids = _data()
    data[0] *= 100
    result = compute_bad_channels(data, 256, ids)
    assert result["groups"][0]["status"] == "reference_group_unconfirmed"
    assert result["groups"][0]["reference_group_declared"] is False
    assert result["groups"][0]["grouping_basis"] == "unreported"
    assert result["candidates"] == []
    assert result["channels"][0]["amplitude_z"] > 5


def test_explicit_group_is_a_caller_declaration_not_acquisition_proof():
    data, ids = _data()
    result = compute_bad_channels(data, 256, ids, groups=["dataset_acquisition"] * len(ids))
    group = result["groups"][0]
    assert group["reference_group_declared"] is True
    assert group["grouping_basis"] == "caller_declared"
    assert "reference_group_confirmed" not in group


def test_common_slow_wave_amplitude_gain_is_not_a_bad_channel_standard():
    data, ids = _data()
    t = np.arange(data.shape[1]) / 256
    data += 100e-6 * np.sin(2 * np.pi * 1.5 * t)
    normal = compute_bad_channels(data, 256, ids, groups=["reference"] * len(ids))
    amplified = compute_bad_channels(data * 10, 256, ids, groups=["reference"] * len(ids))
    assert not any("amplitude_deviation" in ch["flags"] for ch in normal["channels"])
    np.testing.assert_allclose([ch["amplitude_z"] for ch in normal["channels"]], [ch["amplitude_z"] for ch in amplified["channels"]], atol=1e-10)


def test_common_reference_noise_can_be_highly_correlated_and_requires_separate_review():
    data, ids = _data()
    shared_noise = np.random.default_rng(3).normal(size=data.shape[1]) * 100e-6
    result = compute_bad_channels(data + shared_noise, 256, ids, groups=["reference"] * len(ids))
    assert min(np.asarray(result["window_metrics"]["max_abs_correlation"]).ravel()) > 0.9
    assert not any("low_correlation" in ch["flags"] for ch in result["channels"])
    assert "quality_pass" not in result
    assert result["automatic_exclusions"] == []


def test_identical_channels_have_unavailable_cohort_z_instead_of_all_rejected():
    data, ids = _data()
    identical = np.repeat(data[:1], len(ids), axis=0)
    result = compute_bad_channels(identical, 256, ids, groups=["reference"] * len(ids))
    assert all(ch["amplitude_z"] is None for ch in result["channels"])
    assert all(ch["hf_z"] is None for ch in result["channels"])
    assert result["candidates"] == []
    json.dumps(result, allow_nan=False)


def test_nonfinite_and_flat_inputs_are_localized_before_filtering():
    data, ids = _data()
    data[0, 512:514] = np.nan
    data[1, 1024] = np.inf
    data[2] = 2e-6
    result = compute_bad_channels(data, 256, ids, groups=["reference"] * len(ids))
    bad = [item for item in result["candidates"] if item["kind"] == "nonfinite_input"]
    assert [(item["channel_id"], item["start_sample"], item["stop_sample"]) for item in bad] == [(ids[0], 512, 514), (ids[1], 1024, 1025)]
    assert result["channels"][2]["flags"] == ["flat_signal"]
    assert all(ch["robust_amplitude_uv"] is not None for ch in result["channels"][3:])
    json.dumps(result, allow_nan=False)


def test_low_sample_rate_records_hf_availability():
    data, ids = _data(sfreq=100)
    result = compute_bad_channels(data, 100, ids, groups=["reference"] * len(ids))
    assert result["preprocessing"]["hf_status"] == "sample_rate_at_or_below_100hz"
    assert all(ch["hf_z"] is None for ch in result["channels"])
    assert not any("high_frequency_noise" in ch["flags"] for ch in result["channels"])


def test_short_data_reports_applicability_and_partial_tail():
    data, ids = _data(seconds=2.5)
    result = compute_bad_channels(data, 256, ids, groups=["reference"] * len(ids))
    assert result["groups"][0]["status"] == "duration_shorter_than_analysis_filter"
    assert result["unwindowed_tail"]["start_sample"] == 512
    assert result["unwindowed_tail"]["stop_sample"] == 640
    assert result["candidates"] == []


def test_no_comparable_reference_groups_is_not_a_pass():
    pytest.importorskip("pyprep")
    data, ids = _data(channels=1)
    result = compare_with_pyprep(data, 256, ids, groups=["one"])
    assert result["compared_group_count"] == 0
    assert result["all_compared_measurements_match"] is False


@pytest.mark.parametrize("kwargs", [{"sfreq": 0}, {"start_sample": -1}, {"start_seconds": float("nan")}, {"groups": ["one"]}, {"groups": {"missing": "group"}}])
def test_invalid_contracts_raise(kwargs):
    data, ids = _data()
    params = {"sfreq": 256, **kwargs}
    with pytest.raises(ValueError):
        compute_bad_channels(data, channel_ids=ids, **params)


def test_complex_samples_are_not_silently_reinterpreted_as_real_volts():
    data, ids = _data()
    with pytest.raises(ValueError, match="real numeric"):
        compute_bad_channels(data.astype(complex) + 1j, 256, ids)


@pytest.mark.parametrize("kwargs", [{"min_group_channels": 2}, {"min_group_channels": 4.5}, {"correlation_threshold": 1}, {"deviation_z": float("nan")}, {"window_seconds": True}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        BadChannelConfig(**kwargs)
