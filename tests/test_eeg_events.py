"""Event clock and reference-engine contracts; generated signals test arithmetic."""

import json

import numpy as np
import pytest

from brainelecqc.eeg.events import EEGEventConfig, compare_events_to_mne, compute_event_qc


def signals(seconds=30, sfreq=128):
    time = np.arange(seconds * sfreq) / sfreq
    ecg = np.zeros(len(time))
    eog = np.zeros(len(time))
    for point in np.arange(0.8, seconds - 0.5, 0.9):
        ecg += 250e-6 * np.exp(-((time - point) / 0.02) ** 2)
    for point in (0.15, 3, 8, 14, 22, seconds - 0.15):
        eog += 180e-6 * np.exp(-((time - point) / 0.15) ** 2)
    data = np.array([10e-6 * np.sin(2 * np.pi * 10 * time) + ecg * 0.1,
                     15e-6 * np.cos(2 * np.pi * 8 * time) + eog * 0.2])
    return data, {"ecg": ecg, "eog": eog}, sfreq


def test_public_mne_events_and_epoch_statistics_match_with_independent_origins():
    data, references, sfreq = signals()
    original = data.copy()
    result = compute_event_qc(data, sfreq, ["ch-A", "ch-B"], ["same", "same"],
                              references=references, start_sample=8712, start_seconds=90.25,
                              reference_channels={"eog": {"channel_id": "aux-7", "raw_name": "EOG E1-M2"}})
    check = compare_events_to_mne(data, references, result)
    assert np.array_equal(data, original)
    assert result["reference_engine"]["role"] == "public detector implementation reuse"
    assert result["epoch_processing"]["baseline"] is None
    assert result["epoch_processing"]["eeg_filter"] is None
    assert result["by_kind"]["eog"]["source_reference"]["channel_id"] == "aux-7"
    for kind in ("ecg", "eog"):
        assert check["by_kind"][kind]["event_indices_equal"]
        assert check["by_kind"][kind]["mean_max_abs_error_volts"] < 1e-18
        assert check["by_kind"][kind]["median_max_abs_error_volts"] < 1e-18
        assert check["by_kind"][kind]["sem_max_abs_error_volts"] < 1e-18
        assert check["by_kind"][kind]["relative_time_max_abs_error_seconds"] < 1e-14
        for event in result["by_kind"][kind]["event_table"]:
            assert event["sample"] == 8712 + event["sample_offset"]
            assert event["time_seconds"] == 90.25 + event["sample_offset"] / sfreq
    json.dumps(result, allow_nan=False)


def test_boundary_epochs_are_listed_and_data_are_preserved():
    data, references, sfreq = signals()
    result = compute_event_qc(data, sfreq, ["a", "b"], references=references)
    table = result["by_kind"]["eog"]["event_table"]
    boundary = [event for event in table if event["status"] == "boundary"]
    assert boundary
    assert result["by_kind"]["eog"]["boundary_excluded_count"] == len(boundary)
    assert all(not row["included_in_average"] for row in boundary)
    assert all(0 <= row["epoch_start_sample"] < row["epoch_stop_sample"] <= data.shape[1] for row in table)
    assert "automatic_exclusions" not in result


def test_eog_effective_threshold_and_event_time_counts_match_reference():
    import mne

    data, refs, sfreq = signals()
    result = compute_event_qc(data, sfreq, ["a", "b"], references=refs,
                              config={"event_count_window_seconds": 5.0})
    filtered = mne.filter.filter_data(refs["eog"], sfreq, 1, 10, filter_length="10s",
                                     l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                                     phase="zero-double", fir_window="hann", fir_design="firwin2",
                                     verbose=False)
    eog = result["by_kind"]["eog"]
    assert eog["detector"]["effective_threshold_volts"] == np.ptp(filtered) / 4
    assert sum(window["event_count"] for window in eog["event_count_windows"]) == eog["detected_event_count"]
    assert eog["event_time_span_seconds"] == [eog["event_table"][0]["time_seconds"], eog["event_table"][-1]["time_seconds"]]


def test_shared_reference_signal_produces_review_evidence_without_artifact_label():
    data, refs, sfreq = signals()
    data[0] = refs["eog"]
    result = compute_event_qc(data, sfreq, ["shared-reference", "b"], references=refs)
    eeg = result["by_kind"]["eog"]["channel_summaries"][0]
    reference = result["by_kind"]["eog"]["reference_summary"]
    assert eeg["mean_volts"] == reference["mean_volts"]
    assert result["by_kind"]["eog"]["review_state"] == "pending"
    assert "artifact_detected" not in eeg


@pytest.mark.parametrize("duration,reference,status", [
    (2, "varying", "insufficient_duration"),
    (30, "constant", "constant_reference"),
    (30, "missing", "missing_reference"),
])
def test_short_constant_and_missing_reference_states(duration, reference, status):
    data = np.zeros((1, duration * 128))
    refs = {} if reference == "missing" else {"ecg": np.zeros(data.shape[1]) if reference == "constant"
                                             else np.sin(np.arange(data.shape[1])) * 1e-5}
    result = compute_event_qc(data, 128, ["a"], references=refs)
    assert result["by_kind"]["ecg"]["status"] == status
    assert result["by_kind"]["ecg"]["averaged_event_count"] == 0
    json.dumps(result, allow_nan=False)


def test_auxiliary_sampling_rate_is_explicitly_checked():
    data = np.zeros((1, 30 * 64))
    result = compute_event_qc(data, 64, ["a"], references={"ecg": np.sin(np.arange(data.shape[1])) * 1e-5})
    assert result["by_kind"]["ecg"]["status"] == "insufficient_sampling_rate"


def test_deterministic_memory_selection_retains_all_event_locations():
    data, refs, sfreq = signals()
    config = EEGEventConfig(max_events_for_average=3, max_epoch_elements_per_channel=3 * 129)
    result = compute_event_qc(data, sfreq, ["a", "b"], references=refs, config=config)
    ecg = result["by_kind"]["ecg"]
    assert ecg["detected_event_count"] > 3
    assert ecg["averaged_event_count"] == 3
    assert len(ecg["event_table"]) == ecg["detected_event_count"]
    assert any(event["status"] == "event_limit" for event in ecg["event_table"])
    assert ecg["channel_summaries"][0]["chronological_half_correlation"] is None
    assert ecg["channel_summaries"][0]["odd_even_correlation"] is None
    check = compare_events_to_mne(data, refs, result)
    assert check["by_kind"]["ecg"]["mean_max_abs_error_volts"] < 1e-18


def test_one_event_is_described_without_sem_or_contamination_classification():
    data, refs, sfreq = signals()
    result = compute_event_qc(data, sfreq, ["a", "b"], references=refs,
                              config={"max_events_for_average": 1})
    for kind in ("ecg", "eog"):
        block = result["by_kind"][kind]
        assert block["averaged_event_count"] == 1
        assert block["channel_summaries"][0]["sem_volts"] is None
        assert block["channel_summaries"][0]["chronological_half_correlation"] is None
        assert block["review_state"] == "pending"
        assert "artifact_detected" not in block


def test_reference_fingerprint_is_checked_during_comparison():
    data, refs, sfreq = signals()
    result = compute_event_qc(data, sfreq, ["a", "b"], references=refs)
    with pytest.raises(ValueError, match="reference fingerprint"):
        compare_events_to_mne(data, {**refs, "ecg": refs["ecg"] + 1e-6}, result)
    with pytest.raises(ValueError, match="data fingerprint"):
        compare_events_to_mne(data + 1e-6, refs, result)


@pytest.mark.parametrize("config", [
    {"ecg_qrs_threshold": 0}, {"ecg_qrs_threshold": "guess"}, {"eog_threshold_volts": -1},
    {"epoch_tmin_seconds": 0.5}, {"max_events_for_average": True}, {"min_duration_seconds": 2},
    {"filter_length_seconds": float("nan")}, {"eog_h_freq_hz": 0.5},
])
def test_invalid_configs_fail(config):
    with pytest.raises(ValueError):
        EEGEventConfig(**config)


def test_invalid_input_alignment_and_ids_fail():
    data, refs, sfreq = signals()
    with pytest.raises(ValueError, match="sample count"):
        compute_event_qc(data, sfreq, ["a", "b"], references={"ecg": refs["ecg"][:-1]})
    with pytest.raises(ValueError, match="distinct"):
        compute_event_qc(data, sfreq, ["a", "a"])
    with pytest.raises(ValueError, match="start_sample"):
        compute_event_qc(data, sfreq, ["a", "b"], start_sample=-1)
    with pytest.raises(ValueError, match="epoch exceeds"):
        compute_event_qc(data, sfreq, ["a", "b"], config={"max_epoch_elements_per_channel": 10})
