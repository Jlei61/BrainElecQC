"""Known event-locked waveforms exercise the project integer review bins."""
import copy
import json

import numpy as np
import pytest

from brainelecqc.eeg.event_score import score_event_coupling


def _result(amplitude=0.7, n_events=40, n_channels=1, origin=1000, start_seconds=50.25):
    width, hop, sfreq = 9, 12, 10.0
    data = np.zeros((n_channels, n_events*hop+4))
    events = []
    background = np.zeros(width)
    background[:2] = [1, -1]
    locked = np.zeros(width)
    locked[4] = amplitude
    for i in range(n_events):
        a = 2+i*hop
        data[0, a:a+width] = (locked + (-1 if i % 4 < 2 else 1)*background)*1e-6
        for ci in range(1, n_channels):
            data[ci, a:a+width] = (-1 if (i+ci) % 4 < 2 else 1)*background*1e-6
        sample, begin, end = origin+a+4, origin+a, origin+a+width
        events.append({"event_id": f"ecg:{sample}", "sample": sample,
                       "epoch_start_sample": begin, "epoch_stop_sample": end,
                       "time_seconds": start_seconds+(sample-origin)/sfreq,
                       "epoch_start_seconds": start_seconds+(begin-origin)/sfreq,
                       "epoch_stop_seconds": start_seconds+(end-origin)/sfreq,
                       "included_in_average": True, "complete_epoch": True})
    ids, names = [f"channel-{i}" for i in range(n_channels)], [f"EEG{i}" for i in range(n_channels)]
    f = {"channel_ids": ids, "channel_names": names, "start_sample": origin,
         "stop_sample": origin+data.shape[1], "start_seconds": start_seconds, "sampling_rate_hz": sfreq}
    block = {"source_reference": {"channel_id": "aux-ecg", "raw_name": "ECG"}, "status": "computed",
             "event_table": events, "selected_event_ids": [e["event_id"] for e in events]}
    return {"features": f, "raw_data_volts": data,
            "method_results": {"events": {"by_kind": {"ecg": block, "eog": copy.deepcopy(block)}}}}


@pytest.mark.parametrize("amplitude,expected_ratio,points", [(0.1, 0.05, 10), (0.3, 0.15, 8), (0.7, 0.35, 5), (1.2, 1.2/2.2, 2)])
@pytest.mark.parametrize("kind", ["ecg", "eog"])
def test_known_locked_amplitude_bins_and_raw_samples_are_preserved(amplitude, expected_ratio, points, kind):
    result = _result(amplitude)
    original = result["raw_data_volts"].copy()
    score = score_event_coupling(result, kind, {})
    assert score["score"] == points and isinstance(score["score"], int)
    channel = score["metrics"]["channels"][0]
    assert channel["ratio"] == pytest.approx(expected_ratio)
    assert channel["repeatability_pass"]
    assert channel["chronological_half_correlation"] == pytest.approx(1)
    assert channel["odd_even_correlation"] == pytest.approx(1)
    assert np.array_equal(result["raw_data_volts"], original)
    json.dumps(score, allow_nan=False)


def test_one_strong_outlier_cannot_create_a_locked_median_artifact():
    result = _result(0)
    result["raw_data_volts"][0, 2+4] = 1
    score = score_event_coupling(result, "ecg")
    assert score["score"] == 10
    assert score["metrics"]["channels"][0]["ratio"] == 0


def test_random_event_epochs_do_not_pass_repeated_shape_gate():
    result = _result()
    rng = np.random.default_rng(615)
    result["raw_data_volts"][:] = rng.normal(0, 1e-6, result["raw_data_volts"].shape)
    score = score_event_coupling(result, "ecg")
    assert score["score"] == 10
    assert not score["metrics"]["channels"][0]["repeatability_pass"]


@pytest.mark.parametrize("failed_split", ["chronological", "odd_even"])
def test_both_independent_split_checks_are_required(failed_split):
    result = _result(0.7)
    for i in range(40):
        a = 2+i*12
        sign = (-1 if i < 20 else 1) if failed_split == "chronological" else (-1 if i % 2 else 1)
        result["raw_data_volts"][0, a:a+2] = np.array([sign, -sign])*1e-6
    score = score_event_coupling(result, "ecg")
    channel = score["metrics"]["channels"][0]
    assert channel["ratio"] == pytest.approx(0.35)
    assert score["score"] == 10 and not channel["repeatability_pass"]
    correlations = [channel["chronological_half_correlation"], channel["odd_even_correlation"]]
    assert min(correlations) < 0.5 < max(correlations)


def test_ratio_at_a_configured_boundary_uses_the_next_deduction_bin():
    result = _result(0.7)
    ratio = score_event_coupling(result, "ecg")["metrics"]["channels"][0]["ratio"]
    assert score_event_coupling(result, "ecg", {"ratio_edges": [0.1, ratio, 0.5]})["score"] == 5


def test_healthy_channels_do_not_dilute_the_worst_reliable_channel():
    alone = score_event_coupling(_result(1.2), "ecg")
    many = score_event_coupling(_result(1.2, n_channels=64), "ecg")
    assert alone["score"] == many["score"] == 2
    assert alone["evidence"] == many["evidence"]
    assert many["metrics"]["worst_channel_id"] == "channel-0"


def test_original_clock_is_preserved_for_representative_event():
    result = _result(origin=90001, start_seconds=4.125)
    score = score_event_coupling(result, "ecg")
    event = score["evidence"][0]
    assert event["time_seconds"] == 4.125+(event["sample"]-90001)/10
    assert event["start_seconds"] == 4.125+(event["start_sample"]-90001)/10
    assert event["stop_seconds"] == 4.125+(event["stop_sample"]-90001)/10
    assert event["event_id"] in score["metrics"]["event_ids"]


@pytest.mark.parametrize("status", ["missing_reference", "constant_reference", "insufficient_duration", "insufficient_sampling_rate", "detector_failed"])
def test_unavailable_reference_or_unrun_detection_is_unscored(status):
    result = _result()
    result["method_results"]["events"]["by_kind"]["ecg"]["status"] = status
    score = score_event_coupling(result, "ecg")
    assert score["score"] is None and score["max_points"] == 10 and score["reason"]
    assert score["metrics"]["channels"] == []


def test_missing_source_reference_is_not_a_zero_score():
    result = _result()
    result["method_results"]["events"]["by_kind"]["ecg"]["source_reference"] = None
    assert score_event_coupling(result, "ecg")["score"] is None


def test_complete_selected_events_are_sorted_and_overlap_removed_before_scoring():
    result = _result()
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    original_ids = list(block["selected_event_ids"])
    extra = copy.deepcopy(block["event_table"][0])
    extra["event_id"] = "ecg:overlap"
    for key in ("sample", "epoch_start_sample", "epoch_stop_sample"):
        extra[key] += 1
    for key in ("time_seconds", "epoch_start_seconds", "epoch_stop_seconds"):
        extra[key] += 0.1
    block["event_table"].insert(0, extra)
    block["event_table"].reverse()
    block["selected_event_ids"].append(extra["event_id"])
    score = score_event_coupling(result, "ecg")
    assert score["score"] == 5
    assert score["metrics"]["event_ids"] == original_ids
    assert score["metrics"]["overlap_excluded_count"] == 1
    assert score["metrics"]["overlap_excluded_event_ids"] == [extra["event_id"]]


def test_overlap_removal_can_make_event_evidence_insufficient():
    result = _result(n_events=20)
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    block["event_table"][1] = copy.deepcopy(block["event_table"][0])
    block["event_table"][1]["event_id"] = "ecg:duplicate-window"
    block["selected_event_ids"] = [e["event_id"] for e in block["event_table"]]
    score = score_event_coupling(result, "ecg")
    assert score["score"] is None
    assert score["metrics"]["nonoverlapping_event_count"] == 19
    assert "19" in score["reason"]


def test_incomplete_and_unselected_events_are_excluded():
    result = _result()
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    block["event_table"][0]["complete_epoch"] = False
    block["event_table"][1]["included_in_average"] = False
    block["selected_event_ids"].remove(block["event_table"][2]["event_id"])
    score = score_event_coupling(result, "ecg")
    assert score["metrics"]["nonoverlapping_event_count"] == 37
    assert score["metrics"]["noncomplete_excluded_count"] == 1


@pytest.mark.parametrize("status", ["no_events", "no_complete_epochs"])
def test_executed_detector_with_no_usable_events_remains_insufficient(status):
    result = _result(n_events=0)
    result["method_results"]["events"]["by_kind"]["ecg"]["status"] = status
    score = score_event_coupling(result, "ecg")
    assert score["score"] is None and "0 个" in score["reason"]


@pytest.mark.parametrize("config", [{"min_events": True}, {"min_events": 3}, {"min_events": 20.5},
    {"repeatability_min": -0.1}, {"repeatability_min": float("nan")}, {"repeatability_min": 1.1},
    {"ratio_edges": [0.1, 0.1, 0.5]}, {"ratio_edges": [0.1, float("inf"), 0.5]},
    {"ratio_edges": None}, {"ratio_edges": [0.1, 0.2]}, {"points": [10, 8.0, 5, 2]},
    {"points": [10, True, 5, 2]}, {"points": [10, 5, 8, 2]}, {"points": [9, 8, 5, 2]},
    {"points": [10, 8, 5, -1]}, {"unknown": 1}, []])
def test_invalid_configuration_fails(config):
    with pytest.raises(ValueError):
        score_event_coupling(_result(), "ecg", config)


def test_configured_edges_and_integer_points_are_used():
    score = score_event_coupling(_result(0.7), "ecg", {"ratio_edges": [0.2, 0.4, 0.6], "points": [10, 7, 4, 1]})
    assert score["score"] == 7


@pytest.mark.parametrize("failure", ["outside", "time", "offsets", "duplicate_id", "unknown_id", "method_origin"])
def test_malformed_event_clock_and_identity_fail(failure):
    result = _result()
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    if failure == "outside":
        block["event_table"][0]["epoch_start_sample"] = 0
    elif failure == "time":
        block["event_table"][0]["time_seconds"] += 1
    elif failure == "offsets":
        block["event_table"][0]["epoch_start_sample"] += 1
        block["event_table"][0]["epoch_start_seconds"] += 0.1
    elif failure == "duplicate_id":
        block["event_table"][0]["event_id"] = block["event_table"][1]["event_id"]
    elif failure == "unknown_id":
        block["selected_event_ids"].append("missing")
    else:
        result["method_results"]["events"]["start_sample"] = 0
    with pytest.raises(ValueError):
        score_event_coupling(result, "ecg")


@pytest.mark.parametrize("value", [0, float("nan")])
def test_unavailable_eeg_amplitude_does_not_receive_full_points(value):
    result = _result()
    result["raw_data_volts"][:] = value
    score = score_event_coupling(result, "ecg")
    assert score["score"] is None
    assert score["metrics"]["unassessed_channel_ids"] == ["channel-0"]
