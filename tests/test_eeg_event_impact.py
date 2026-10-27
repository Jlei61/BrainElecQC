"""Independent event windows, counterexamples and exact original-clock unions."""
import json
import numpy as np
import pytest

from brainelecqc.eeg.event_impact import score_event_impact
from brainelecqc.eeg.event_score import score_event_coupling
from brainelecqc.eeg.impact import time_burden, time_points
from tests.test_eeg_event_score import _result


def _score(result, kind="ecg"):
    return score_event_impact(result, kind, score_event_coupling(result, kind))


@pytest.mark.parametrize("kind", ["ecg", "eog"])
def test_supported_windows_use_all_events_original_clock_and_time_union(kind):
    result = _result(amplitude=4, n_events=40, origin=90001, start_seconds=4.125)
    before = result["raw_data_volts"].copy()
    score = _score(result, kind)
    assert score["metrics"]["impact_affected_event_count"] == 40
    assert score["metrics"]["time_burden"]["affected_samples"] == 40*9
    assert score["metrics"]["time_burden"]["fraction"] == pytest.approx(360/484)
    assert score["score"] == time_points(360/484) == 0
    assert score["evidence"][0]["start_seconds"] == 4.125+.2
    assert np.array_equal(before, result["raw_data_volts"])
    json.dumps(score, allow_nan=False)


def test_a_real_auxiliary_event_without_eeg_coupling_does_not_count_as_bad_time():
    result = _result(amplitude=0)
    score = _score(result)
    assert score["score"] == 10
    assert score["metrics"]["time_burden"]["fraction"] == 0
    assert score["metrics"]["impact_complete_event_count"] == 40


def test_random_eeg_and_one_large_outlier_do_not_create_event_time_burden():
    result = _result(amplitude=0)
    result["raw_data_volts"][0, 6] = 1
    assert _score(result)["metrics"]["time_burden"]["fraction"] == 0
    result["raw_data_volts"][:] = np.random.default_rng(615).normal(0, 1e-6, result["raw_data_volts"].shape)
    assert _score(result)["metrics"]["time_burden"]["fraction"] == 0


def test_nonmatching_individual_events_are_excluded_even_when_global_coupling_passes():
    result = _result(amplitude=4)
    for index in (3, 11, 23, 31):
        a = 2+index*12
        result["raw_data_volts"][0, a+4] = 0
    score = _score(result)
    assert score["metrics"]["impact_affected_event_count"] == 36
    assert score["metrics"]["time_burden"]["affected_samples"] == 36*9
    assert len(score["metrics"]["impact_observations"]) == 40


def test_each_window_is_held_out_from_its_template_even_when_averaging_is_subsampled():
    result = _result(amplitude=4, n_events=80)
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    block["selected_event_ids"] = [e["event_id"] for e in block["event_table"][::2]]
    score = _score(result)
    excluded = score["metrics"]["impact_boundary_excluded_event_ids"]
    assert score["metrics"]["impact_affected_event_count"] == 80-len(excluded)
    assert len(excluded) == 1  # This event straddles the independent template split.
    assert score["metrics"]["impact_unassessed_time"]["affected_samples"] == 9
    by_id = {e["event_id"]: e for e in block["event_table"]}
    for observation in score["metrics"]["impact_observations"]:
        training = score["metrics"]["impact_training_event_ids"][observation["template_half"]]
        assert observation["event_id"] not in training
        assert all(observation["stop_sample"] <= by_id[e]["epoch_start_sample"] or
                   observation["start_sample"] >= by_id[e]["epoch_stop_sample"] for e in training)


def test_duplicate_channels_do_not_multiply_affected_time():
    result = _result(amplitude=4, n_channels=2)
    result["raw_data_volts"][1] = result["raw_data_volts"][0]
    score = _score(result)
    assert len(score["evidence"]) == 80
    assert score["metrics"]["time_burden"]["affected_samples"] == 360


def test_no_reference_remains_unassessed_and_has_no_zero_time_claim():
    result = _result()
    result["method_results"]["events"]["by_kind"]["ecg"]["source_reference"] = None
    score = _score(result)
    assert score["score"] is None and "time_burden" not in score["metrics"]


def test_unselected_event_clock_is_validated_before_localization():
    result = _result(amplitude=4, n_events=80)
    block = result["method_results"]["events"]["by_kind"]["ecg"]
    block["selected_event_ids"] = [e["event_id"] for e in block["event_table"][::2]]
    block["event_table"][1]["time_seconds"] += .1
    with pytest.raises(ValueError, match="time"):
        _score(result)


def test_interval_union_is_not_a_channel_time_average():
    f = _result()["features"]
    origin = f["start_sample"]
    burden = time_burden(f, [(origin, origin+50), (origin+25, origin+75), (origin, origin+50)])
    assert burden["affected_samples"] == 75
    assert burden["intervals"] == [[origin, origin+75]]
