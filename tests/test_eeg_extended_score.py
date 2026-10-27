"""Integer policy boundaries and evidence accounting on the original clock."""
import copy

import numpy as np
import pytest

from brainelecqc.eeg.extended_score import extend_integer_scores
from brainelecqc.eeg.score_rules import POLICY


def _inputs(channels=4):
    ids = [f"eeg:{i}" for i in range(channels)]
    result = {"raw_data_volts": np.ones((channels, 10000))*1e-6,
              "references": {"emg": np.ones(10000)*1e-6},
              "features": {"channel_ids": ids, "channel_names": [f"Original {i}" for i in range(channels)],
                           "start_sample": 711, "stop_sample": 10711, "sampling_rate_hz": 100,
                           "start_seconds": 90.25},
              "method_results": {"continuity": {"candidates": []}, "muscle": {
                  "status": "ok", "candidates": [], "reference": {"status": "ok", "valid_seconds": 100},
                  "channels": [{"channel_id": cid, "valid_seconds": 100} for cid in ids]}}}
    review = {"rows": [{"dimension_id": key} for key in POLICY["max_points"]]}
    outcomes = {key: {"score": value, "reason": "测量完整", "components": []}
                for key, value in POLICY["max_points"].items()}
    outcomes["channels"]["metrics"] = {"worst_channel_id": ids[0],
        "channel_fractions": dict.fromkeys(ids, 0), "convergent_fractions": dict.fromkeys(ids, 0)}
    for kind in ("ecg", "eog"):
        outcomes[kind] = {"score": None, "reason": "缺少辅助导联", "metrics": {}, "evidence": []}
    return result, review, outcomes


def _candidate(a, b, *, channel=0, kind="near_zero_flatline"):
    return {"channel_id": f"eeg:{channel}", "kind": kind, "start_sample": 711+a, "stop_sample": 711+b}


@pytest.mark.parametrize("dimension,kind", [("continuity", "near_zero_flatline"), ("muscle", "emg_supported_burst")])
@pytest.mark.parametrize("samples,expected", [(0, 10), (1, 9), (130, 9), (199, 9), (200, 8),
    (499, 8), (500, 7), (999, 7), (1000, 5), (2499, 5), (2500, 3), (4999, 3), (5000, 0)])
def test_integer_duration_boundaries(dimension, kind, samples, expected):
    result, review, outcomes = _inputs()
    if samples:
        result["method_results"][dimension]["candidates"] = [_candidate(0, samples, kind=kind)]
    version, policy = extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes[dimension]["score"] == expected
    assert type(outcomes[dimension]["score"]) is int
    assert sum(policy["max_points"].values()) == 100
    assert version == "eeg-integer-score-7.0"
    assert POLICY["max_points"]["integrity"] == 20  # Other runs retain their saved policy.


def test_overlapping_duplicate_and_multiple_kinds_count_union_once():
    result, review, outcomes = _inputs()
    result["method_results"]["continuity"]["candidates"] = [
        _candidate(0, 100), _candidate(50, 150, kind="calibration_endpoint_plateau"),
        _candidate(0, 100, kind="constant_value")]
    extend_integer_scores(result, review, outcomes, POLICY)
    outcome = outcomes["continuity"]
    assert outcome["score"] == 9
    assert outcome["metrics"]["channel_fractions"]["eeg:0"] == .015
    first = outcome["evidence"][0]
    assert first["raw_name"] == "Original 0"
    assert first["start_seconds"] == 90.25
    assert first["stop_seconds"] == 91.25
    assert outcome["metrics"]["time_burden"]["affected_seconds"] == 1.5


@pytest.mark.parametrize("channels,expected", [(8, 9), (9, 9)])
def test_same_time_on_multiple_channels_is_counted_once(channels, expected):
    result, review, outcomes = _inputs(channels)
    result["method_results"]["continuity"]["candidates"] = [_candidate(0, 100, channel=i) for i in (0, 1)]
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["continuity"]["score"] == expected
    assert outcomes["continuity"]["metrics"]["time_burden"]["fraction"] == .01


def test_different_channel_intervals_contribute_their_time_union():
    result, review, outcomes = _inputs(64)
    result["method_results"]["continuity"]["candidates"] = [
        _candidate(0, 600), _candidate(500, 1100, channel=1)]
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["continuity"]["metrics"]["time_burden"]["fraction"] == .11
    assert outcomes["continuity"]["score"] == 5


@pytest.mark.parametrize("coverage,expected", [(94.99, None), (95, 10)])
@pytest.mark.parametrize("source", ["eeg", "emg"])
def test_valid_coverage_gate_includes_reference_and_boundary(source, coverage, expected):
    result, review, outcomes = _inputs()
    target = result["method_results"]["muscle"]["channels"][0] if source == "eeg" else result["method_results"]["muscle"]["reference"]
    target["valid_seconds"] = coverage
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["muscle"]["score"] == expected


def test_missing_emg_is_pending_and_general_high_frequency_is_not_emg_score():
    result, review, outcomes = _inputs()
    result["method_results"]["muscle"]["candidates"] = [_candidate(0, 5000, kind="high_frequency_burst")]
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["muscle"]["score"] == 10
    result, review, outcomes = _inputs()
    result["references"].pop("emg")
    result["method_results"]["muscle"]["reference"]["status"] = "missing_reference"
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["muscle"]["score"] is None
    assert next(r for r in review["rows"] if r["dimension_id"] == "muscle")["human_grade"] is None


def test_continuity_cannot_receive_full_points_for_unobserved_samples():
    result, review, outcomes = _inputs()
    result["raw_data_volts"][0, :501] = np.nan
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["continuity"]["score"] is None


@pytest.mark.parametrize("affected,expected", [(1, 19), (2, 18), (6, 16), (7, 12), (16, 6), (32, 0), (64, 0)])
def test_persistent_bad_channel_scope_has_distinct_bands(affected, expected):
    result, review, outcomes = _inputs(64)
    for i in range(affected):
        outcomes["channels"]["metrics"]["channel_fractions"][f"eeg:{i}"] = .9
        outcomes["channels"]["metrics"]["convergent_fractions"][f"eeg:{i}"] = .2
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == expected
    assert sum(c["deduction"] for c in outcomes["channels"]["components"]) == 20-expected
    assert len(outcomes["channels"]["metrics"]["automatic_bad_channel_ids"]) == affected


@pytest.mark.parametrize("fraction,expected", [(0, 20), (.001, 20), (.099, 20), (.1, 19),
                                              (.1001, 19), (.5, 19), (.5001, 19), (1, 19)])
def test_one_channel_duration_and_persistence_boundaries(fraction, expected):
    result, review, outcomes = _inputs(64)
    outcomes["channels"]["metrics"]["channel_fractions"]["eeg:0"] = fraction
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == expected


@pytest.mark.parametrize("count,convergent,expected", [(1, .01, 18), (2, .01, 16), (4, .01, 12),
                                                    (10, .01, 6), (10, .01001, 6), (20, .01, 0)])
def test_channel_prevalence_is_the_scoring_axis(count, convergent, expected):
    result, review, outcomes = _inputs(40)
    for i in range(count):
        outcomes["channels"]["metrics"]["channel_fractions"][f"eeg:{i}"] = 1
        outcomes["channels"]["metrics"]["convergent_fractions"][f"eeg:{i}"] = convergent
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == expected


def test_many_brief_outliers_do_not_count_as_many_persistent_bad_channels():
    result, review, outcomes = _inputs(64)
    outcomes["channels"]["metrics"]["channel_fractions"] = dict.fromkeys(result["features"]["channel_ids"], .01)
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == 20
    assert outcomes["channels"]["metrics"]["automatic_bad_channel_ids"] == []
    assert len(outcomes["channels"]["metrics"]["localized_channel_ids"]) == 64


def test_unlocalized_statistics_preserve_extent_unknown_and_coverage_gate():
    result, review, outcomes = _inputs()
    review["channel_profiles"] = [{"dimension_id": "channels", "channel_id": "eeg:0",
                                   "location_status": "whole_record_statistic_only"}]
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == 19
    assert outcomes["channels"]["metrics"]["automatic_bad_channel_ids"] == []
    assert outcomes["channels"]["metrics"]["unlocalized_channel_ids"] == ["eeg:0"]
    result, review, outcomes = _inputs()
    outcomes["channels"] = {"score": None, "reason": "覆盖不足", "metrics": {}}
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"] == {"score": None, "reason": "覆盖不足", "metrics": {}}


def test_new_and_saved_legacy_rulebooks_keep_their_own_integer_bands():
    from brainelecqc.eeg.score_rules import rubric_rows
    old = dict(rubric_rows("eeg-integer-score-4.0"))["通道质量 · 20"]
    new = dict(rubric_rows("eeg-integer-score-5.0"))["通道质量 · 20"]
    assert "扣 15" in old and "单一坏道最多扣 10" in new
    assert "20/18/16/12/8/4/0" in dict(rubric_rows("eeg-integer-score-6.0"))["通道质量 · 20"]


def test_one_bad_channel_in_a_small_group_cannot_exhaust_the_item():
    result, review, outcomes = _inputs(4)
    outcomes["channels"]["metrics"]["channel_fractions"]["eeg:0"] = 1
    extend_integer_scores(result, review, outcomes, POLICY)
    assert outcomes["channels"]["score"] == 10


def test_legacy_result_preserves_its_rubric_and_input_objects():
    result, review, outcomes = _inputs()
    result["method_results"].pop("muscle")
    before = copy.deepcopy((review, outcomes, POLICY))
    version, policy = extend_integer_scores(result, review, outcomes, POLICY)
    assert version is None and policy == POLICY
    assert (review, outcomes, POLICY) == before
