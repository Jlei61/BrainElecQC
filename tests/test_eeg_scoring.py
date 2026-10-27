"""Integer initial scores preserve evidence extents and human adjustment fields."""
import copy
import csv
import json

import numpy as np
import pytest

from brainelecqc.eeg.scoring import compute_quality_score
from brainelecqc.eeg.score_report import render_score_review, write_score_files


def _result(n_channels=2):
    ids, names = [f"c{i}" for i in range(n_channels)], [f"Ch{i}" for i in range(n_channels)]
    window = {"start_sample": 100, "stop_sample": 110, "start_seconds": 50, "stop_seconds": 55}
    event = {"source_reference": None, "detected_event_count": 0, "selected_event_ids": [], "status": "missing_reference", "event_table": []}
    lines = [{**window, "channel_id": cid, "nominal_hz": hz, "status": "computed", "flagged": False}
             for cid in ids for hz in (50, 60)]
    return {"raw_data_volts": np.ones((n_channels, 10))*1e-6,
        "features": {**window, "sampling_rate_hz": 2, "channel_ids": ids, "channel_names": names, "data_sha256": "test"},
        "decisions": {"review_queue": []},
        "method_results": {"bad_channels": {
            "parameters": {"hf_noise_z": 5}, "groups": [{"thresholds_active": True, "channel_ids": ids}],
            "channels": [{"channel_id": cid} for cid in ids], "windows": [window], "candidates": [],
            "window_metrics": {key: [[1] for _ in ids] for key in ("amplitude_z", "hf_z", "max_abs_correlation")}},
            "spectral": {"line_noise": {"observations": lines}, "parameterization": {
                "full": [{"alpha_peaks": [], "status": "computed"} for _ in ids]}},
            "events": {"by_kind": {"ecg": copy.deepcopy(event), "eog": copy.deepcopy(event)}}}}


def _candidate(a, b, kind="low_correlation"):
    return {"channel_id": "c0", "kind": kind, "start_sample": a, "stop_sample": b,
            "start_seconds": 50+(a-100)/2, "stop_seconds": 50+(b-100)/2}


def _row(review, key):
    return next(row for row in review["rows"] if row["dimension_id"] == key)


def test_automatic_integer_score_and_human_adjustments_remain_blank():
    review = compute_quality_score(_result())
    assert review["rubric_version"] == "eeg-integer-score-3.0"
    assert "auto_score_100" not in review
    assert review["human_overall_grade"] is None
    assert all(r["human_grade"] is None and r["human_points"] is None for r in review["rows"])
    assert review["auto_total"] == review["assessed_max"] == 70
    assert review["pending_max"] == 30
    assert _row(review, "ecg")["status"] == "辅助导联待补充"


def test_duplicate_overlapping_windows_are_counted_once_with_exact_origin():
    result = _result()
    result["method_results"]["bad_channels"]["candidates"] = [
        _candidate(101, 105), _candidate(103, 107), _candidate(101, 105)]
    review = compute_quality_score(result)
    profile = review["channel_profiles"][0]
    assert profile["covered_seconds"] == profile["longest_continuous_seconds"] == 3
    assert profile["intervals"] == [{"start_sample": 101, "stop_sample": 107, "start_seconds": 50.5, "stop_seconds": 53.5}]


def test_short_scattered_windows_are_not_presented_as_one_continuous_artifact():
    result = _result()
    bad = result["method_results"]["bad_channels"]
    bad["candidates"] = [_candidate(100, 110, "high_frequency_noise")]
    bad["windows"] = [{"start_sample": a, "stop_sample": a+2} for a in range(100, 110, 2)]
    for key in bad["window_metrics"]:
        bad["window_metrics"][key] = [[1]*5, [1]*5]
    bad["window_metrics"]["hf_z"][0] = [6, 1, 6, 1, 6]
    profile = compute_quality_score(result)["channel_profiles"][0]
    assert profile["covered_seconds"] == 3  # More than half the input, but disjoint.
    assert profile["longest_continuous_seconds"] == 1
    assert len(profile["intervals"]) == 3


def test_whole_record_hf_candidate_without_local_hits_remains_visible():
    result = _result()
    result["method_results"]["bad_channels"]["candidates"] = [_candidate(100, 110, "high_frequency_noise")]
    review = compute_quality_score(result)
    profile = review["channel_profiles"][0]
    assert profile["location_status"] == "whole_record_statistic_only"
    assert profile["covered_seconds"] is None and profile["longest_continuous_seconds"] is None
    assert "高频比例离群 1 道" in _row(review, "channels")["finding"]
    assert "定位待核" in render_score_review(result, {}, with_figures=False)


def test_hf_localization_uses_saved_config_and_preserves_raw_measurement():
    result = _result()
    bad = result["method_results"]["bad_channels"]
    bad["parameters"]["hf_noise_z"] = 7
    bad["candidates"] = [_candidate(100, 110, "high_frequency_noise")]
    bad["window_metrics"]["hf_z"][0] = [7]
    assert compute_quality_score(result)["channel_profiles"][0]["covered_seconds"] is None
    bad["window_metrics"]["hf_z"][0] = [8]
    review = compute_quality_score(result)
    assert review["channel_profiles"][0]["covered_seconds"] == 5
    assert review["evidence"][0]["hf_z"] == 8


def test_healthy_channels_do_not_dilute_an_affected_channels_extent():
    small, large = _result(2), _result(64)
    for result in (small, large):
        result["method_results"]["bad_channels"]["candidates"] = [_candidate(100, 110)]
    assert compute_quality_score(small)["channel_profiles"] == compute_quality_score(large)["channel_profiles"]


def test_overlap_across_line_frequencies_is_merged_per_channel():
    result = _result()
    observations = result["method_results"]["spectral"]["line_noise"]["observations"]
    for hz in (50, 60, 100):
        observations.append({"channel_id": "c0", "start_sample": 101, "stop_sample": 105,
                             "nominal_hz": hz, "status": "computed", "flagged": True,
                             "peak_frequency_hz": hz, "amplitude_uv": 2})
    profile = compute_quality_score(result)["channel_profiles"][0]
    assert profile["covered_seconds"] == 2
    assert profile["frequencies_hz"] == [50, 60, 100]


def test_missing_60hz_coverage_does_not_hide_a_measured_50hz_finding():
    result = _result()
    line = result["method_results"]["spectral"]["line_noise"]
    line["observations"] = [r for r in line["observations"] if r["nominal_hz"] == 50]
    line["observations"][0].update(flagged=True, peak_frequency_hz=50, amplitude_uv=2)
    review = compute_quality_score(result)
    assert review["channel_profiles"][0]["covered_seconds"] == 5
    row = _row(review, "line")
    assert row["status"] == "部分范围已计算"
    assert "50 Hz 候选" in row["finding"] and "待核 60 Hz" in row["finding"]


def test_ineligible_or_missing_statistics_are_reported_as_pending():
    result = _result()
    bad = result["method_results"]["bad_channels"]
    bad["window_metrics"]["hf_z"][0][0] = None
    row = _row(compute_quality_score(result), "channels")
    assert row["status"] == "部分统计待核" and "未检出" not in row["finding"]
    bad["groups"][0]["thresholds_active"] = False
    bad["groups"][0]["status"] = "small_comparison_group"
    row = _row(compute_quality_score(result), "channels")
    assert row["status"] == "比较组待核" and "未检出" not in row["finding"]
    assert all(g["reason"] == "可比较通道数量不足" for g in compute_quality_score(result)["coverage_gaps"])


def test_one_valid_window_does_not_hide_missing_statistics_or_tail():
    result = _result()
    bad = result["method_results"]["bad_channels"]
    bad["windows"] = [{"start_sample": 100, "stop_sample": 102}, {"start_sample": 102, "stop_sample": 108}]
    for key in bad["window_metrics"]:
        bad["window_metrics"][key] = [[1, None], [1, None]]
    review = compute_quality_score(result)
    assert _row(review, "channels")["status"] == "部分统计待核"
    gaps = [g for g in review["coverage_gaps"] if g["dimension_id"] == "channels"]
    assert [(g["start_sample"], g["stop_sample"]) for g in gaps] == [(102,110),(102,110)]


def test_frequency_computed_on_one_channel_does_not_hide_another_channel_gap():
    result = _result()
    for obs in result["method_results"]["spectral"]["line_noise"]["observations"]:
        if obs["channel_id"] == "c1":
            obs["status"] = "no_residual_power"
    review = compute_quality_score(result)
    assert _row(review, "line")["status"] == "部分范围已计算"
    gaps = [g for g in review["coverage_gaps"] if g["dimension_id"] == "line"]
    assert len(gaps) == 2 and all(g["channel_id"] == "c1" for g in gaps)


@pytest.mark.parametrize("status", ["insufficient_duration", "insufficient_sampling_rate", "constant_reference"])
def test_unrun_event_detector_does_not_report_zero_events(status):
    result = _result()
    result["method_results"]["events"]["by_kind"]["ecg"].update(
        status=status, source_reference={"raw_name": "ECG", "channel_id": "ref"})
    row = _row(compute_quality_score(result), "ecg")
    assert row["status"] == "检测条件待核"
    assert "0 个" not in row["finding"]


@pytest.mark.parametrize("kind", ["amplitude_deviation", "flat_signal", "file_endpoint_plateau", "flatline"])
def test_amplitude_and_platform_observations_remain_for_human_judgment(kind):
    result = _result()
    target = result["method_results"]["bad_channels"]["candidates"] if kind in {"amplitude_deviation", "flat_signal"} else result["decisions"]["review_queue"]
    target.append(_candidate(100, 110, kind))
    review = compute_quality_score(result)
    assert len(review["observations_for_human"]) == 1
    assert review["human_overall_grade"] is None


def test_alpha_fit_failure_and_event_count_remain_distinct_from_quality():
    result = _result()
    result["method_results"]["spectral"]["parameterization"]["full"] = [
        {"alpha_peaks": [{"height_db": 20}], "status": "computed"}, {"alpha_peaks": [], "status": "fit_failed"}]
    result["method_results"]["events"]["by_kind"]["eog"].update(
        source_reference={"raw_name": "EOG"}, detected_event_count=500,
        status="computed", event_time_span_seconds=[50, 54])
    review = compute_quality_score(result)
    assert _row(review, "spectrum")["status"] == "部分拟合待核"
    assert "500" in _row(review, "eog")["finding"]
    assert all(r["human_grade"] is None for r in review["rows"])


def test_missing_auxiliary_placeholder_is_not_a_measured_zero_event_count():
    result = _result()
    result["method_results"]["events"]["by_kind"]["ecg"].update(
        status="missing_reference", source_reference={"raw_name": "ECG", "channel_id": None})
    row = _row(compute_quality_score(result), "ecg")
    assert row["status"] == "辅助导联待补充"
    assert "0 个" not in row["finding"]


def test_nonfinite_samples_are_located_on_original_clock():
    result = _result()
    result["raw_data_volts"][0, 1:3] = np.nan
    profile = compute_quality_score(result)["channel_profiles"][0]
    assert profile["covered_seconds"] == 1
    assert profile["intervals"][0]["start_sample"] == 101


def test_human_template_has_initial_score_reasons_and_blank_adjustments(tmp_path):
    result = _result()
    result["features"]["channel_names"] = ["same|<tag>", "same|<tag>"]
    result["method_results"]["bad_channels"]["candidates"] = [_candidate(100, 110)]
    review = write_score_files(result, tmp_path)
    report = render_score_review(result, {"title": "<example>"}, with_figures=False)
    assert "same&#124;&lt;tag&gt; [0]" in report and "<tag>" not in report
    assert "累计证据窗" in report and "最长连续证据窗" in report
    assert all(word in report for word in ("扣分", "满分", "自动初评", "人工调整后得分", "评分标准"))
    assert all(word not in report for word in ("验收", "doi.org", "methods.md"))
    assert "score_review.tsv" in report and ".png" not in report
    assert json.loads((tmp_path/"quality_score.json").read_text()) == review
    with (tmp_path/"score_review.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 7
    assert all(row["人工调整后得分"] in {"", "—"} for row in rows)
    assert all(row["调整理由"] == row["审核人"] == row["日期"] == "" for row in rows)
    assert all(row["人工调整后得分"] == "—" for row in rows if row["自动得分"] == "未评分")
    assert rows[-1]["检查项"] == "整体质量"


def _measured(n_channels=20):
    """Known analysis arrays isolate rule evaluation from detector performance."""
    result = _result(n_channels)
    f = result["features"]
    f.update(sampling_rate_hz=200, start_sample=1000, stop_sample=3000, start_seconds=5, stop_seconds=15)
    time = np.arange(2000)/200
    result["raw_data_volts"] = np.tile(10e-6*np.sin(2*np.pi*10*time), (n_channels, 1))
    bad, spectral = result["method_results"]["bad_channels"], result["method_results"]["spectral"]
    bad["windows"] = [{"start_sample": 1000, "stop_sample": 3000}]
    spectral["psd"] = {"frequencies_hz": list(range(101)), "power_v2_hz": np.ones((n_channels,101)).tolist(),
                       "channel_ids": f["channel_ids"]}
    spectral["line_noise"]["observations"] = [
        {"channel_id": cid, "nominal_hz": hz, "status": "computed", "flagged": False,
         "start_sample": 1000, "stop_sample": 3000} for cid in f["channel_ids"] for hz in (50,60)]
    return result


@pytest.mark.parametrize("missing,expected", [(0,20),(1,15),(19,15),(20,10),(199,10),(200,5),(999,5),(1000,0),(2000,0)])
def test_integrity_integer_boundaries_use_worst_channel(missing, expected):
    result = _measured()
    result["raw_data_volts"][0, :missing] = np.nan
    review = compute_quality_score(result)
    assert _row(review, "integrity")["auto_points"] == expected
    assert all(r["auto_points"] is None or type(r["auto_points"]) is int for r in review["rows"])
    assert review["auto_total"] == sum(r["auto_points"] for r in review["rows"] if r["auto_points"] is not None)


@pytest.mark.parametrize("length,expected", [(0,30),(1,25),(200,25),(201,20),(1000,20),(1001,15),(2000,15)])
def test_channel_duration_integer_boundaries(length, expected):
    result = _measured()
    if length:
        result["method_results"]["bad_channels"]["candidates"] = [_candidate(1000,1000+length)]
    assert _row(compute_quality_score(result), "channels")["auto_points"] == expected


def test_extra_healthy_channels_do_not_hide_one_bad_channel_score():
    scores = []
    for count in (4,64):
        result = _measured(count)
        result["method_results"]["bad_channels"]["candidates"] = [_candidate(1000,3000)]
        scores.append(_row(compute_quality_score(result), "channels")["auto_points"])
    assert scores == [15,15]


def test_convergent_evidence_and_spatial_extent_get_separate_integer_deductions():
    result = _measured()
    bad = result["method_results"]["bad_channels"]
    bad["candidates"] = [_candidate(1000,3000), _candidate(1000,3000,"high_frequency_noise"),
                         {**_candidate(1000,1400), "channel_id": "c1"}]
    bad["window_metrics"]["hf_z"][0] = [8]
    row = _row(compute_quality_score(result), "channels")
    assert row["auto_points"] == 0  # 15 duration + 5 multiple + 5 scope + 5 convergence.
    bad["candidates"] += copy.deepcopy(bad["candidates"])
    assert _row(compute_quality_score(result), "channels")["auto_points"] == 0


def test_line_strength_and_scope_change_score_and_preserve_clock():
    results = []
    for ratio, n_hits in ((.05,1),(.30,1),(.30,10)):
        result = _measured()
        observations = result["method_results"]["spectral"]["line_noise"]["observations"]
        for i in range(n_hits):
            observations.append({"channel_id": f"c{i}", "nominal_hz":50, "peak_frequency_hz":50,
                "amplitude_uv":10*ratio, "start_sample":1200, "stop_sample":2000, "status":"computed", "flagged":True})
        row = _row(compute_quality_score(result), "line")
        results.append(row["auto_points"])
        assert row["score_metrics"]["max_rms_ratio"] == pytest.approx(ratio)
        assert row["score_evidence"][0]["start_seconds"] == 6
        assert row["score_evidence"][0]["stop_seconds"] == 10
    assert results == [16,10,4]


def test_psd_alpha_and_model_fit_do_not_replace_spectral_analyzability():
    result = _measured()
    spec = result["method_results"]["spectral"]
    assert _row(compute_quality_score(result), "spectrum")["auto_points"] == 10
    for row in spec["parameterization"]["full"]:
        row.update(alpha_peaks=[], status="fit_failed", r_squared=0)
    assert _row(compute_quality_score(result), "spectrum")["auto_points"] == 10
    spec["psd"]["power_v2_hz"][0] = [0]*101
    assert _row(compute_quality_score(result), "spectrum")["auto_points"] == 5
    for i in range(3):
        spec["psd"]["power_v2_hz"][i] = [0]*101
    assert _row(compute_quality_score(result), "spectrum")["auto_points"] == 0


def test_single_zero_psd_bin_preserves_valid_spectrum_and_frequency_scope_is_literal():
    result = _measured()
    spec = result["method_results"]["spectral"]
    for row in spec["psd"]["power_v2_hz"]:
        row[10] = 0
    spec["config"] = {"line_fundamentals_hz": [50]}
    spec["line_noise"]["observations"] = [r for r in spec["line_noise"]["observations"] if r["nominal_hz"] == 50]
    review = compute_quality_score(result)
    assert _row(review, "spectrum")["auto_points"] == 10
    assert "50 Hz" in _row(review, "line")["score_reason"]
    assert "60" not in _row(review, "line")["score_reason"]


@pytest.mark.parametrize("header,expected,known", [("HP:0Hz LP:0Hz N:0Hz",10,[]),
    ("HP:0.2Hz LP:30Hz",5,["HP","LP"]),("HP:0.2Hz LP:10Hz",0,["HP","LP"])])
def test_known_filters_limit_psd_band_and_zero_headers_remain_unspecified(header,expected,known):
    result = _measured()
    result["manifest"] = {"acquisition_filter_headers": [{"channel_id":cid,"prefilter":header} for cid in result["features"]["channel_ids"]]}
    row = _row(compute_quality_score(result), "spectrum")
    assert row["auto_points"] == expected
    assert row["score_metrics"]["channels"][0]["declared_filter_fields"] == known


def test_score_evaluation_preserves_raw_input_and_decisions():
    result = _measured()
    before = copy.deepcopy(result)
    compute_quality_score(result)
    np.testing.assert_array_equal(result.pop("raw_data_volts"), before.pop("raw_data_volts"))
    assert result == before
