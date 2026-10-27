"""Report decisions stay tied to measured features and original derivations."""
import copy

import pytest

from brainelecqc.eeg.automatic_report import render_automatic_report
from brainelecqc.eeg.display import display_channel_name, display_channel_names
from brainelecqc.eeg.score_summary import attach_score_summary
from brainelecqc.eeg.scoring import compute_quality_score
from tests.test_eeg_scoring import _result, _candidate


@pytest.mark.parametrize("points,weight,expected", [(43,70,61),(36,70,51),(32,70,46),
    (54,70,77),(67,80,84),(41,70,59),(1,8,13),(7,8,88),(0,70,0),(70,70,100)])
def test_percentage_uses_integer_half_up_rounding(points, weight, expected):
    rows = [{"dimension_id":"scored", "label":"已测", "auto_points":points,
             "max_points":weight, "auto_deduction":weight-points, "score_reason":"已测证据"},
            {"dimension_id":"missing", "label":"缺导联", "auto_points":None,
             "max_points":100-weight, "auto_deduction":None, "score_reason":"缺辅助导联"}]
    review = attach_score_summary({"rows":rows})
    assert review["score_100"] == expected
    assert review["score_max"] == 100
    assert review["assessment_coverage"]["evaluated_weight"] == weight
    assert review["assessment_coverage"]["unavailable_dimensions"] == ["missing"]
    assert rows[1]["auto_points"] is None and not rows[1]["human_adjustable"]


def test_no_measured_features_has_no_score_not_perfect_score():
    review = attach_score_summary({"rows":[{"dimension_id":"missing", "auto_points":None, "max_points":100}]})
    assert review["score_100"] is None
    assert review["assessment_coverage"]["evaluated_weight"] == 0


def test_equal_coverage_keeps_different_evaluated_feature_profiles_visible():
    first, second = _result(), _result()
    second["method_results"]["bad_channels"]["groups"][0]["thresholds_active"] = False
    a, b = compute_quality_score(first), compute_quality_score(second)
    assert a["score_100"] == b["score_100"] == 100
    assert a["assessment_coverage"] != b["assessment_coverage"]
    assert "channels" in b["assessment_coverage"]["unavailable_dimensions"]


@pytest.mark.parametrize("raw,label", [("EEG F4-M1","EEG F4-Ref"),("C3-M2","C3-Ref"),
    ("EOG E1-M2","EOG E1-Ref"),("F4-A1","F4-Ref"),("A1-A2","A1-Ref"),
    ("LA1-LA2","LA1-LA2"),("M1","M1"),("F4-M10","F4-M10"),("C3.","C3.")])
def test_reference_display_preserves_contact_pairs_and_source_names(raw, label):
    assert display_channel_name(raw) == label


def test_alias_collisions_remain_distinct():
    assert display_channel_names(["F4-M1", "F4-M2"]) == ["F4-Ref [row 0]", "F4-Ref [row 1]"]


def test_report_starts_with_algorithm_decision_and_links_each_png_to_scorecard():
    result = _result()
    result["features"]["channel_names"] = ["EEG F4-M1", "EEG C3-M2"]
    result["method_results"]["bad_channels"]["candidates"] = [_candidate(100,110)]
    before = copy.deepcopy(result)
    report = render_automatic_report(result, {"title":"record", "figure_keys":["02_bad_channels", "04_line_noise"]})
    assert report.startswith("# record · EEG 自动 QC 报告")
    assert "主要扣分项" in report and "分项扣 15 分" in report
    assert "F4-Ref" in report and "F4-M1" not in report
    assert report.count("[对应评分报告](score_report.md)") == 2
    assert "EEG QC 图册" not in report and "Haaglanden" not in report
    assert ".svg" not in report and ".pdf" not in report and "分待评" not in report
    assert result["features"] == before["features"]
    assert result["method_results"] == before["method_results"]
    section = report.split("## 坏道：高频与相关\n\n")[1].split("## 工频检测")[0]
    assert section.startswith("**算法自动检测：坏道 EEG F4-Ref（1/2 道）。**")
    assert section.index("**算法自动检测") < section.index("- EEG F4-Ref") < section.index("算法自动评分")
    assert "自动分项" not in report


def test_report_separates_eeg_and_auxiliary_names_and_links_supplemental_alpha():
    result = _result()
    result["features"]["channel_names"] = ["EEG F4-M1", "EEG C3-M2"]
    result["features"]["reference_channels"] = {"eog": {"raw_name": "EOG E1-M2"}, "ecg": {"raw_name": "ECG"}}
    report = render_automatic_report(result, {"figure_keys": ["03_psd_alpha", "08_alpha_time", "05_ecg", "06_eog"]})
    assert "本次 EEG：EEG F4-Ref、EEG C3-Ref" in report
    assert "本次辅助导联：EOG E1-Ref、ECG" in report
    assert "![PSD 与 α 能量](03_psd_alpha.png)" in report
    assert "[α 能量随时间变化](08_alpha_time.png)" in report
    assert "![α 能量随时间变化]" not in report
    assert "![PSD 与 α 能量](08_alpha_time.png)" not in report
    assert "核对下方心搏发生时" in report and "核对下方眼动发生时" in report


def test_bad_channel_heading_keeps_local_outliers_and_amplitude_only_separate():
    result = _result(3)
    result["features"]["stop_sample"] = 200
    import numpy as np
    result["raw_data_volts"] = np.ones((3, 100))*1e-6
    result["method_results"]["bad_channels"]["windows"][0]["stop_sample"] = 200
    result["method_results"]["bad_channels"]["candidates"] = [
        _candidate(100, 200), {**_candidate(100, 101), "channel_id": "c1"},
        {**_candidate(100, 200, "amplitude_deviation"), "channel_id": "c2"}]
    report = render_automatic_report(result, {"figure_keys": ["02_bad_channels"]})
    section = report.split("## 坏道：高频与相关\n\n")[1].split("## 人工核对")[0]
    assert section.startswith("**算法自动检测：坏道 Ch0（1/3 道）；局部异常 Ch1。**")
    assert "Ch2" not in section
    assert "Ch1：低相关；累计 0.5 s" in section


def test_unavailable_bad_channel_checks_are_not_reported_as_normal():
    result = _result()
    result["method_results"]["bad_channels"]["groups"][0]["thresholds_active"] = False
    report = render_automatic_report(result, {"figure_keys": ["02_bad_channels"]})
    section = report.split("## 坏道：高频与相关\n\n")[1].split("## 人工核对")[0]
    assert "**算法自动检测：通道统计条件不足" in section
    assert "未检出坏道" not in section
