"""Report conclusions preserve frequency ambiguity and measured alpha evidence."""
import copy

import pytest

from brainelecqc.eeg.automatic_report import render_automatic_report
from brainelecqc.eeg.spectral_findings import alpha_conclusion, line_noise_conclusion
from tests.test_eeg_scoring import _candidate, _result


@pytest.mark.parametrize("frequencies,expected", [([], "工频检测合格：未检出显著 50/60 Hz 干扰。"),
    ([50], "检出 50 Hz 工频干扰"), ([60], "检出 60 Hz 工频干扰"),
    ([50, 60], "检出 50/60 Hz 工频干扰"), ([100], "谐波检查检出 100 Hz 窄带干扰")])
def test_line_conclusion_uses_detected_frequencies(frequencies, expected):
    result = _result()
    observations = result["method_results"]["spectral"]["line_noise"]["observations"]
    observations.extend({"channel_id": "c0", "nominal_hz": hz, "flagged": True} for hz in frequencies)
    before = copy.deepcopy(result["method_results"])
    review = {"rows": [{"dimension_id": "line", "auto_points": 20 if not frequencies else 10}]}
    conclusion = line_noise_conclusion(result, review)
    assert expected in conclusion
    if frequencies:
        assert "工频检测合格" not in conclusion and "Ch0（1/2 道）" in conclusion
    if frequencies == [100]:
        assert "检出 50 Hz 工频干扰" not in conclusion
    assert result["method_results"] == before


def test_partial_line_coverage_cannot_claim_pass_or_an_unmeasured_frequency():
    result = _result()
    review = {"rows": [{"dimension_id": "line", "auto_points": None}]}
    assert line_noise_conclusion(result, review).startswith("工频检测覆盖不足")
    result["method_results"]["spectral"]["config"] = {"line_fundamentals_hz": [50]}
    review["rows"][0]["auto_points"] = 20
    assert line_noise_conclusion(result, review) == "工频检测合格：未检出显著 50 Hz 干扰。"


def _alpha_result():
    result = _result()
    result["features"]["channel_names"] = ["EEG O2-M1", "T8"]
    spectral = result["method_results"]["spectral"]
    spectral["parameterization"]["full"] = [
        {"channel_id": "c0", "status": "computed", "alpha_peaks": [
            {"center_frequency_hz": 9., "height_db": 2.}, {"center_frequency_hz": 10., "height_db": 5.}]},
        {"channel_id": "c1", "status": "computed", "alpha_peaks": []}]
    spectral["alpha_power"] = {"full": [
        {"channel_id": "c0", "channel_name": "EEG O2-M1", "alpha_power_status": "computed",
         "relative_alpha_status": "computed", "alpha_power_uv2": 123.4, "relative_alpha_percent": 56.7,
         "start_seconds": 50, "stop_seconds": 54}]}
    return result


def test_alpha_summary_reports_measured_named_leads_and_does_not_infer_eye_state():
    result = _alpha_result()
    before = copy.deepcopy(result["method_results"])
    conclusion = alpha_conclusion(result)
    assert "1/2 道检出拟合 α 峰" in conclusion
    assert "EEG O2-Ref：主峰 10.0 Hz，α 功率 123.4 µV²，占比 56.7%" in conclusion
    assert "50–54 s" in conclusion and "闭眼" not in conclusion and "合格" not in conclusion
    assert result["method_results"] == before


def test_failed_fit_is_not_counted_as_absent_alpha_and_power_remains_visible():
    result = _alpha_result()
    result["method_results"]["spectral"]["parameterization"]["full"][0]["status"] = "failed"
    conclusion = alpha_conclusion(result)
    assert "已完成拟合的 1/2 道未检出独立 α 峰" in conclusion
    assert "主峰 10.0" not in conclusion and "α 功率 123.4" in conclusion


def test_all_channel_psd_is_followed_by_alpha_line_and_channel_findings():
    result = _result()
    result["method_results"]["bad_channels"]["candidates"] = [_candidate(100, 110)]
    report = render_automatic_report(result, {"figure_keys": ["03_all_channels_psd", "04_line_noise"]})
    section = report.split("## 全通道频谱\n\n")[1].split("## 工频检测")[0]
    assert section.index("![全通道频谱]") < section.index("- **α：**") < section.index("- **工频：**")
    assert "可疑坏道：Ch0（低相关）" in section
    line = report.split("## 工频检测\n\n")[1].split("## 人工核对")[0]
    assert line.startswith("**工频检测合格：未检出显著 50/60 Hz 干扰。**")
    assert "算法自动评分：**20/20**" in line
    assert "道命中" not in line and "已测 50 / 60" not in line
