"""Review reports preserve every candidate and distinguish pending human decisions."""
import re

from brainelecqc.eeg.report import render_eeg_review_report, render_eeg_review_items
from tests.test_eeg_plotting import _example


def test_report_locations_coverage_and_pending_decisions():
    _, features, decisions, _ = _example()
    base = decisions["review_queue"][0]
    decisions["review_queue"] = [dict(base, kind="eog"), dict(base, kind="ecg"),
        dict(base, kind="eog", start_seconds=108, stop_seconds=110, start_sample=10800, stop_sample=11000)]
    features["reference_channels"] = {"ecg": {"raw_name": "ECG ref"}}
    report = render_eeg_review_report(features, decisions, {"title": "EEG review"})
    items = render_eeg_review_items(features, decisions)
    assert "核对 1 处自动剔除，复核 2 类候选" in report
    assert "EEG B | 102.000–105.000 s" in report
    assert "EEG A 100.000–102.000 s" in report
    assert "ECG = ECG ref" in report and "保留 85.00%" in report
    assert report.count("](") == 4  # Three figures and the detailed review list.
    rows = re.findall(r"\| R\d{3} \|[^\n]+", items)
    assert len(rows) == 3
    assert sum("100.000–102.000 s" in row for row in rows) == 2
    assert sum("108.000–110.000 s" in row for row in rows) == 1
    assert all("待复核" in row for row in rows)
    assert "接受 / 调整 / 恢复" in report


def test_empty_queue_no_figures_and_channel_label_escaping():
    _, features, decisions, _ = _example()
    decisions["review_queue"] = []
    features["channel_names"] = ["same|<tag>", "same|<tag>"]
    report = render_eeg_review_report(features, decisions, {"title": "<EEG>"}, with_figures=False)
    assert "![" not in report
    assert "候选清单为空" in report
    assert "&lt;EEG&gt;" in report
    assert "same&#124;&lt;tag&gt; [1]" in report
    assert "<tag>" not in report


def test_review_only_report_explains_scores_and_traces_signed_reference_peak():
    import numpy as np
    from brainelecqc.eeg.features import compute_eeg_features
    from brainelecqc.eeg.policy import propose_eeg_decisions
    from brainelecqc.eeg.methods import render_measurement_methods
    time = np.arange(2048)/256
    data = np.sin(2*np.pi*10*time)[None]*20e-6
    features = compute_eeg_features(data, 256, ["c0"], ["EEG"], references={"ecg": -data[0]})
    decisions = propose_eeg_decisions(data, features)
    report = render_eeg_review_report(features, decisions, {})
    methods = render_measurement_methods(features, decisions)
    items = render_eeg_review_items(features, decisions)
    assert "保留全部样本" in report and "## 3. 待审核数据" in report
    assert "自动剔除" not in report and "methods.md" in report
    assert "50/60 是两组待查频率" in report
    assert "| EEG | ECG | -1.000 | 0–4 s |" in methods
    assert "Thomson F" in methods and "待实现" in methods
    assert all(x["method_id"] in features["measurement_methods"] for x in decisions["review_queue"])
    assert "ECG r -1" in items
    assert all(x["candidate_id"] in items for x in decisions["review_queue"])
