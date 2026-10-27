"""Front-of-report proportions agree with their scoring measurements."""
import csv
from brainelecqc.eeg.automatic_report import render_automatic_report
from brainelecqc.eeg.scoring import compute_quality_score
from brainelecqc.eeg.score_report import write_score_files, render_score_review
from tests.test_eeg_scoring import _result, _candidate


def test_channel_count_is_separate_from_short_local_events():
    result = _result(64)
    result["method_results"]["bad_channels"]["candidates"] = [_candidate(100, 110)]
    review = compute_quality_score(result)
    summary = next(r for r in review["quality_overview"] if r["dimension_id"] == "channels")
    assert summary["fraction"] == 1/64
    assert summary["numerator"] == 1 and summary["denominator"] == 64
    assert summary["unit"] == "道"


def test_unavailable_measurements_remain_blank_in_front_summary():
    review = compute_quality_score(_result())
    for key in ("ecg", "eog"):
        entry = next(r for r in review["quality_overview"] if r["dimension_id"] == key)
        assert entry["fraction"] is None and entry["auto_points"] is None
        assert entry["metric"] == "未评分"


def test_same_summary_is_first_in_both_reports_and_exported_as_tsv(tmp_path):
    result = _result()
    primary = render_automatic_report(result, {"figure_keys": ["03_all_channels_psd"]})
    score_report = render_score_review(result, {}, with_figures=False)
    assert primary.index("## 质量总结") < primary.index("## 主要扣分项") < primary.index("## 全通道频谱")
    assert score_report.index("## 质量总结") < score_report.index("| 检查项 | 满分")
    assert "异常通道占比 **0.0%**（0 / 2 道）" in primary
    assert "算法自动评分" in primary
    assert "| 主要结果 | 人工复核 |" in primary
    assert "核对未评分原因，补足条件后重新计算" in primary
    assert "本次分析范围" in primary and "| 结论 | 待评：" in primary
    review = write_score_files(result, tmp_path)
    with (tmp_path/"quality_overview.tsv").open() as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    assert [e["dimension_id"] for e in entries] == [e["dimension_id"] for e in review["quality_overview"]]
    assert next(e for e in entries if e["dimension_id"] == "ecg")["fraction"] == ""
