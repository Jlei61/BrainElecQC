"""Extent-based integer bands, shared by tables and automatic scores."""
from .impact import percent, proportion_points, time_burden, union_intervals


def score_line_burden(result, review, outcome):
    if outcome["score"] is None:
        return outcome
    f = result["features"]
    hits = [e for e in review.get("evidence", []) if e["dimension_id"] == "line"]
    per = {cid: union_intervals((e["start_sample"], e["stop_sample"]) for e in hits
                               if e["channel_id"] == cid) for cid in f["channel_ids"]}
    affected = sum(b-a for intervals in per.values() for a, b in intervals)
    total = len(per)*(f["stop_sample"]-f["start_sample"])
    fraction, count = affected/total, sum(bool(intervals) for intervals in per.values())
    points = proportion_points(fraction, 20)
    burden = time_burden(f, [(e["start_sample"], e["stop_sample"]) for e in hits])
    metrics = {**outcome.get("metrics", {}), "time_burden": burden,
               "channel_time_burden": {"affected_channel_samples": affected, "total_channel_samples": total,
                   "fraction": fraction, "affected_channels": count, "total_channels": len(per),
                   "basis": "affected_channel_samples_over_all_channel_samples"}}
    reason = (f"工频影响 {count}/{len(per)} 道；通道×时间占比 {percent(fraction)}，扣 {20-points} 分"
              if hits else outcome["reason"])
    return {**outcome, "score": points, "reason": reason, "metrics": metrics,
            "components": [{"id": "channel_time_burden", "deduction": 20-points}]}


def score_spectrum_burden(result, outcome):
    if outcome["score"] is None:
        return outcome
    metrics = dict(outcome.get("metrics", {}))
    channels = metrics.get("channels", [])
    if not channels:
        return {**outcome, "score": None, "reason": "待计算逐通道目标频带覆盖", "metrics": metrics}
    count = sum(c["band_fraction"] < 1 or not c["valid_psd"] for c in channels)
    fraction = count/len(channels)
    points = proportion_points(fraction, 5)
    metrics["limited_channel_fraction"] = fraction
    metrics["limited_channel_count"] = count
    return {**outcome, "score": points, "metrics": metrics,
            "reason": f"2–40 Hz 频谱受限 {count}/{len(channels)} 道，占比 {percent(fraction)}",
            "components": [{"id": "limited_channel_fraction", "deduction": 5-points}]}


def render_proportion_bands():
    return ["## 异常占比与整数分档", "",
        "| 异常占比 | 时间异常（各 10 分） | 坏道 / 工频（各 20 分） | 缺失 / PSD 受限（各 5 分） |",
        "| --- | ---: | ---: | ---: |",
        "| 0 | 10 | 20 | 5 |", "| >0–<2% | 9 | 19 | 4 |",
        "| 2–<5% | 8 | 18 | 4 |", "| 5–<10% | 7 | 16 | 3 |",
        "| 10–<25% | 5 | 12 | 2 |", "| 25–<50% | 3 | 6 | 1 |",
        "| ≥50% | 0 | 0 | 0 |", "",
        "时间异常：短平线/平台、ECG、EOG、EMG 各自计算时间并集占比；坏道：持续异常通道占比；"
        "工频：逐道合并命中窗后的通道×时间占比；缺失：缺失最多一道的样本占比；PSD：频带受限通道占比。",
        "", "头文件独立计 10 分，逐项核对格式、通道、采样率、标定、长度与时间。合计满分 100。", ""]
