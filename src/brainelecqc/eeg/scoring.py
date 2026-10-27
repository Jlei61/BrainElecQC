"""Feature-specific evidence and integer automatic scores for human review."""
from collections import defaultdict

import numpy as np
from .score_rules import attach_integer_scores, VERSION


RUBRIC_VERSION = VERSION
DIMENSIONS = (("integrity", "数值完整性"), ("channels", "通道质量"),
              ("line", "工频干扰"), ("spectrum", "PSD / α"),
              ("ecg", "心电影响"), ("eog", "眼电影响"))
HUMAN_GRADES = {
    "可用": "目标信号可辨，满足本次分析要求",
    "局部处理后可用": "问题范围明确，处理指定通道或片段后可满足分析要求",
    "重点处理后复核": "干扰较明显或涉及关键通道，处理后重新检查目标信号",
    "可用性受限": "关键分析信息缺失或受损，需要调整分析目标或补充采集",
    "待评": "补充记录状态、用途或定位证据后判断",
}
LABELS = {"nonfinite_input": "缺失数值", "high_frequency_noise": "高频比例离群",
          "low_correlation": "低相关", "intermittent_dropout": "间歇掉线",
          "amplitude_deviation": "幅度离群", "flat_signal": "平直信号",
          "file_endpoint_plateau": "端点平台", "flatline": "低变化平台"}


def _union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([int(a), int(b)])
    return merged


def _runs(mask, origin):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a+origin), int(b+origin)) for a, b in edges.reshape(-1, 2)]


def _gaps(intervals, origin, stop):
    missing, cursor = [], origin
    for a, b in _union(intervals):
        if a > cursor:
            missing.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < stop:
        missing.append((cursor, stop))
    return missing


def compute_quality_score(result):
    """Summarize measured findings without averaging across healthy channels.

    HF whole-record candidates are localized using existing per-window z values
    and the detector's configured HF threshold. Coverage is the union of evidence
    windows; longest continuous coverage is reported separately. Neither is an
    estimate of the true artifact onset/offset. All human grades start empty.
    """
    f, methods = result["features"], result["method_results"]
    data = np.asarray(result["raw_data_volts"])
    ids, names = f["channel_ids"], f["channel_names"]
    origin, stop, fs = f["start_sample"], f["stop_sample"], f["sampling_rate_hz"]
    if data.shape != (len(ids), stop-origin) or not ids or stop <= origin:
        raise ValueError("Review requires nonempty channel × sample data matching the input")
    bad, spectral = methods["bad_channels"], methods["spectral"]
    evidence, observations, coverage_gaps = [], [], []

    def note_gaps(key, cid, intervals, reason):
        for a, b in _gaps(intervals, origin, stop):
            coverage_gaps.append({"dimension_id": key, "channel_id": cid, "raw_name": names[ids.index(cid)],
                "start_sample": a, "stop_sample": b,
                "start_seconds": f["start_seconds"]+(a-origin)/fs,
                "stop_seconds": f["start_seconds"]+(b-origin)/fs, "reason": reason})

    def add(key, kind, cid, a, b, reason, **measurements):
        if cid not in ids or not origin <= a < b <= stop:
            raise ValueError("Review evidence lies outside the selected input")
        evidence.append({"dimension_id": key, "kind": kind, "channel_id": cid,
            "raw_name": names[ids.index(cid)], "start_sample": int(a), "stop_sample": int(b),
            "start_seconds": f["start_seconds"]+(a-origin)/fs,
            "stop_seconds": f["start_seconds"]+(b-origin)/fs, "reason": reason, **measurements})

    for ci, cid in enumerate(ids):
        for a, b in _runs(~np.isfinite(data[ci]), origin):
            add("integrity", "nonfinite_input", cid, a, b, LABELS["nonfinite_input"])

    # A full-record HF statistic identifies a channel, not all its time points.
    hf_candidates = {c["channel_id"] for c in bad["candidates"] if c["kind"] == "high_frequency_noise"}
    eligible = {cid for group in bad["groups"] if group["thresholds_active"] for cid in group["channel_ids"]}
    group_reasons = {"small_comparison_group": "可比较通道数量不足",
                     "reference_group_unconfirmed": "采集参考分组待确认",
                     "duration_shorter_than_analysis_filter": "片段长度不足以完成通道统计"}
    inactive = {cid: group_reasons.get(group.get("status"), "比较条件待核")
                for group in bad["groups"] if not group["thresholds_active"] for cid in group["channel_ids"]}
    for ci, cid in enumerate(ids):
        valid = []
        if cid in eligible:
            for wi, window in enumerate(bad["windows"]):
                values = [bad["window_metrics"][key][ci][wi] for key in ("amplitude_z", "hf_z", "max_abs_correlation")]
                if all(v is not None and np.isfinite(v) for v in values):
                    valid.append((window["start_sample"], window["stop_sample"]))
        note_gaps("channels", cid, valid, inactive.get(cid, "逐窗统计待核"))
    for cid in (cid for cid in ids if cid in hf_candidates & eligible):
        ci = ids.index(cid)
        for wi, window in enumerate(bad["windows"]):
            z = bad["window_metrics"]["hf_z"][ci][wi]
            if z is not None and np.isfinite(z) and z > bad["parameters"]["hf_noise_z"]:
                add("channels", "high_frequency_noise", cid, window["start_sample"], window["stop_sample"],
                    LABELS["high_frequency_noise"], hf_z=z)
    for candidate in bad["candidates"]:
        kind = candidate["kind"]
        if kind in {"low_correlation", "intermittent_dropout"}:
            add("channels", kind, candidate["channel_id"], candidate["start_sample"], candidate["stop_sample"], LABELS[kind])
        elif kind in {"amplitude_deviation", "flat_signal"}:
            observations.append(candidate)
    observations += [r for r in result["decisions"]["review_queue"] if r["kind"] in {"file_endpoint_plateau", "flatline"}]

    line = spectral["line_noise"]
    measured_intervals = defaultdict(list)
    for row in line["observations"]:
        if row["status"] in {"computed", "no_ac_power"}:
            measured_intervals[(row["channel_id"], row["nominal_hz"])].append((row["start_sample"], row["stop_sample"]))
    computed_hz = sorted({hz for _, hz in measured_intervals})
    target_hz = set(spectral.get("config", {}).get("line_fundamentals_hz", (50, 60)))
    target_hz.update(r["nominal_hz"] for r in line["observations"])
    for hz in sorted(target_hz):
        for cid in ids:
            intervals = measured_intervals[(cid, hz)]
            note_gaps("line", cid, intervals, f"{hz:g} Hz 检测范围待核")
    for row in line["observations"]:
        if row["flagged"]:
            add("line", "sinusoidal_line", row["channel_id"], row["start_sample"], row["stop_sample"],
                f"{row['peak_frequency_hz']:g} Hz 窄带正弦",
                nominal_hz=row["nominal_hz"], peak_frequency_hz=row["peak_frequency_hz"], amplitude_uv=row["amplitude_uv"])

    # Channel × feature summaries keep distinct mechanisms separate.
    keys = {(r["dimension_id"], r["kind"], r["channel_id"]) for r in evidence}
    keys |= {("channels", "high_frequency_noise", cid) for cid in hf_candidates}
    profiles = []
    for key, kind, cid in sorted(keys, key=lambda k: (k[0], ids.index(k[2]), k[1])):
        entries = [e for e in evidence if (e["dimension_id"], e["kind"], e["channel_id"]) == (key, kind, cid)]
        intervals = _union([(e["start_sample"], e["stop_sample"]) for e in entries])
        n = sum(b-a for a,b in intervals)
        profiles.append({"dimension_id": key, "kind": kind, "channel_id": cid, "raw_name": names[ids.index(cid)],
            "intervals": [{"start_sample": a, "stop_sample": b,
                "start_seconds": f["start_seconds"]+(a-origin)/fs,
                "stop_seconds": f["start_seconds"]+(b-origin)/fs} for a,b in intervals],
            "location_status": "window_localized" if intervals else "whole_record_statistic_only",
            "covered_seconds": n/fs if intervals else None,
            "longest_continuous_seconds": max((b-a for a,b in intervals), default=0)/fs if intervals else None,
            "reason": LABELS.get(kind, "窄带正弦"),
            "frequencies_hz": sorted({e["peak_frequency_hz"] for e in entries if "peak_frequency_hz" in e}),
            "amplitude_range_uv": [min(e["amplitude_uv"] for e in entries), max(e["amplitude_uv"] for e in entries)] if key == "line" else None})

    rows = [{"dimension_id": key, "label": label, "status": "待核", "finding": "", "review_prompt": "",
             "human_grade": None, "human_reason": ""} for key, label in DIMENSIONS]
    lookup = {r["dimension_id"]: r for r in rows}
    missing = int(np.count_nonzero(~np.isfinite(data)))
    lookup["integrity"].update(status="已检查", finding=f"{missing} 个非有限样本" if missing else "选定片段内数值完整",
                                review_prompt="核对缺失位置及原始波形")
    kinds = defaultdict(set)
    for c in bad["candidates"]:
        if c["kind"] not in {"nonfinite_input"}:
            kinds[c["kind"]].add(c["channel_id"])
    detail = "；".join(f"{LABELS.get(k,k)} {len(v)} 道" for k,v in sorted(kinds.items()))
    ready = len(eligible)
    statistics_available = bool(ready) and not any(g["dimension_id"] == "channels" for g in coverage_gaps)
    if ready == len(ids) and statistics_available:
        status = "已计算"
    elif ready:
        status = "部分统计待核" if not statistics_available else f"部分计算（{ready}/{len(ids)} 道）"
    else:
        status = "比较组待核"
    lookup["channels"].update(status=status, finding=detail or ("已计算范围内候选未检出" if statistics_available else "核对采集参考、比较组和统计量"),
                               review_prompt="逐道核对高频波形、低相关时段及参考分组")
    line_channels = {e["channel_id"] for e in evidence if e["dimension_id"] == "line"}
    found_hz = sorted({e["nominal_hz"] for e in evidence if e["dimension_id"] == "line"})
    line_complete = not any(g["dimension_id"] == "line" for g in coverage_gaps)
    line_status = "已计算" if line_complete else "部分范围已计算" if computed_hz else "待核频率覆盖"
    missing_hz = [hz for hz in sorted(target_hz) if hz not in computed_hz]
    finding = (f"{' / '.join(f'{hz:g}' for hz in found_hz)} Hz 候选，涉及 {len(line_channels)} 道" if line_channels else
               "已检查频率中未检出候选" if computed_hz else "频率或片段长度待核对")
    if computed_hz and missing_hz:
        finding += "；待核 " + "/".join(str(hz) for hz in missing_hz) + " Hz"
    lookup["line"].update(status=line_status, finding=finding, review_prompt="结合窄峰振幅、单道覆盖时间与目标频带判断影响")
    fits = spectral["parameterization"]["full"]
    computed = sum(r["status"] == "computed" for r in fits)
    lookup["spectrum"].update(status="已计算，状态待核" if computed == len(ids) else "部分拟合待核",
        finding=f"{sum(bool(r['alpha_peaks']) for r in fits)}/{len(ids)} 道有 α 峰；{computed}/{len(ids)} 道完成谱拟合",
        review_prompt="结合睁闭眼、睡眠或任务状态核对 PSD 形状、α 峰和残差")
    for kind in ("ecg", "eog"):
        block = methods["events"]["by_kind"][kind]
        if block["source_reference"] and block.get("status") != "missing_reference":
            finding = f"参考 {block['source_reference']['raw_name']}"
            if block.get("status") in {"computed", "no_events", "no_complete_epochs"}:
                finding += f"；{block['detected_event_count']} 个事件候选"
                times = block.get("event_time_span_seconds")
                if times:
                    finding += f"；{times[0]:.3f}–{times[-1]:.3f} s"
                status = "事件待核"
            else:
                reasons = {"insufficient_duration": "片段长度待核", "insufficient_sampling_rate": "采样率条件待核",
                           "constant_reference": "辅助导联恒定，核对接线与导联选择"}
                status = "检测条件待核"
                finding += "；"+reasons.get(block.get("status"), "核对检测输入与参数")
        else:
            status, finding = "辅助导联待补充", "结合 EEG 波形人工检查"
        lookup[kind].update(status=status, finding=finding, review_prompt="核对事件标记、单次 EEG 和平均形态，确认影响范围")
    review = {"rubric_version": RUBRIC_VERSION,
        "data_sha256": f["data_sha256"], "channel_ids": ids, "channel_names": names,
        "input_range": {k: f[k] for k in ("start_sample", "stop_sample", "start_seconds", "stop_seconds")},
        "rows": rows, "evidence": evidence, "channel_profiles": profiles, "coverage_gaps": coverage_gaps,
        "observations_for_human": observations, "human_grade_options": HUMAN_GRADES,
        "analysis_goal": "", "target_band": "", "critical_channels": "", "human_overall_grade": None,
        "human_overall_reason": ""}
    return attach_integer_scores(result, review)
