"""Versioned integer screening rubric on top of measured EEG features.

Detector criteria have published sources; severity bands and point allocations
are explicit project policy. Scores rank review needs and never modify signals.
"""
import copy
import re

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid

from .event_score import score_event_coupling


VERSION = "eeg-integer-score-3.0"
POLICY = {
    "max_points": {"integrity": 20, "channels": 30, "line": 20, "spectrum": 10, "ecg": 10, "eog": 10},
    "minimum_coverage": .95,
    "integrity": {"fraction_edges": [.01, .10, .50], "points": [20, 15, 10, 5, 0]},
    "channels": {"time_edges": [.10, .50], "time_deductions": [0, 5, 10, 15],
                 "persistent_fraction": .10, "multiple_count": 2, "multiple_deduction": 5,
                 "widespread_fraction": .10, "widespread_deduction": 5,
                 "convergent_fraction": .01, "convergent_deduction": 5},
    "line": {"detection_deduction": 2, "ratio_edges": [.10, .25, .50, 1.0],
             "ratio_deductions": [0, 3, 6, 10, 15], "scope_edges": [.10, .25, .50],
             "scope_deductions": [0, 2, 4, 6], "time_edges": [.10, .50], "time_deductions": [0, 2, 4]},
    "spectrum": {"target_band_hz": [2, 40], "minimum_band_fraction": .50, "minor_channel_fraction": .10,
                 "points": [10, 5, 0]},
    "event": {"min_events": 20, "repeatability_min": .50, "ratio_edges": [.10, .25, .50], "points": [10, 8, 5, 2]},
}


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


def _length(intervals):
    return sum(b-a for a, b in _union(intervals))


def _tier(value, edges, values):
    return values[sum(value >= edge for edge in edges)]


def _scored(points, reason, metrics=None, components=None, evidence=None):
    return {"score": points, "reason": reason, "metrics": metrics or {},
            "components": components or [], "evidence": evidence or []}


def _covered(review, dimension, ids, count):
    return all(1-_length([(g["start_sample"], g["stop_sample"]) for g in review["coverage_gaps"]
                         if g["dimension_id"] == dimension and g["channel_id"] == cid])/count
               >= POLICY["minimum_coverage"] for cid in ids)


def _integrity(result):
    data, f = np.asarray(result["raw_data_volts"]), result["features"]
    missing = np.count_nonzero(~np.isfinite(data), axis=1)
    ci, count = int(np.argmax(missing)), data.shape[1]
    ratio = float(missing[ci]/count)
    cfg = POLICY["integrity"]
    points = cfg["points"][0] if ratio == 0 else cfg["points"][1+sum(ratio >= e for e in cfg["fraction_edges"])]
    reason = "全部样本数值完整" if not ratio else f"{f['channel_names'][ci]} 缺失 {int(missing[ci])} 个样本，占该道 {ratio:.0%}"
    return _scored(points, reason, {"worst_channel_id": f["channel_ids"][ci], "missing_samples": int(missing[ci]), "fraction": ratio})


def _channels(result, review):
    f, cfg = result["features"], POLICY["channels"]
    ids, count = f["channel_ids"], f["stop_sample"]-f["start_sample"]
    if not _covered(review, "channels", ids, count):
        return _scored(None, "比较组或有效逐窗统计覆盖不足 95%，核对采集参考与待核范围")
    per = {cid: [] for cid in ids}
    hf, corr = {cid: [] for cid in ids}, {cid: [] for cid in ids}
    for e in review["evidence"]:
        if e["dimension_id"] != "channels":
            continue
        pair = (e["start_sample"], e["stop_sample"])
        per[e["channel_id"]].append(pair)
        if e["kind"] == "high_frequency_noise":
            hf[e["channel_id"]].append(pair)
        if e["kind"] == "low_correlation":
            corr[e["channel_id"]].append(pair)
    fractions = {cid: _length(v)/count for cid, v in per.items()}
    worst = max(ids, key=fractions.get)
    fraction = fractions[worst]
    global_only = any(p["dimension_id"] == "channels" and p["location_status"] == "whole_record_statistic_only"
                      for p in review["channel_profiles"])
    time_deduction = (cfg["time_deductions"][0] if not fraction else
                      cfg["time_deductions"][1+sum(fraction > e for e in cfg["time_edges"])])
    if global_only:
        time_deduction = max(time_deduction, cfg["time_deductions"][1])
    persistent = [cid for cid in ids if fractions[cid] >= cfg["persistent_fraction"]]
    concurrent = {cid: _length([(max(a, c), min(b, d)) for a,b in _union(hf[cid]) for c,d in _union(corr[cid])])/count for cid in ids}
    convergent = [cid for cid in ids if concurrent[cid] > cfg["convergent_fraction"]]
    multiple = len(persistent) >= cfg["multiple_count"]
    widespread = multiple and len(persistent)/len(ids) >= cfg["widespread_fraction"]
    components = [{"id": "duration", "deduction": time_deduction},
        {"id": "multiple_channels", "deduction": cfg["multiple_deduction"] if multiple else 0},
        {"id": "widespread_channels", "deduction": cfg["widespread_deduction"] if widespread else 0},
        {"id": "hf_and_low_correlation", "deduction": cfg["convergent_deduction"] if convergent else 0}]
    why = []
    if fraction:
        ci = ids.index(worst)
        why.append(f"{f['channel_names'][ci]} 异常窗累计 {_length(per[worst])/f['sampling_rate_hz']:g} s / {count/f['sampling_rate_hz']:g} s，扣 {time_deduction} 分")
    elif global_only:
        why.append("全段高频离群、逐窗定位待核，扣 5 分")
    if multiple:
        why.append(f"{len(persistent)} 道各有至少 10% 时间命中，扣 5 分")
    if widespread:
        why.append("上述通道达到总通道数 10%，再扣 5 分")
    if convergent:
        why.append("、".join(f["channel_names"][ids.index(cid)] for cid in convergent)+" 高频与低相关同时命中超过 1% 时间，扣 5 分")
    return _scored(30-sum(c["deduction"] for c in components), "；".join(why) or "逐窗高频、相关与掉线检查通过",
                   {"worst_channel_id": worst, "channel_fractions": fractions, "persistent_channels": persistent,
                    "convergent_fractions": concurrent}, components)


def _line(result, review):
    f, cfg = result["features"], POLICY["line"]
    ids, count, fs = f["channel_ids"], f["stop_sample"]-f["start_sample"], f["sampling_rate_hz"]
    if not _covered(review, "line", ids, count):
        return _scored(None, "待查频率的有效时间覆盖不足 95%，核对检测频率与片段长度")
    hits = [e for e in review["evidence"] if e["dimension_id"] == "line"]
    if not hits:
        frequencies = sorted({r["nominal_hz"] for r in result["method_results"]["spectral"]["line_noise"]["observations"]
                              if r["status"] in {"computed", "no_ac_power"}})
        return _scored(20, "已测 "+" / ".join(f"{v:g}" for v in frequencies)+" Hz 中，显著正弦未检出")
    evidence, ratios = [], []
    band = POLICY["spectrum"]["target_band_hz"]
    for e in hits:
        ci = ids.index(e["channel_id"])
        data = np.asarray(result["raw_data_volts"])[ci, e["start_sample"]-f["start_sample"]:e["stop_sample"]-f["start_sample"]]
        if data.size < 4 or not np.all(np.isfinite(data)) or fs/2 < band[1]:
            return _scored(None, "检出工频候选；计算同窗 2–40 Hz 相对振幅的条件不足")
        freq, psd = welch(data, fs=fs, window="hann", nperseg=len(data), noverlap=0, detrend="constant", scaling="density")
        mask = (freq >= band[0]) & (freq <= band[1])
        power = float(trapezoid(psd[mask], freq[mask]))
        if power <= 0 or not np.isfinite(power):
            return _scored(None, "检出工频候选；同窗目标频带功率为零或无效，核对原始波形")
        rms_uv = np.sqrt(power)*1e6
        ratio = float(e["amplitude_uv"]/np.sqrt(2)/rms_uv)
        evidence.append({**e, "band_rms_uv": float(rms_uv), "line_to_band_rms": ratio})
        ratios.append(ratio)
    worst = evidence[int(np.argmax(ratios))]
    affected = {e["channel_id"] for e in hits}
    durations = {cid: _length([(e["start_sample"], e["stop_sample"]) for e in hits if e["channel_id"] == cid])/count for cid in affected}
    parts = [("significant_line", cfg["detection_deduction"]),
             ("relative_amplitude", _tier(max(ratios), cfg["ratio_edges"], cfg["ratio_deductions"])),
             ("channel_scope", _tier(len(affected)/len(ids), cfg["scope_edges"], cfg["scope_deductions"])),
             ("duration", _tier(max(durations.values()), cfg["time_edges"], cfg["time_deductions"]))]
    reason = (f"显著工频扣 {parts[0][1]} 分；{worst['raw_name']} {worst['start_seconds']:g}–{worst['stop_seconds']:g} s，"
              f"{worst['peak_frequency_hz']:g} Hz / {worst['amplitude_uv']:.2f} µV，振幅比 {max(ratios):.2f}，扣 {parts[1][1]} 分；"
              f"{len(affected)} 道命中，扣 {parts[2][1]} 分；单道最长累计覆盖 {max(durations.values()):.0%}，扣 {parts[3][1]} 分")
    return _scored(max(0, 20-sum(p[1] for p in parts)), reason,
        {"max_rms_ratio": max(ratios), "affected_channels": len(affected), "channel_time_fractions": durations,
         "band_hz": band, "rms_estimator": "same-window Hann periodogram, constant detrend, trapezoidal band integral"},
        [{"id": key, "deduction": value} for key,value in parts], evidence)


def _spectrum(result):
    spectral, f, cfg = result["method_results"]["spectral"], result["features"], POLICY["spectrum"]
    psd = spectral.get("psd")
    if not psd:
        return _scored(None, "待计算目标频带 PSD")
    freq, power = np.asarray(psd["frequencies_hz"], dtype=float), np.asarray(psd["power_v2_hz"], dtype=float)
    ids = f["channel_ids"]
    if power.shape != (len(ids), len(freq)) or psd.get("channel_ids", ids) != ids:
        raise ValueError("PSD scoring requires matching channel order and frequency axis")
    if len(freq) < 2 or not np.all(np.isfinite(freq)) or not np.all(np.diff(freq) > 0):
        raise ValueError("PSD scoring requires a finite increasing frequency axis")
    low, high = cfg["target_band_hz"]
    filters = {r["channel_id"]: r.get("prefilter", "") for r in result.get("manifest", {}).get("acquisition_filter_headers", [])}
    channel_metrics = []
    for ci, cid in enumerate(ids):
        lo, hi = max(low, freq[0]), min(high, freq[-1], f["sampling_rate_hz"]/2)
        header, known = filters.get(cid, ""), []
        for label in ("HP", "LP"):
            match = re.search(rf"\b{label}\s*:\s*([\d.]+)\s*Hz", header, flags=re.I)
            value = float(match[1]) if match else None
            if value is not None and value > 0:
                known.append(label)
                if label == "HP": lo = max(lo, value)
                else: hi = min(hi, value)
        mask = (freq >= lo) & (freq <= hi)
        valid = bool(mask.sum() >= 2 and np.all(np.isfinite(power[ci, mask])) and np.all(power[ci, mask] >= 0)
                     and trapezoid(power[ci, mask], freq[mask]) > 0)
        coverage = max(0., hi-lo)/(high-low) if valid else 0.
        channel_metrics.append({"channel_id": cid, "raw_name": f["channel_names"][ci], "band_fraction": float(coverage),
                                "valid_psd": valid, "prefilter_header": header, "declared_filter_fields": known})
    fractions = [r["band_fraction"] for r in channel_metrics]
    limited = [r for r in channel_metrics if r["band_fraction"] < 1]
    severely_limited = [r for r in channel_metrics if r["band_fraction"] < cfg["minimum_band_fraction"]]
    if not limited:
        points, why = 10, f"{low}–{high} Hz 导出信号频谱可计算，{len(ids)} 道 PSD 数值有效"
    elif len(severely_limited)/len(ids) > cfg["minor_channel_fraction"]:
        points, why = 0, f"{len(severely_limited)} 道可分析频带不足目标的一半或 PSD 无效"
    else:
        points, why = 5, f"{len(limited)} 道目标频带覆盖受限或 PSD 无效"
    return _scored(points, why, {"target_band_hz": [low, high], "channels": channel_metrics,
                               "scope": "exported_signal_spectral_analyzability", "minimum_band_fraction": min(fractions)})


def attach_integer_scores(result, review):
    """Attach reproducible integer initial scores; human entries stay empty."""
    outcomes = {"integrity": _integrity(result), "channels": _channels(result, review),
                "line": _line(result, review), "spectrum": _spectrum(result)}
    outcomes.update({key: score_event_coupling(result, key, POLICY["event"]) for key in ("ecg", "eog")})
    from .extended_score import extend_integer_scores
    expanded_version, policy = extend_integer_scores(result, review, outcomes, POLICY)
    for row in review["rows"]:
        key, outcome = row["dimension_id"], outcomes[row["dimension_id"]]
        points, maximum = outcome["score"], policy["max_points"][key]
        if points is not None and (type(points) is not int or not 0 <= points <= maximum):
            raise ValueError("Automatic points must be bounded integers")
        row.update(max_points=maximum, auto_points=points, auto_deduction=None if points is None else maximum-points,
                   score_reason=outcome["reason"], score_metrics=outcome["metrics"],
                   score_components=outcome.get("components", []), score_evidence=outcome.get("evidence", []),
                   human_points=None)
    assessed = [r for r in review["rows"] if r["auto_points"] is not None]
    review.update(rubric_version=expanded_version or VERSION, rubric_basis="published_detector_metrics_with_project_integer_severity_policy",
                  scoring_policy=copy.deepcopy(policy), auto_total=sum(r["auto_points"] for r in assessed),
                  assessed_max=sum(r["max_points"] for r in assessed),
                  pending_max=sum(r["max_points"] for r in review["rows"] if r["auto_points"] is None),
                  human_total=None, reviewer="", review_date="")
    from .score_summary import attach_score_summary
    from .quality_overview import attach_quality_overview
    return attach_quality_overview(attach_score_summary(review), result)


def rubric_rows(version=VERSION):
    """Human-readable counterpart of the versioned project rulebook."""
    if version == "eeg-integer-score-7.0":
        from .extended_score import rubric_rows as expanded_rows
        return expanded_rows(version)
    if version in {"eeg-integer-score-4.0", "eeg-integer-score-5.0", "eeg-integer-score-6.0"}:
        from .extended_score import rubric_rows as expanded_rows
        extended, original = expanded_rows(version), rubric_rows()
        if version == "eeg-integer-score-6.0":
            event = ("ECG、EOG · 各 10", "先通过跨半段与奇偶锁时重复性检查，再逐事件与另一半段的独立模板比较。相关 ≥0.50 且投影振幅比 ≥0.10 的完整事件窗合并计时；占本段时长 0 / >0–<1% / 1–<10% / 10–<50% / ≥50%，得 10/8/5/2/0。事件窗沿用保存的前后时长，覆盖不足时列出原因。")
            return extended[:3] + original[2:4] + [event] + extended[3:]
        return extended[:3] + original[2:] + extended[3:]
    return [
        ("数值完整性 · 20", "按缺失最多的一道：无缺失 20；缺失 <1% 得 15；1–<10% 得 10；10–<50% 得 5；≥50% 得 0。"),
        ("通道质量 · 30", "起始 30。最严重一道的异常窗：0 不扣；>0–10% 扣 5；>10–50% 扣 10；>50% 扣 15。至少 2 道各命中 ≥10% 时间再扣 5；这些道达到总道数 10% 再扣 5；同道高频和低相关重合 >1% 时间再扣 5。全段高频候选待定位时，时长项先扣 5。"),
        ("工频干扰 · 20", "起始 20。显著正弦扣 2；工频 RMS / 同窗 2–40 Hz RMS：<0.10 / 0.10–<0.25 / 0.25–<0.50 / 0.50–<1 / ≥1，扣 0/3/6/10/15；命中道数 ≥10%/25%/50% 扣 2/4/6；单道累计时间 ≥10%/50% 扣 2/4。每组取最高档，累计最多扣 20。"),
        ("PSD 可分析性 · 10", "2–40 Hz 覆盖完整且各道 PSD 有效得 10；部分频带或少量通道受限得 5；超过 10% 的通道可分析频带不足一半或 PSD 无效得 0。α 峰、谱背景及拟合残差用于解释状态。"),
        ("ECG、EOG · 各 10", "至少 20 个不重叠事件。奇偶、前后半锁时中位波形相关均 ≥0.50 时，取最严重一道：锁时中位波形峰峰值 / 单次峰峰值中位数 <0.10 / 0.10–<0.25 / 0.25–<0.50 / ≥0.50，得 10/8/5/2。重复性条件未命中得 10；事件不足或辅助导联缺失时记录未评分原因。"),
    ]
