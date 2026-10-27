"""Integer review bands for raw continuity and EMG-supported high-frequency bursts."""
import copy
import numpy as np
from .impact import TIME_POLICY, PROPORTION_POINTS, proportion_points, percent, time_burden, time_points
from .event_impact import POLICY as EVENT_IMPACT_POLICY, score_event_impact


VERSION = "eeg-integer-score-7.0"
MAX_POINTS = {"header": 10, "integrity": 5, "continuity": 10, "channels": 20, "line": 20,
              "spectrum": 5, "ecg": 10, "eog": 10, "muscle": 10}
CHANNEL_POLICY = {"persistent_fraction": .10, "fraction_edges": [.02, .05, .10, .25, .50],
                  "points": PROPORTION_POINTS[20], "single_channel_min_points": 10,
                  "unlocalized_only_points": 19, "basis": "persistent_channels_over_total_channels"}
INTERVAL_POLICY = TIME_POLICY
KINDS = {"near_zero_flatline": "近零平线 / 掉线候选", "calibration_endpoint_plateau": "量程端点平台",
         "constant_value": "恒值平台", "high_frequency_burst": "宽带高频突发",
         "emg_supported_burst": "与 EMG 同步的高频突发"}


def _length(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged)


def _interval_score(result, candidates, label):
    f = result["features"]
    ids, names = f["channel_ids"], f["channel_names"]
    origin, stop, fs = f["start_sample"], f["stop_sample"], f["sampling_rate_hz"]
    per = {cid: [] for cid in ids}
    evidence = []
    for item in candidates:
        cid, a, b = item["channel_id"], item["start_sample"], item["stop_sample"]
        if cid not in ids or not origin <= a < b <= stop:
            raise ValueError("Scoring candidate must address the original channel and sample range")
        per[cid].append((a, b))
        evidence.append({"kind": item["kind"], "channel_id": cid, "raw_name": names[ids.index(cid)],
                         "start_sample": a, "stop_sample": b,
                         "start_seconds": f["start_seconds"]+(a-origin)/fs,
                         "stop_seconds": f["start_seconds"]+(b-origin)/fs,
                         "reason": KINDS.get(item["kind"], label)})
    fractions = {cid: _length(intervals)/(stop-origin) for cid, intervals in per.items()}
    worst = max(ids, key=fractions.get)
    affected = [cid for cid in ids if fractions[cid] > 0]
    burden = time_burden(f, [(e["start_sample"], e["stop_sample"]) for e in evidence])
    points = time_points(burden["fraction"])
    if not affected:
        reason = f"已检查范围内{label}未检出"
    else:
        kinds = "、".join(dict.fromkeys(KINDS.get(c["kind"], label) for c in candidates))
        reason = (f"{kinds}；异常区间合并 {burden['affected_seconds']:.2f} / {burden['total_seconds']:g} s，"
                  f"时间占比 {percent(burden['fraction'])}，时长档扣 {10-points} 分")
    return {"score": points, "reason": reason,
            "metrics": {"channel_fractions": fractions, "worst_channel_id": worst,
                        "affected_channel_ids": affected, "interval_basis": "union_in_original_samples",
                        "time_burden": burden},
            "components": [{"id": "time_burden", "deduction": 10-points}], "evidence": evidence}


def _channel_score(result, review, outcome):
    """Score persistent bad-channel prevalence, retaining per-channel evidence."""
    if outcome["score"] is None:
        return outcome
    f, cfg = result["features"], CHANNEL_POLICY
    ids, names = f["channel_ids"], dict(zip(f["channel_ids"], f["channel_names"]))
    metrics = copy.deepcopy(outcome["metrics"])
    fractions = metrics["channel_fractions"]
    persistent = [cid for cid in ids if fractions[cid] >= cfg["persistent_fraction"]]
    unlocalized = [p["channel_id"] for p in review.get("channel_profiles", [])
                   if p["dimension_id"] == "channels" and p["location_status"] == "whole_record_statistic_only"]
    count = len(persistent)
    fraction = count/len(ids)
    points = cfg["points"][0 if not count else 1+sum(fraction >= e for e in cfg["fraction_edges"])]
    if count == 1:
        points = max(points, cfg["single_channel_min_points"])
    if not count and unlocalized:
        points = cfg["unlocalized_only_points"]
    why = [f"异常通道 {count}/{len(ids)} 道，占比 {percent(fraction)}，通道档扣 {20-points} 分"]
    if unlocalized:
        why.append("、".join(names[cid] for cid in unlocalized)+" 全段高频离群、逐窗定位待核")
    metrics.update(persistent_channels=persistent, automatic_bad_channel_ids=persistent,
                   localized_channel_ids=[cid for cid in ids if 0 < fractions[cid] < cfg["persistent_fraction"]],
                   unlocalized_channel_ids=unlocalized,
                   channel_burden={"affected_channels": count, "total_channels": len(ids), "fraction": fraction})
    return {**outcome, "score": points, "metrics": metrics, "reason": "；".join(why),
            "components": [{"id": "channel_burden", "deduction": 20-points}]}


def extend_integer_scores(result, review, outcomes, policy):
    """Use the expanded 100-point rubric when both new method blocks are saved."""
    methods = result["method_results"]
    if not {"continuity", "muscle"} <= set(methods):
        return None, policy
    policy = copy.deepcopy(policy)
    policy["max_points"] = MAX_POINTS.copy()
    policy["integrity"] = {"fraction_edges": TIME_POLICY["fraction_edges"], "points": PROPORTION_POINTS[5]}
    policy["channels"] = copy.deepcopy(CHANNEL_POLICY)
    policy["continuity"] = copy.deepcopy(INTERVAL_POLICY)
    policy["muscle"] = copy.deepcopy(INTERVAL_POLICY)
    policy["muscle"]["requires_emg_reference"] = True
    policy["event_impact"] = {**copy.deepcopy(EVENT_IMPACT_POLICY), "time_policy": copy.deepcopy(TIME_POLICY)}
    missing = float(np.max((~np.isfinite(result["raw_data_volts"])).mean(axis=1)))
    outcomes["integrity"]["score"] = proportion_points(missing, 5)
    from .proportion_score import score_line_burden, score_spectrum_burden
    from .header_qc import score_header, CHECKS
    outcomes["header"] = score_header(methods.get("header"))
    policy["header"] = {"checks": CHECKS, "points": "sum of passed technical checks"}
    outcomes["line"] = score_line_burden(result, review, outcomes["line"])
    policy["line"] = {"fraction_edges": TIME_POLICY["fraction_edges"], "points": PROPORTION_POINTS[20],
                      "basis": "affected_channel_samples_over_all_channel_samples"}
    outcomes["spectrum"] = score_spectrum_burden(result, outcomes["spectrum"])
    policy["spectrum"].update(fraction_edges=TIME_POLICY["fraction_edges"], points=PROPORTION_POINTS[5])
    outcomes["channels"] = _channel_score(result, review, outcomes["channels"])
    continuity = [c for c in methods["continuity"]["candidates"] if c["kind"] in
                  {"near_zero_flatline", "calibration_endpoint_plateau", "constant_value"}]
    outcomes["continuity"] = _interval_score(result, continuity, "持续恒值")
    finite_coverage = float(np.min(np.isfinite(result["raw_data_volts"]).mean(axis=1)))
    if finite_coverage < .95:
        outcomes["continuity"] = {"score": None, "reason": "补齐缺失样本后核对连续性；至少一道有效数值覆盖不足 95%",
                                  "metrics": {"minimum_valid_fraction": finite_coverage}, "evidence": []}
    muscle = methods["muscle"]
    supported = [c for c in muscle["candidates"] if c["kind"] == "emg_supported_burst"]
    duration = result["raw_data_volts"].shape[1] / result["features"]["sampling_rate_hz"]
    coverage = min([c.get("valid_seconds", 0)/duration for c in muscle["channels"]]
                   + [muscle["reference"].get("valid_seconds", 0)/duration])
    if (muscle.get("status") != "ok" or muscle["reference"].get("status") != "ok"
            or coverage < .95 or "emg" not in result.get("references", {})):
        outcomes["muscle"] = {"score": None, "reason": "核对可用高频带宽、有效分析时长及同步 EMG 导联",
                              "metrics": {"status": muscle.get("status"), "minimum_valid_fraction": coverage}, "evidence": []}
    else:
        outcomes["muscle"] = _interval_score(result, supported, "与 EMG 同步的高频突发")
    for kind in ("ecg", "eog"):
        outcomes[kind] = score_event_impact(result, kind, outcomes[kind])
    for key, label in (("header", "头文件完整性"), ("continuity", "连续性与平台"), ("muscle", "肌电相关高频")):
        outcome = outcomes[key]
        review["rows"].append({"dimension_id": key, "label": label,
            "status": "已计算" if outcome["score"] is not None else "待评",
            "finding": outcome["reason"], "review_prompt": "对照原始波形与定位范围，核对整数分档",
            "human_grade": None, "human_reason": ""})
    review["rows"].sort(key=lambda r: list(MAX_POINTS).index(r["dimension_id"]))
    return VERSION, policy


def rubric_rows(version=VERSION):
    if version == VERSION:
        return [
            ("头文件完整性 · 10", "格式签名 1、头长度 1、通道表 1、采样率 2、单位与标定 2、数据长度 2、开始时间 1；逐项通过得分。匿名身份字段保留状态。"),
            ("数值完整性 · 5", "以缺失最多一道的缺失样本占比，使用统一 5 分档。"),
            ("通道质量 · 20", "累计异常 ≥10% 时间的通道计为坏道；坏道数 / EEG 总道数，使用统一 20 分档。单一坏道最低 10 分；仅有待定位全段异常得 19。"),
            ("短平线、平台、ECG、EOG、EMG · 各 10", "使用统一 10 分档。按原始样本合并全部异常时间，再除以本段时长。ECG/EOG 使用通过独立模板核对的事件窗；EMG 使用同步证据。"),
            ("工频干扰 · 20", "每道先合并工频命中窗；受影响的通道×样本数 / 全部通道×样本数，使用统一 20 分档。报告另列影响道数、时间与振幅比。"),
            ("PSD 可分析性 · 5", "2–40 Hz 频带覆盖不完整或 PSD 无效的通道 / EEG 总道数，使用统一 5 分档。α 功率与独立谱峰用于解释记录状态。"),
        ]
    if version == "eeg-integer-score-6.0":
        return [
            ("数值完整性 · 10", "缺失最多的一道：无缺失 10；<1% 得 8；1–<10% 得 5；10–<50% 得 2；≥50% 得 0。"),
            ("连续性与平台 · 10", "跨通道异常区间取时间并集，占本段时长：0 / >0–<1% / 1–<10% / 10–<50% / ≥50%，得 10/8/5/2/0。"),
            ("通道质量 · 20", "累计异常 ≥10% 时间的通道列为坏道。坏道占全部 EEG：0 / >0–<2% / 2–<5% / 5–<10% / 10–<25% / 25–<50% / ≥50%，得 20/18/16/12/8/4/0。单一坏道最低得 10；仅有待定位全段高频异常时得 18。"),
            ("肌电相关高频 · 10", "EMG 支持的异常区间跨通道合并计时，占本段时长：0 / >0–<1% / 1–<10% / 10–<50% / ≥50%，得 10/8/5/2/0。"),
        ]
    return [
        ("数值完整性 · 10", "缺失最多的一道：无缺失 10；<1% 得 8；1–<10% 得 5；10–<50% 得 2；≥50% 得 0。"),
        ("连续性与平台 · 10", "近零平线、量程端点平台、其他恒值分别定位后合并计时。最严重一道：无命中 10；>0–<1% 得 8；1–<10% 得 5；10–<50% 得 2；≥50% 得 0。至少 2 道且占总道数 ≥25% 再扣 2，最低 0。"),
        ("通道质量 · 20", "起始 20。范围档按累计异常 ≥10% 时间的通道计数：0 道扣 0；1 道扣 2；至少 2 道且占比 <10% / 10–<25% / ≥25%，扣 4/8/12。最严重一道异常时长：0 / >0–10% / >10–50% / >50%，扣 0/2/4/6。同道高频与低相关重合 >1% 时间，再扣 2。全段高频候选待定位时，时长项最低扣 2。三项相加；单一坏道最多扣 10。" if version == "eeg-integer-score-5.0" else
         "起始 20。异常窗：>0–10% 扣 5；>10–50% 扣 10；>50% 扣 15。至少 2 道各命中 ≥10% 时间再扣 5；这些道达到总道数 10% 再扣 5；同道高频和低相关重合 >1% 再扣 5。累计最多扣 20。"),
        ("肌电相关高频 · 10", "同步 EMG 可用时，按已通过宽带与 EMG 同步条件的候选计时：无命中 10；>0–<1% 得 8；1–<10% 得 5；10–<50% 得 2；≥50% 得 0。至少 2 道且占总道数 ≥25% 再扣 2，最低 0。EMG 或检测条件不足时记录未评分原因。"),
    ]
