"""Readable conclusions from saved spectral measurements and channel evidence."""
from .display import display_channel_names
from .spectral_summary import occipital_name


def _channel_names(result):
    features = result["features"]
    return dict(zip(features["channel_ids"], display_channel_names(features["channel_names"])))


def line_noise_conclusion(result, review):
    """Report detected frequencies; an empty detection never assigns a mains frequency."""
    spectral = result["method_results"]["spectral"]
    observations = spectral["line_noise"]["observations"]
    fundamentals = set(spectral.get("config", {}).get("line_fundamentals_hz", (50, 60)))
    row = next(r for r in review["rows"] if r["dimension_id"] == "line")
    names = _channel_names(result)
    hits = {}
    for observation in observations:
        if observation["flagged"]:
            hits.setdefault(observation["nominal_hz"], set()).add(observation["channel_id"])
    direct = sorted(fundamentals & hits.keys())
    harmonics = sorted(hits.keys()-fundamentals)
    frequencies = lambda values: "/".join(f"{hz:g}" for hz in values)
    if not hits:
        if row["auto_points"] is not None:
            return f"工频检测合格：未检出显著 {frequencies(sorted(fundamentals))} Hz 干扰。"
        return "工频检测覆盖不足：已检查片段未检出显著干扰。"
    parts = []
    if direct:
        parts.append(f"检出 {frequencies(direct)} Hz 工频干扰")
    if harmonics:
        parts.append(f"谐波检查检出 {frequencies(harmonics)} Hz 窄带干扰")
    for hz in sorted(hits):
        affected = [cid for cid in names if cid in hits[hz]]
        channels = "、".join(names[cid] for cid in affected[:6])
        if len(affected) == len(names):
            scope = f"全部 {len(names)} 道"
        else:
            scope = f"{channels} 等 {len(affected)}/{len(names)} 道" if len(affected) > 6 else f"{channels}（{len(affected)}/{len(names)} 道）"
        parts.append(f"{hz:g} Hz 影响 {scope}")
    if row["auto_points"] is None:
        parts.append("其余检测范围见评分报告")
    return "；".join(parts)+"。"


def alpha_conclusion(result):
    """Summarize measured peaks and named occipital power without inferring eye state."""
    spectral = result["method_results"]["spectral"]
    fits = spectral["parameterization"]["full"]
    measured = [r for r in fits if r["status"] == "computed"]
    peaks = [r for r in measured if r["alpha_peaks"]]
    total = len(result["features"]["channel_ids"])
    if peaks:
        opening = f"{len(peaks)}/{total} 道检出拟合 α 峰。"
    elif measured:
        opening = f"已完成拟合的 {len(measured)}/{total} 道未检出独立 α 峰。"
    else:
        opening = "α 峰拟合条件不足。"
    if measured and len(measured) < total:
        opening += f"谱拟合覆盖 {len(measured)}/{total} 道。"
    block = spectral.get("alpha_power")
    if block is None:
        return opening
    names = _channel_names(result)
    by_id = {r["channel_id"]: r for r in fits}
    valid = [r for r in block["full"] if r["alpha_power_status"] == "computed"
             and r["relative_alpha_status"] == "computed"]
    selected = [r for r in valid if occipital_name(r["channel_name"])]
    label = "枕区" if selected else "按原始通道顺序"
    selected = selected[:3] if selected else valid[:3]
    if not selected:
        return opening+"α 能量统计条件不足。"
    details = []
    for item in selected:
        fit = by_id[item["channel_id"]]
        peak = max(fit["alpha_peaks"], key=lambda p: p["height_db"]) if fit["status"] == "computed" and fit["alpha_peaks"] else None
        peak_text = f"主峰 {peak['center_frequency_hz']:.1f} Hz，" if peak else ""
        details.append(f"{names[item['channel_id']]}：{peak_text}α 功率 {item['alpha_power_uv2']:.1f} µV²，"
                       f"占比 {item['relative_alpha_percent']:.1f}%")
    first = selected[0]
    return opening+f"{label}（{first['start_seconds']:g}–{first['stop_seconds']:g} s）："+"；".join(details)+"。"


def suspicious_channel_conclusion(result, review):
    names = _channel_names(result)
    row = next(r for r in review["rows"] if r["dimension_id"] == "channels")
    profiles = [p for p in review["channel_profiles"] if p["dimension_id"] == "channels"]
    fractions = row["score_metrics"].get("channel_fractions", {})
    threshold = review["scoring_policy"]["channels"]["persistent_fraction"]
    persistent = sorted((cid for cid in names if fractions.get(cid, 0) >= threshold), key=lambda cid: -fractions[cid])
    local = [cid for cid in names if cid not in persistent and any(p["channel_id"] == cid and p["intervals"] for p in profiles)]
    unlocalized = list(dict.fromkeys(p["channel_id"] for p in profiles if not p["intervals"]))
    def details(ids):
        return "、".join(names[cid]+"（"+"、".join(dict.fromkeys(p["reason"] for p in profiles if p["channel_id"] == cid))+"）" for cid in ids)
    parts = []
    if persistent:
        parts.append("可疑坏道："+details(persistent))
    if local:
        parts.append("局部异常："+details(local))
    if unlocalized:
        parts.append("待定位高频异常："+"、".join(names[cid] for cid in unlocalized))
    if not parts:
        parts.append("未检出持续性通道异常" if row["auto_points"] is not None else "比较组统计条件不足，局部异常见短平线与平台检测")
    return "；".join(parts)+"。"
