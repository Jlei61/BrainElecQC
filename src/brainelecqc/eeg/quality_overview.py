"""The same channel/time summary for machine output and the front of reports."""
from .display import display_channel_names
from .impact import percent, time_burden


def attach_quality_overview(review, result):
    features = result["features"]
    names = dict(zip(features["channel_ids"], display_channel_names(features["channel_names"])))
    rows = {r["dimension_id"]: r for r in review["rows"]}
    overview = []
    labels = {"header": "头文件完整性", "channels": "坏道", "continuity": "短平线与平台", "ecg": "ECG 影响", "eog": "EOG 影响",
              "muscle": "EMG 影响", "line": "工频干扰", "spectrum": "PSD 可分析性", "integrity": "数值完整性"}
    for key in labels:
        if key not in rows:
            continue
        row, details = rows[key], ""
        metrics = row["score_metrics"]
        entry = {"dimension_id": key, "label": labels[key], "auto_points": row["auto_points"],
                 "max_points": row["max_points"], "metric": "", "fraction": None,
                 "numerator": None, "denominator": None, "unit": None, "basis": None}
        if row["auto_points"] is None:
            entry.update(metric="未评分", detail=row["score_reason"])
            overview.append(entry)
            continue
        if key == "header":
            entry.update(metric="技术字段核对", basis="original_file_header_checks")
            details = row["score_reason"]
        elif key == "channels":
            affected = metrics.get("persistent_channels", [])
            count, total = len(affected), len(names)
            entry.update(metric="异常通道占比", numerator=count, denominator=total, unit="道",
                         fraction=count/total, basis="persistent_channels_over_total_channels")
            details = "、".join(names[cid] for cid in affected[:5])
            if len(affected) > 5:
                details += f" 等 {count} 道"
            details = details or "未检出持续异常通道"
            if metrics.get("unlocalized_channel_ids"):
                details += "；有全段高频异常待定位"
        elif key in {"continuity", "muscle", "ecg", "eog", "line"}:
            burden = metrics.get("time_burden")
            if burden is None and key == "line":
                burden = time_burden(features, [(e["start_sample"], e["stop_sample"])
                                               for e in review["evidence"] if e["dimension_id"] == "line"])
            if burden is not None:
                entry.update(metric="异常事件窗占比" if key in {"ecg", "eog"} else "异常时间占比",
                             numerator=burden["affected_seconds"], denominator=burden["total_seconds"],
                             fraction=burden["fraction"], unit="s", basis=burden["basis"])
            if key in {"ecg", "eog"}:
                if burden is None:
                    entry["metric"] = "锁时振幅与重复性"
                    details = row["score_reason"]
                else:
                    details = f"{metrics['impact_affected_event_count']}/{metrics['impact_complete_event_count']} 个事件窗有影响"
                    offsets = metrics.get("impact_epoch_offsets_samples")
                    if offsets:
                        details += f"；事件窗 {offsets[0]/features['sampling_rate_hz']:+g}～{(offsets[1]-1)/features['sampling_rate_hz']:+g} s"
            elif key == "line":
                frequencies = sorted({e["nominal_hz"] for e in review["evidence"] if e["dimension_id"] == "line"})
                area = metrics.get("channel_time_burden")
                details = ("/".join(f"{hz:g}" for hz in frequencies)+" Hz") if frequencies else "工频检测合格"
                if area:
                    entry.update(metric="异常通道×时间占比", fraction=area["fraction"],
                        numerator=area["affected_channel_samples"]/features["sampling_rate_hz"],
                        denominator=area["total_channel_samples"]/features["sampling_rate_hz"], unit="道·s", basis=area["basis"])
                    details += f"；{area['affected_channels']}/{area['total_channels']} 道，时间并集 {percent(burden['fraction'])}"
            else:
                affected = metrics.get("affected_channel_ids", [])
                details = f"涉及 {len(affected)}/{len(names)} 道" if affected else "未检出对应异常"
        elif key == "integrity":
            entry.update(metric="最大单道缺失占比", fraction=metrics["fraction"],
                         numerator=metrics["missing_samples"], denominator=features["stop_sample"]-features["start_sample"],
                         unit="样本", basis="worst_channel_missing_samples")
            details = row["score_reason"]
        elif key == "spectrum":
            entry.update(metric="频谱受限通道占比", fraction=metrics.get("limited_channel_fraction"), basis="limited_channels_over_total_channels")
            details = row["score_reason"]
        entry["detail"] = details
        overview.append(entry)
    review["quality_overview_schema_version"] = "eeg-quality-overview-1.0"
    review["quality_overview"] = overview
    return review


def render_quality_overview(review):
    from .report import _text
    actions = {
        "header": "核对采样率、单位、标定及样本长度与采集导出信息",
        "channels": "对照原始波形、同组通道与频谱，确认整道或局部异常",
        "continuity": "核对平台前后及恢复位置，确定保留或排除区间",
        "ecg": "核对心搏检出及 EEG 同步波形，确认影响事件窗",
        "eog": "核对眼动检出及 EEG 同步波形，确认影响事件窗",
        "muscle": "比较 EEG 高频与 EMG 同步变化，核对有效带宽",
        "line": "查看实际峰频、强度及持续范围，确认受影响通道",
        "spectrum": "结合记录状态核对谱形、α 峰与关键通道",
        "integrity": "核对缺失样本、通道与原始时间范围",
    }
    lines = ["## 质量总结", "", "| 检查项 | 核心指标 | 算法自动评分 | 主要结果 | 人工复核 |",
             "| --- | --- | ---: | --- | --- |"]
    for entry in review["quality_overview"]:
        metric = entry["metric"]
        if entry["fraction"] is not None:
            metric += " **"+percent(entry["fraction"])+"**"
            if entry["numerator"] is not None:
                metric += f"（{entry['numerator']:.2f} / {entry['denominator']:g} {entry['unit']}）" if entry['unit'] in {'s', '道·s'} else f"（{entry['numerator']:g} / {entry['denominator']:g} {entry['unit']}）"
        score = "未评分" if entry["auto_points"] is None else f"{entry['auto_points']}/{entry['max_points']}"
        action = actions.get(entry["dimension_id"], "核对测量与原始证据")
        if entry["auto_points"] is None:
            action = "核对未评分原因，补足条件后重新计算"
        lines.append(f"| {entry['label']} | {metric} | **{score}** | {_text(entry['detail'])} | {action} |")
    lines += ["", "时间占比按本段总时长计算；跨通道、重复和重叠区间合并后计时。ECG/EOG 按表中事件窗计时。", ""]
    return lines
