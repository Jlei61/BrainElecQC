"""Algorithm-generated EEG report: decisions first, then localized PNG evidence."""
import re
from .display import display_channel_name, display_channel_names
from .report import _text
from .scoring import compute_quality_score
from .spectral_findings import alpha_conclusion, line_noise_conclusion, suspicious_channel_conclusion
from .quality_overview import render_quality_overview
from .header_qc import render_header_checks


def _location(review, row):
    evidence = row["score_evidence"] or [e for e in review["evidence"]
                                        if e["dimension_id"] == row["dimension_id"]]
    worst = row["score_metrics"].get("worst_channel_id")
    selected = [e for e in evidence if e.get("channel_id") == worst] or evidence
    if not selected:
        return ""
    if row["dimension_id"] == "line":
        item = max(selected, key=lambda e: e.get("line_to_band_rms", 0))
    else:
        item = max(selected, key=lambda e: e.get("stop_sample", 0)-e.get("start_sample", 0))
    name = _text(display_channel_name(item.get("raw_name", "")))
    if "time_seconds" in item:
        return f"代表事件：{name}，{item['time_seconds']:.3f} s。"
    if "start_seconds" in item and "stop_seconds" in item:
        return f"典型位置：{name}，{item['start_seconds']:.3f}–{item['stop_seconds']:.3f} s。"
    return ""


def _unavailable_reason(review, row, result):
    if row["dimension_id"] == "muscle" and "emg" not in result.get("references", {}):
        return "缺少同步 EMG 导联"
    if row["dimension_id"] in {"channels", "line"}:
        reasons = list(dict.fromkeys(g["reason"] for g in review["coverage_gaps"]
                                    if g["dimension_id"] == row["dimension_id"]))
        if reasons:
            return "；".join(reasons)
    return row["score_reason"]


def _brief_reason(reason):
    reason = re.sub(r"；本项累计最多扣 \d+ 分|；自动 \d+/\d+ 分", "", reason)
    return re.sub(r"[，,]?(?:通道档|范围档|时长档|证据重合|再)?扣 \d+ 分", "", reason).strip("；， ")


def _bad_channel_findings(review, row):
    """Lead with automatic channel decisions, then their measured intervals."""
    ids = review["channel_ids"]
    names = dict(zip(ids, display_channel_names(review["channel_names"])))
    def labels(selected):
        return "、".join(_text(names[cid]) for cid in selected)
    if row["dimension_id"] == "continuity":
        evidence = row["score_evidence"]
        if not evidence:
            return ["**算法自动检测：未检出短平线或平台。**", ""] if row["auto_points"] is not None else []
        affected = [cid for cid in ids if any(e["channel_id"] == cid for e in evidence)]
        scope = f"全部 {len(ids)} 道" if len(affected) == len(ids) else labels(affected)
        kinds = "、".join(dict.fromkeys(e["reason"] for e in evidence))
        return [f"**算法自动检测：{scope} 存在局部异常（{kinds}）。**", "",
                _location(review, row), ""]
    profiles = [p for p in review["channel_profiles"] if p["dimension_id"] == "channels"]
    fractions = row["score_metrics"].get("channel_fractions", {})
    threshold = review["scoring_policy"]["channels"]["persistent_fraction"]
    persistent = sorted((cid for cid in ids if fractions.get(cid, 0) >= threshold),
                        key=lambda cid: -fractions[cid])
    unlocalized = list(dict.fromkeys(p["channel_id"] for p in profiles if not p["intervals"]))
    localized = [cid for cid in ids if cid not in persistent and any(p["channel_id"] == cid and p["intervals"] for p in profiles)]
    findings = []
    if persistent:
        findings.append(f"坏道 {labels(persistent)}（{len(persistent)}/{len(ids)} 道）")
    if localized:
        findings.append(f"局部异常 {labels(localized)}")
    if unlocalized:
        findings.append(f"全段高频异常 {labels(unlocalized)}（时段待定位）")
    if not findings:
        findings.append("未检出坏道" if row["auto_points"] is not None else "通道统计条件不足，见下方原因")
    lines = ["**算法自动检测："+"；".join(findings)+"。**", ""]
    for cid in dict.fromkeys(persistent+localized+unlocalized):
        for p in (p for p in profiles if p["channel_id"] == cid):
            if p["intervals"]:
                interval = max(p["intervals"], key=lambda i: i["stop_sample"]-i["start_sample"])
                detail = (f"累计 {p['covered_seconds']:g} s，最长连续 {p['longest_continuous_seconds']:g} s；"
                          f"典型时段 {interval['start_seconds']:g}–{interval['stop_seconds']:g} s")
            else:
                detail = "全段统计命中，逐窗定位待核"
            lines.append(f"- {labels([cid])}：{p['reason']}；{detail}。")
    return lines+[""]


def render_automatic_report(result, plot_config, *, with_figures=True):
    """Render the same report for Python, CLI, Linxi and saved-result redraw."""
    review = compute_quality_score(result)
    f = result["features"]
    title = _text(plot_config.get("title", "EEG").removesuffix(" EEG QC"))
    score = review["score_100"]
    score_label = f"{score} / 100" if score is not None else "无可评项目"
    coverage = review["assessment_coverage"]
    heading = "EEG 自动 QC 报告" if title in {"EEG", "EEG QC"} else f"{title} · EEG 自动 QC 报告"
    channels = []
    if len(f["channel_names"]) <= 8:
        channels += ["本次 EEG：" + "、".join(_text(display_channel_name(n)) for n in f["channel_names"]) + "。", ""]
    auxiliary = f.get("reference_channels", {})
    if auxiliary:
        channels += ["本次辅助导联：" + "、".join(
            _text(display_channel_name(row["raw_name"])) for row in auxiliary.values()) + "。", ""]
    lines = [f"# {heading}", "",
             f"{len(f['channel_ids'])} 道 EEG · {f['sampling_rate_hz']:g} Hz · "
             f"本次分析范围 {f['start_seconds']:g}–{f['stop_seconds']:g} s", "",
             f"**自动初评：{score_label}**", "",
             *render_quality_overview(review),
             *channels,
             f"检测覆盖：{len(coverage['evaluated_dimensions'])}/{len(review['rows'])} 项，"
             f"覆盖评分权重 {coverage['evaluated_weight']}/{coverage['total_weight']}。"
             "初评按可评项目的固定权重折算为整数百分制。", "",
             "[查看评分、扣分标准与人工核对表](score_report.md)", "",
             "## 主要扣分项", ""]
    rows = {r["dimension_id"]: r for r in review["rows"]}
    if review["main_deductions"]:
        for item in review["main_deductions"]:
            row = rows[item["dimension_id"]]
            lines += [f"- **{item['label']}：{item['points']}/{item['weight']}，分项扣 {item['deduction']} 分。** "
                      f"{_text(display_channel_name(_brief_reason(item['reason'])))}。{_location(review, row)}"]
    else:
        lines += ["已评项目均达到本次自动评分标准。"]
    unavailable = [r for r in review["rows"] if not r["human_adjustable"]]
    if unavailable:
        lines += ["", "本次未计入初评的检查：" + "；".join(
            f"{r['label']}（{_text(_unavailable_reason(review, r, result))}）" for r in unavailable) + "。"]
    if any(display_channel_name(n) != n for n in f["channel_names"]):
        lines += ["", "图中 **-Ref** 表示原文件中的采集参考；原始名称与具体参考分组保存在通道记录中。"]
    lines += ["", *render_header_checks(result["method_results"].get("header"))]
    if with_figures:
        keys = plot_config.get("figure_keys", [])
        sections = [
            ("原始波形", [k for k in keys if k == "01_initial"], None),
            ("PSD 与 α 能量", [k for k in keys if k == "03_psd_alpha"], "spectrum"),
            ("全通道频谱", [k for k in keys if k == "03_all_channels_psd"], None),
            ("坏道：高频与相关", [k for k in keys if k == "02_bad_channels" or k.startswith("02_bad_channels_comparison_")], "channels"),
            ("坏道：短平线与平台", [k for k in keys if k.startswith("02_bad_channels_") and not k.startswith("02_bad_channels_comparison_")], "continuity"),
            ("工频检测", [k for k in keys if k == "04_line_noise"], "line"),
            ("心电事件锁时", [k for k in keys if k == "05_ecg"], "ecg"),
            ("眼电事件锁时", [k for k in keys if k == "06_eog"], "eog"),
            ("肌电相关高频", [k for k in keys if k == "10_muscle"], "muscle"),
        ]
        for heading, selected, dimension in sections:
            if not selected:
                continue
            lines += ["", f"## {heading}", ""]
            if dimension:
                row = rows.get(dimension)
                if row:
                    if dimension == "line":
                        lines += ["**"+_text(line_noise_conclusion(result, review))+"**", ""]
                    if dimension in {"channels", "continuity"}:
                        lines += _bad_channel_findings(review, row)
                    if row["human_adjustable"]:
                        reason = row["score_reason"] if dimension == "channels" else _brief_reason(row["score_reason"])
                        lines += [f"算法自动评分：**{row['auto_points']}/{row['max_points']}**。"
                                  + (_text(display_channel_name(reason)) if dimension != "line" else ""), ""]
                    else:
                        lines += [_text(_unavailable_reason(review, row, result)) + "。", ""]
                if dimension in {"ecg", "eog"}:
                    event_name = "心搏" if dimension == "ecg" else "眼动"
                    lines += [f"核对下方{event_name}发生时，上方 EEG 是否反复出现同步波形；红线为代表事件，橙线为邻近事件。", ""]
            for key in selected:
                lines += [f"![{heading}]({key}.png)", ""]
                if key == "03_all_channels_psd":
                    lines += ["- **α：** "+_text(alpha_conclusion(result)),
                              "- **工频：** "+_text(line_noise_conclusion(result, review)),
                              "- **通道检查：** "+_text(suspicious_channel_conclusion(result, review)), ""]
                lines += ["[对应评分报告](score_report.md)", ""]
        if "08_alpha_time" in keys:
            alpha = result["method_results"]["spectral"].get("alpha_power")
            window_text = (f"每个点代表 {alpha['config']['window_seconds']:g} 秒 EEG，每 {alpha['config']['step_seconds']:g} 秒更新一次"
                           if alpha else "每个点代表一个 EEG 滑动窗口，窗长与步长见图标题")
            band_text = (f"{alpha['alpha_band_hz'][0]:g}–{alpha['alpha_band_hz'][1]:g} Hz" if alpha else "α 频带")
            reference_text = (f"{alpha['reference_band_hz'][0]:g}–{alpha['reference_band_hz'][1]:g} Hz" if alpha else "参考频带")
            lines += ["", "## 补充查看", "",
                      "[α 能量随时间变化](08_alpha_time.png)", "",
                      window_text+"，横轴是窗口中心时间。"
                      f"上图是 {band_text} 的总功率（µV²，对数刻度）；下图是它占 {reference_text} 总功率的比例。", "",
                      "两条曲线一起升高，说明 α 频带能量相对增强，结合 PSD 中的独立 α 峰判断节律；"
                      "只有功率升高而占比平稳，提示更宽频带的能量一起升高。"
                      "先找变化发生的时间，再对照该时段原始波形、坏道与伪迹标记，以及睁闭眼/睡眠状态。"
                      "α 大小用来描述状态变化，评分依据频谱可分析性和具体异常证据。",
                      "", "[对应评分报告](score_report.md)", ""]
    lines += ["", "## 人工核对", "",
              "核对候选通道和时间段是否与波形一致；需要调整时，在评分表中填写分项得分、理由、审核人和日期。",
              "", "[打开评分与核对表](score_report.md) · [填写核对表](score_review.tsv)", "",
              "| 最终评价 | 人工填写 |", "| --- | --- |",
              "| 分析用途与关键通道 | 待填写 |",
              "| 人工复评分及调整理由 | 待填写 |",
              "| 保留 / 排除的通道和时间范围 | 待填写 |",
              "| 结论 | 待评：可用 / 局部处理后可用 / 重点处理后复核 / 可用性受限 |",
              "| 审核人、日期 | 待填写 |", ""]
    return "\n".join(lines)
