"""Integer automatic scorecards, evidence and editable human adjustments."""
import csv
import json
from pathlib import Path

from .report import _text, _names
from .scoring import compute_quality_score
from .score_rules import rubric_rows
from .proportion_score import render_proportion_bands
from .header_qc import render_header_checks
from .spectral_summary import spectral_display_rows
from .spectral_findings import line_noise_conclusion
from .quality_overview import render_quality_overview
from .display import display_channel_name


def _spectrum_section(result, with_figures):
    rows = spectral_display_rows(result)
    if not rows:
        return []
    alpha = result["method_results"]["spectral"]["alpha_power"]
    lo, hi = alpha["alpha_band_hz"]
    rlo, rhi = alpha["reference_band_hz"]
    lines = ["", "## 关键频谱", "",
             f"α 功率为 {lo:g}–{hi:g} Hz 的 PSD 积分；α 占比为该功率占 {rlo:g}–{rhi:g} Hz 总功率的百分比。", "",
             "| 通道 | 选择依据 | α 功率 (µV²) | α 占比 | 拟合 α 峰 (Hz) |",
             "| --- | --- | ---: | ---: | ---: |"]
    def number(value, suffix=""):
        return "待评" if value is None else f"{value:.1f}{suffix}"
    for row in rows:
        peak = number(row["fitted_alpha_peak_hz"]) if row["fitted_alpha_peak_hz"] is not None else "未检出"
        lines.append(f"| {_text(row['display_name'])} | {row['selection_reason']} | "
                     f"{number(row['alpha_power_uv2'])} | {number(row['relative_alpha_percent'], '%')} | {peak} |")
    lines += ["", f"统计范围：{rows[0]['start_seconds']:g}–{rows[0]['stop_seconds']:g} s。"
              "人工核对：α 分布是否符合记录状态；短时突增是否与波形伪迹同时出现。", ""]
    links = ["[全部通道与逐窗数值](alpha_power.tsv)"]
    if with_figures:
        links += ["[PSD 与 α 功率](03_psd_alpha.png)", "[α 随时间变化](08_alpha_time.png)"]
    return lines + [" · ".join(links), ""]


def write_score_files(result, output):
    review = compute_quality_score(result)
    output = Path(output)
    (output/"quality_score.json").write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output/"quality_overview.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review["quality_overview"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(review["quality_overview"])
    fields = ["检查项", "满分", "自动得分", "扣分", "具体理由", "自动发现", "人工调整后得分", "调整理由",
              "审核人", "日期", "分析目的", "目标频带", "关键通道"]
    with (output/"score_review.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        for row in review["rows"]:
            writer.writerow([row["label"], row["max_points"],
                             row["auto_points"] if row["human_adjustable"] else "未评分",
                             row["auto_deduction"], row["score_reason"], row["finding"],
                             "" if row["human_adjustable"] else "—", "", "", "", "", "", ""])
        score = review["score_100"]
        writer.writerow(["整体质量", 100, score, 100-score if score is not None else "",
                         "可评项目按固定权重折算；检测覆盖 "
                         f"{review['assessed_max']}/100；人工调整后沿用相同折算公式", "", "", "", "", "", "", "", ""])
    with (output/"score_evidence.tsv").open("w", encoding="utf-8", newline="") as handle:
        evidence = list(review["evidence"])
        for row in review["rows"]:
            evidence.extend({"dimension_id": row["dimension_id"], **e} for e in row["score_evidence"])
        fields = ["dimension_id", "kind", "channel_id", "raw_name", "start_sample", "stop_sample", "start_seconds", "stop_seconds", "reason"]
        fields += sorted({k for e in evidence for k in e}-set(fields))
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(evidence)
    return review


def render_score_review(result, plot_config, *, with_figures=True):
    review = compute_quality_score(result)
    f, names = result["features"], _names(result["features"])
    score = review["score_100"]
    score_label = f"{score} / 100 分" if score is not None else "无可评项目"
    title = _text(plot_config.get("title", "EEG").removesuffix(" EEG QC"))
    lines = [f"# {title} · 自动评分与人工核对", "",
        f"{len(names)} 通道 · {f['sampling_rate_hz']:g} Hz · {f['start_seconds']:g}–{f['stop_seconds']:g} s", "",
        "分析目的：________　目标频带：________　关键通道：________", "",
        f"**自动初评：{score_label}**　检测覆盖：{review['assessed_max']}/100（按评分权重）。", "",
        *render_quality_overview(review),
        "审核人：________　日期：________　人工调整后合计：________", "",
        "| 检查项 | 满分 | 自动得分 | 具体理由 | 人工调整后得分 | 调整理由 |",
        "| --- | ---: | ---: | --- | --- | --- |"]
    for row in review["rows"]:
        score = "未评分" if row["auto_points"] is None else str(row["auto_points"])
        reason = row["score_reason"]
        if row["dimension_id"] in {"ecg", "eog"} and row["score_evidence"]:
            e = row["score_evidence"][0]
            reason += f"；代表事件 {e['time_seconds']:.3f} s"
        human = "________" if row["human_adjustable"] else "—"
        lines.append(f"| {row['label']} | {row['max_points']} | {score} | {_text(reason)} | {human} | ________ |")
    lines += ["", f"百分制初评 = {review['auto_total']} ÷ {review['assessed_max']} × 100，四舍五入为整数。"
              if review['assessed_max'] else "当前自动测量条件不足，覆盖原因见分项表。",
              "分项表保留固定权重。未评分项列出原因并从折算中排除；补充信号或检测条件后重新自动计算。"
              "人工核对已有初评，调整分项时填写理由，再按相同公式重算。", "",
              "比较记录时，对齐已评项目、检测参数、通道和记录条件。", ""]
    lines += ["**" + _text(line_noise_conclusion(result, review)) + "**", ""]
    localized = [r for r in review["rows"] if r["dimension_id"] in {"continuity", "muscle"} and r["score_evidence"]]
    if localized:
        lines += ["| 需要核对 | 通道 | 原始时间 | 人工意见 |", "| --- | --- | --- | --- |"]
        for row in localized:
            grouped = []
            for e in sorted(row["score_evidence"], key=lambda e: (e["kind"],e["start_sample"])):
                if grouped and grouped[-1]["kind"] == e["kind"] and e["start_seconds"] <= grouped[-1]["stop"]:
                    grouped[-1]["stop"] = max(grouped[-1]["stop"],e["stop_seconds"])
                    grouped[-1]["names"].append(e["raw_name"])
                else:
                    grouped.append({"kind":e["kind"], "reason":e["reason"], "start":e["start_seconds"],
                                    "stop":e["stop_seconds"], "names":[e["raw_name"]]})
            for group in grouped:
                unique = list(dict.fromkeys(group["names"]))
                display = "、".join(unique[:4]) + (f" 等 {len(unique)} 道" if len(unique)>4 else "")
                lines.append(f"| {group['reason']} | {_text(display)} | {group['start']:.3f}–{group['stop']:.3f} s | 保留 / 标记时间段 / 标坏道；原因：____ |")
        lines += ["", "定位与样本区间：[评分证据](score_evidence.tsv)。", ""]
    lines += _spectrum_section(result, with_figures)
    if review["rubric_version"] == "eeg-integer-score-7.0":
        lines += render_proportion_bands()
        lines += render_header_checks(result["method_results"].get("header"))
    lines += ["## 评分标准", "", "| 项目与满分 | 整数分档与扣分条件 |", "| --- | --- |"]
    lines += [f"| {label} | {rule} |" for label, rule in rubric_rows(review["rubric_version"])]
    lines += ["", "通道和工频项目要求有效检测覆盖至少 95% 的片段。各项满分共 100；评分规则 " + review["rubric_version"] + "。", ""]
    if review["coverage_gaps"]:
        grouped = {}
        labels = {r["dimension_id"]: r["label"] for r in review["rows"]}
        for gap in review["coverage_gaps"]:
            key = (gap["dimension_id"], gap["start_seconds"], gap["stop_seconds"], gap["reason"])
            grouped.setdefault(key, []).append(names[gap["channel_id"]])
        lines += ["", "**待核范围**", "", "| 检查 | 通道 | 时间 | 原因 |", "| --- | --- | --- | --- |"]
        for (dimension, start, stop, reason), channels in grouped.items():
            display = "、".join(channels[:4]) + (f" 等 {len(channels)} 道" if len(channels) > 4 else "")
            lines.append(f"| {labels[dimension]} | {display} | {start:g}–{stop:g} s | {_text(reason)} |")
        lines.append("")
    if review["channel_profiles"]:
        lines += ["", "## 候选范围", "",
                  "| 通道 | 特征 | 累计证据窗 | 最长连续证据窗 | 最长窗位置 |",
                  "| --- | --- | --- | --- | --- |"]
        for row in review["channel_profiles"]:
            if row["location_status"] == "whole_record_statistic_only":
                lines.append(f"| {names[row['channel_id']]} | {row['reason']} | 全段统计候选 | 定位待核 | 查看该道波形 |")
                continue
            longest = max(row["intervals"], key=lambda r: r["stop_sample"]-r["start_sample"])
            reason = row["reason"]
            if row["dimension_id"] == "line":
                reason = "/".join(f"{v:g}" for v in row["frequencies_hz"]) + " Hz"
                lo, hi = row["amplitude_range_uv"]
                reason += f"；{lo:.2f}–{hi:.2f} µV"
            lines.append(f"| {names[row['channel_id']]} | {_text(reason)} | {row['covered_seconds']:g} s | "
                         f"{row['longest_continuous_seconds']:g} s | {longest['start_seconds']:g}–{longest['stop_seconds']:g} s |")
        lines += ["", "时间按命中的分析窗合并；累计时长与最长连续时长分别查看。", ""]
    observations = review["observations_for_human"]
    if observations:
        lines += ["**补充核对：** " + "；".join(dict.fromkeys(
            f"{names[r['channel_id']]} {r['start_seconds']:g}–{r['stop_seconds']:g} s"
            for r in observations)) + " 的幅度或平台观测，结合记录状态、波形和频谱判断。", ""]
    spectrum = next(r for r in review["rows"] if r["dimension_id"] == "spectrum")
    lines += ["", "**频谱状态：** " + _text(spectrum["finding"]) + "。结合睁闭眼、睡眠或任务状态核对。", "",
              "[填写人工评分表](score_review.tsv) · [定位明细](score_evidence.tsv)", ""]
    if with_figures:
        labels = {"01_initial": "原始波形", "02_bad_channels": "坏道 PSD 与波形", "03_all_channels_psd": "全部通道 PSD",
                  "04_line_noise": "50/60 Hz", "03_psd_alpha": "PSD / α", "05_ecg": "ECG 上下对齐",
                  "06_eog": "EOG 上下对齐", "08_alpha_time": "α 随时间变化", "10_muscle": "EEG / EMG 同步"}
        keys = plot_config.get("figure_keys")
        if keys is None:
            keys = ["01_initial", "02_bad_channels", "04_line_noise", "03_psd_alpha"]
            keys += [f"{i:02d}_{k}" for i,k in enumerate(("ecg", "eog"),5)
                     if result["method_results"]["events"]["by_kind"][k]["selected_event_ids"]]
        links = [f"[{labels.get(k, '坏道平台复核')}]({k}.png)" for k in keys]
        lines += [" · ".join(links), ""]
    return display_channel_name("\n".join(lines))
