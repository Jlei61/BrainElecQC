"""Concise human review documents accompanying the three EEG evidence figures."""
from collections import defaultdict
from html import escape


REVIEW_ACTIONS = {
    "file_endpoint_plateau": ("文件端点平台", "核对设备量程、导出缩放和平台前后波形"),
    "flatline": ("低变化平台", "核对接触状态、通道类型和数字分辨率"),
    "eog": ("眼电相关", "定位眼动或眨眼，核对 EEG 的锁时形态"),
    "ecg": ("心电相关", "定位心搏，核对 EEG 的重复锁时形态"),
    "emg": ("肌电相关", "对照 EMG，核对高频包络与波形"),
    "line_50": ("50 Hz 邻域", "查看局部 PSD 的峰频、峰宽与跨窗重复性"),
    "line_60": ("60 Hz 邻域", "查看局部 PSD 的峰频、峰宽与跨窗重复性"),
    "amplitude": ("幅值 / 漂移", "核对漂移、体动及生理波形"),
    "high_frequency": ("高频占比", "结合 EMG 与记录状态判断高频活动"),
    "low_frequency": ("低频占比", "结合记录状态判断慢波与漂移"),
}


def _text(value):
    return escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def _names(features):
    names = features["channel_names"]
    return {cid: _text(f"{name} [{index}]" if names.count(name) > 1 else name)
            for index, (cid, name) in enumerate(zip(features["channel_ids"], names))}


def _time(item):
    return f"{item['start_seconds']:.3f}–{item['stop_seconds']:.3f} s"


def _evidence(item):
    evidence = item.get("evidence", {})
    labels = {"duration_seconds": ("平台时长", "s"), "boundary_uv": ("端点", "µV"),
              "range_uv": ("极差", "µV"), "rms_uv": ("AC RMS", "µV"),
              "relative_rms": ("相对幅度", "倍"), "prominence_db": ("邻域峰值比", "dB"),
              "ecg_correlation": ("ECG r", ""), "eog_correlation": ("EOG r", ""),
              "emg_envelope_correlation": ("EMG 包络 r", ""),
              "large_amplitude_fraction": ("大幅度样本比例", ""),
              "high_frequency_fraction": ("高频占比", "")}
    return "；".join(f"{label} {evidence[key]:.3g} {unit}".strip()
                     for key, (label, unit) in labels.items() if evidence.get(key) is not None) or "见原始波形"


def _groups(decisions):
    groups = defaultdict(list)
    for item in decisions["review_queue"]:
        groups[item["kind"]].append(item)
    return sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))


def render_eeg_review_report(features, decisions, plot_config, *, with_figures=True):
    """Main report: image, location and review action; computations stay in JSON."""
    names, summary = _names(features), decisions["summary"]
    auto = decisions["automatic_exclusions"]
    start, fs = features["start_seconds"], features["sampling_rate_hz"]
    end = start+(features["stop_sample"]-features["start_sample"])/fs
    groups = _groups(decisions)
    lines = [f"# {_text(plot_config.get('title', 'EEG'))} · 复核报告", "",
             f"{len(names)} 通道 · {fs:g} Hz · 分析 {start:g}–{end:g} s · 待复核", "",
             (f"**本次待办：核对 {len(auto)} 处自动剔除，复核 {len(groups)} 类候选。**" if auto else
              f"**本次保留全部样本，复核 {len(groups)} 类候选后记录处理决定。**"), "",
             "## 1. 原始 EEG", ""]
    if with_figures:
        lines += ["![原始 EEG](01_initial.png)", ""]
    if auto:
        affected = list(dict.fromkeys(names[x["channel_id"]] for x in auto))
        lines += [f"- [ ] 查看 {'、'.join(affected)} 的异常形态与前后边界（具体区间见第 3 节）。", ""]
    else:
        lines += ["- [ ] 查看整体幅度、基线与短时异常。", ""]
    lines += ["## 2. 自动特征与待复核内容", ""]
    if with_figures:
        lines += ["![自动特征](02_features_review.png)", ""]
    if groups:
        lines += ["| 待核对 | 项数 | 定位示例 | 复核动作 |",
                  "| --- | ---: | --- | --- |"]
        for kind, items in groups:
            label, action = REVIEW_ACTIONS.get(kind, (kind, items[0].get("reason", kind)))
            first = min(items, key=lambda x: (x["start_sample"], x["channel_id"]))
            lines.append(f"| {_text(label)} | {len(items)} | {names[first['channel_id']]} {_time(first)} | {_text(action)} |")
        lines += ["", f"共 {len(decisions['review_queue'])} 项，按“通道 × 时间窗 × 原因”计数。"
                  "逐项位置见 [完整复核清单](review_items.md)。", ""]
    else:
        lines += ["本次候选清单为空。", ""]
    refs = features.get("reference_channels", {})
    if refs:
        lines += ["参考通道：" + "；".join(f"{k.upper()} = {_text(v['raw_name'])}" for k, v in refs.items()) + "。", ""]
    if features.get("measurement_methods"):
        cfg = features["config"]
        lines += ["**图中指标怎样读**", "",
                  f"- 50/60 是两组待查频率。分数 = 10 log10（±{cfg['line_peak_half_width_hz']:g} Hz 内最高 PSD / "
                  f"两侧 {cfg['line_baseline_inner_hz']:g}–{cfg['line_baseline_outer_hz']:g} Hz 背景中位数）；分数与逐窗定位见方法报告。图 b 放大对应 PSD。",
                  "- PSD：横轴为 Hz，纵轴为 dB(µV²/Hz)。先看全频谱，再放大窄峰，核对同类通道及记录状态。",
                  f"- 图 c 放大节律及周围 PSD，灰色区域为 {cfg['alpha_band'][0]:g}–{cfg['alpha_band'][1]:g} Hz；核对该频带是否形成高于周围背景的峰，再结合睡眠/睁闭眼状态判断。",
                  "- 参考图为各窗最大 |r|；回到对应窗口核对有符号 r、同步波形和事件重复性。",
                  "", "[计算定义、论文对应关系及本次数值](methods.md)", ""]
    lines += ["- [ ] 结合睡眠、睁闭眼或任务状态，核对局部谱峰与整体 PSD。",
              "- [ ] 对候选选择 **保留 / 剔除 / 调整区间**，记录原因。", "",
              "## 3. " + ("自动剔除结果" if auto else "待审核数据（全部保留）"), ""]
    if with_figures:
        lines += ["![当前数据与原始对照](03_auto_excluded.png)", ""]
    lines += [f"**保留 {100*(1-summary['excluded_fraction']):.2f}%**。蓝色为保留信号，灰色为原始对照。", ""]
    if auto:
        lines += ["| 编号 | 通道 | 自动剔除区间 | 原因 | 复核：接受 / 调整 / 恢复 |",
                  "| --- | --- | --- | --- | --- |"]
        for index, item in enumerate(auto, 1):
            lines.append(f"| A{index:02d} | {names[item['channel_id']]} | {_time(item)} | {_text(item['reason'])} | 待复核 |")
        lines += [""]
    lines += ["- [ ] 记录每项候选的决定、依据和实际处理范围。", "",
              "复核人：________　日期：________　备注：________", ""]
    return "\n".join(lines)


def render_eeg_review_items(features, decisions):
    """Keep each candidate separately addressable instead of losing windows in a summary."""
    names = _names(features)
    lines = ["# EEG 逐项复核清单", "", "[返回三图报告](review_report.md)", "",
             "每行选择保留、剔除或调整区间，并记录原因。", ""]
    number = 0
    for kind, items in _groups(decisions):
        label, action = REVIEW_ACTIONS.get(kind, (kind, items[0].get("reason", kind)))
        lines += [f"## {_text(label)}（{len(items)} 项）", "", _text(action), "",
                  "| 编号 | 通道 | 原始时间 | 测量证据 | 决定 |", "| --- | --- | --- | --- | --- |"]
        for item in sorted(items, key=lambda x: (x["start_sample"], x["channel_id"])):
            number += 1
            identifier = item.get("candidate_id", f"R{number:03d}")
            evidence = _evidence(item)
            lines.append(f"| {identifier} | {names[item['channel_id']]} | {_time(item)} | {evidence} | 待复核 |")
        lines += [""]
    return "\n".join(lines)
