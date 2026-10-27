"""Measurement definitions and the distinct role of each cited source."""
from copy import deepcopy
from html import escape
from statistics import median


def _label(value):
    return escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def _number(value):
    return "待评估" if value is None else f"{value:.3f}"


def _measured_tables(features, decisions):
    lines = ["## 本次观测值与定位", "", "### 50/60 Hz：全程谱与逐窗统计", "",
             "逐窗超阈比例使用本次 policy_config.review_line_db；属于候选排序统计。", "",
             "| 通道 | 待查 Hz | 全程峰频 Hz | 全程比值 dB | 各窗中位数 dB | 超阈窗 / 有效窗 | 最高窗 dB 与时间 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for ci, channel in enumerate(features["channel_features"]):
        for hz in (50, 60):
            values = [(i, v) for i, v in enumerate(features["window_features"][f"line_{hz}_db"][ci]) if v is not None]
            peak = channel.get(f"line_{hz}", {})
            maximum = "待评估"
            if values:
                wi, value = max(values, key=lambda pair: pair[1])
                window = features["windows"][wi]
                maximum = f"{value:.3f} @ {window['start_seconds']:g}–{window['stop_seconds']:g} s"
            count = sum(v >= decisions["policy_config"]["review_line_db"] for _, v in values)
            lines.append(f"| {_label(channel['channel_name'])} | {hz} | {_number(peak.get('peak_hz'))} | "
                         f"{_number(peak.get('prominence_db'))} | {_number(median([v for _, v in values]) if values else None)} | "
                         f"{count}/{len(values)} | {maximum} |")
    lines += ["", "### 辅助参考：图中最大 |r| 对应哪个窗口", "",
              "数值保留正负号。相同最大值取最早窗口；完整逐窗数值保存在 features.json。", "",
              "| EEG 通道 | 辅助参考 | 有符号 r | 原始时间 |",
              "| --- | --- | ---: | --- |"]
    for ci, name in enumerate(features["channel_names"]):
        for ref, key in (("ecg", "ecg_correlation"), ("eog", "eog_correlation"), ("emg", "emg_envelope_correlation")):
            values = [(i, v) for i, v in enumerate(features["window_features"][key][ci]) if v is not None]
            if not values:
                continue
            wi, value = max(values, key=lambda pair: abs(pair[1]))
            window = features["windows"][wi]
            reference = features.get("reference_channels", {}).get(ref, {}).get("raw_name", ref.upper())
            lines.append(f"| {_label(name)} | {_label(reference)} | {value:.3f} | {window['start_seconds']:g}–{window['stop_seconds']:g} s |")
    return lines + [""]


METHODS = {
    "welch_psd": {
        "name": "Welch 功率谱密度", "implementation": "standard_estimator",
        "definition": "Hann 子窗、逐窗去均值、单边密度谱、完整子窗等权平均；PSD 单位 V²/Hz。",
        "evidence": "spectral_metadata 保存实际窗长、重叠、FFT 点数、频率间隔和平均次数。",
        "review": "按通道和同一记录状态核对低频、宽带背景、窄峰及采集滤波；PSD 保留原始频率点。",
        "source_role": "estimator_basis",
        "source": "Welch (1967)", "url": "https://doi.org/10.1109/TAU.1967.1161901",
    },
    "endpoint_plateau": {
        "name": "文件标定端点平台", "implementation": "project_screening",
        "definition": "计数同一文件标定端点内的连续样本；最短持续时间控制候选定位；上下文范围单独记录。",
        "evidence": "evidence 保存标定来源、电压、容差、实际样本边界和持续时间。",
        "review": "核对数字原值、采集增益/ADC 范围、导出缩放或限幅、平台前后波形，再决定处理范围。",
        "source_role": "file_calibration_semantics_only",
        "source": "EDF+ specification §2.1.3", "url": "https://www.edfplus.info/specs/edfplus.html",
        "difference": "文件标定端点提供缩放信息；硬件饱和原因和排除标准需要设备证据及人工判断。",
    },
    "line_neighbour_ratio": {
        "name": "50/60 Hz 邻域峰值比", "implementation": "project_screening",
        "definition": "10 log10(候选频率 ± peak_half_width 内最大 PSD / 距候选 baseline_inner 至 baseline_outer 两侧 PSD 中位数)。",
        "evidence": "config 保存频带；channel_features 保存全程峰频、峰值比和背景；window_features 保存逐窗分数。",
        "review": "50、60 是待查频率；在局部 PSD 上核对峰频、峰宽、谐波和跨窗持续性。",
        "source_role": "replacement_design_reference",
        "source": "Bigdely-Shamlo et al. (2015), PREP / CleanLine", "url": "https://doi.org/10.3389/fninf.2015.00016",
        "difference": "现有峰值比是项目筛查统计。论文方法使用多窗频谱、正弦回归和 Thomson F 检验；接入状态：待实现。",
    },
    "local_peak_background": {
        "name": "局部谱峰与背景描述", "implementation": "project_description",
        "definition": "在 alpha_band 内按局部两侧下降的突出度找峰；另对 log10 PSD 与 log10 f 作迭代裁剪直线拟合。",
        "evidence": "config、channel_features 和 window_feature_status 保存频带、局部峰阈值、实际拟合范围、斜率、误差和缺峰状态。",
        "review": "核对峰与背景、时长及状态；8–13 Hz 局部峰作为谱形观测，alpha 节律结合生理状态判断。",
        "source_role": "replacement_design_reference",
        "source": "Donoghue et al. (2020)", "url": "https://doi.org/10.1038/s41593-020-00744-x",
        "difference": "论文联合拟合非周期背景与高斯谱峰；现有局部峰和独立直线拟合与该模型不同。接入状态：待实现。",
    },
    "reference_correlation": {
        "name": "辅助参考相关", "implementation": "project_screening",
        "definition": "每个时间窗计算 EEG 与 ECG/EOG 的有符号零延迟 Pearson r；EMG 计算带限 Hilbert 包络之间的 r。",
        "evidence": "reference_channels、windows、window_features 保存参考来源、实际时间和有符号数值；图中汇总为各窗最大 |r|。",
        "review": "先对照同步原始波形；进一步定位心搏/眼动事件，核对单次片段与锁时平均的重复形态。",
        "source_role": "event_review_design_reference",
        "source": "Jas et al. (2018), §3.5.2", "url": "https://doi.org/10.3389/fnins.2018.00530",
        "difference": "相关阈值为项目筛查参数；事件定位、锁时平均和重复性评价的接入状态：待实现。",
    },
    "amplitude_screening": {
        "name": "幅度、低变化及频带占比", "implementation": "project_screening",
        "definition": "逐窗去均值 RMS、峰峰值、相邻差、相对通道中位数幅度、频带 PSD 积分与总功率之比。",
        "evidence": "config、policy_config、window_features 保存阈值、频带、持续时间和各窗测量值。",
        "review": "结合参考方式、同类通道、睡眠/任务状态及同步 EMG 检查候选。",
        "source_role": "replacement_design_reference",
        "source": "PREP (2015); autoreject / Jas et al. (2017)",
        "url": "https://doi.org/10.1016/j.neuroimage.2017.06.030",
        "difference": "现有固定阈值为项目筛查参数。PREP 稳健跨导联统计和 autoreject 交叉验证阈值的适配与实现待进行。",
    },
}


def measurement_methods():
    return deepcopy(METHODS)


def render_measurement_methods(features, decisions):
    lines = ["# EEG 计算定义与方法依据", "", "[返回复核报告](review_report.md)", "",
             "本次策略：全部样本保留，候选交给人工复核。阈值来源：项目筛查参数。", "",
             "论文的引用角色分别记录为估计方法、文件语义或下一步方法设计。", ""]
    lines += _measured_tables(features, decisions)
    for key, item in features["measurement_methods"].items():
        lines += [f"## {item['name']} · `{key}`", "", item["definition"], "",
                  item["evidence"], "", f"复核：{item['review']}", "",
                  f"依据：[{item['source']}]({item['url']})；引用角色：`{item['source_role']}`。", ""]
        if item.get("difference"):
            lines += [item["difference"], ""]
    import json
    lines += ["## 本次参数", "", "```json", json.dumps({"feature_config": features["config"],
              "policy_config": decisions["policy_config"]}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)
