"""Short, actionable human review beside reproducible paper measurements."""
import json

from .report import _text, _names


LABELS = {"amplitude_deviation": "幅度离群", "high_frequency_noise": "高频噪声", "low_correlation": "低相关",
          "intermittent_dropout": "间歇掉线", "flat_signal": "平直信号", "nonfinite_input": "非有限值",
          "file_endpoint_plateau": "文件端点平台", "flatline": "低变化平台", "sinusoidal_line": "窄带正弦",
          "near_zero_flatline": "近零平线 / 掉线候选", "calibration_endpoint_plateau": "量程端点平台",
          "constant_value": "恒值平台", "high_frequency_burst": "宽带高频突发", "emg_supported_burst": "EEG 高频与 EMG 同步"}


def _range(row):
    return f"{row['start_seconds']:.3f}–{row['stop_seconds']:.3f} s"


def render_paper_review(result, plot_config, *, with_figures=True):
    from .automatic_report import render_automatic_report
    return render_automatic_report(result, plot_config, with_figures=with_figures)


def render_paper_items(result):
    names = _names(result["features"])
    lines = ["# 逐项复核", "", "[返回报告](review_report.md)", "",
             "| ID | 通道 | 原始时间 / 样本区间 | 候选依据 | 决定 |", "| --- | --- | --- | --- | --- |"]
    for row in result["decisions"]["review_queue"]:
        lines.append(f"| {row['candidate_id']} | {names[row['channel_id']]} | {_range(row)} / [{row['start_sample']}, {row['stop_sample']}) | "
                     f"{LABELS.get(row['kind'], row['kind'])} | 保留 / 标坏道 / 标记时间段 |")
    lines += ["", "ECG/EOG 逐事件定位：events_ecg.tsv / events_eog.tsv。α 功率：alpha_power.tsv；背景与谱峰拟合：spectral_fits.tsv。", ""]
    return "\n".join(lines)


def render_paper_methods(result):
    methods = result["method_results"]
    cfg = methods["spectral"]["config"]
    bad_cfg = methods["bad_channels"]["parameters"]
    estimator = methods["spectral"]["psd"]["estimator"]
    fs = result["features"]["sampling_rate_hz"]
    lines = ["# 计算与图示说明", "",
        "- [PREP (Bigdely-Shamlo et al., 2015)](https://doi.org/10.3389/fninf.2015.00016)：稳健幅度、高频噪声、低相关与掉线分项，PyPREP 0.5.0 数值对照。",
        "- [Mitra & Pesaran, 1999](https://doi.org/10.1016/S0006-3495(99)77236-X)：多锥窗正弦 F 检验；全部通道、窗口和扫描频率统一 Bonferroni 校正。",
        "- [Donoghue et al., 2020](https://doi.org/10.1038/s41593-020-00744-x)：FOOOF 1.1.1 周期/非周期拟合，记录峰频、相对背景高度、2σ 带宽和残差。",
        "- [Jas et al., 2018 §3.5.2](https://doi.org/10.3389/fnins.2018.00530)：MNE ECG/EOG 检出与事件锁时复核；中位数、SEM、分半相关为附加描述统计。", "",
        "坏道图 B 显示宽频 PSD，C 显示同尺度原始波形。高频候选与同参考组对照一起展示；近零平线、量程端点平台按类别放在坏道图页，阴影精确对应原始样本区间。全部通道 PSD 单独输出。选图范围见 figure_selection.json。",
        f"工频图显示全部通道的 50 Hz、60 Hz 两个热图。横轴为 {cfg['line_window_seconds']:g} s 窗的起点，步长 {cfg['line_step_seconds']:g} s；统一色值为 −log10(校正 p)，方框定位显著窗口。标题给出频率和受影响通道数，逐窗数值与谐波结果在 line_windows.tsv。",
        "关键频谱图优先展示名称明确的 O1/Oz/O2，再补高频噪声、幅度离群、工频候选中统计量最大的通道，最后按原始顺序补足六道。所有面板使用同一通道、同一颜色；选择依据保存在 spectral_display.json。",
        f"α 功率为观测 Welch PSD 在 {cfg['alpha_band_hz'][0]:g}–{cfg['alpha_band_hz'][1]:g} Hz 的积分，单位 µV²；α 占比的分母为 {cfg['fit_range_hz'][0]:g}–{cfg['fit_range_hz'][1]:g} Hz 积分。边界线性插值后做梯形积分，计入该带内的周期和非周期功率。时间图逐个 Welch 子窗计算，按原始窗中心定位；全段及逐窗数值见 alpha_power.tsv。",
        f"全部通道与 {cfg['parameter_window_seconds']:g} s 背景、谱峰拟合保存在 spectral_fits.tsv / paper_methods.json。",
        f"全段 Welch 实际覆盖 {estimator['covered_samples']/fs:g} s，尾部 {estimator['tail_samples']/fs:g} s 保留于原始波形。每次拟合的覆盖范围记录在 estimator.covered_samples。",
        "ECG/EOG 每图画一组上下波形：EEG 在上，辅助导联在下，共用原始时间轴。窗口从评分代表事件向邻近两侧扩展，目标为 3 个事件。ECG 相邻间隔最多 3 秒、事件跨度最多 6 秒；EOG 分别为 6 秒、10 秒；两端各留 0.5 秒，最短显示 2 秒。单事件用方形图。红线为代表事件，橙线为邻近检出事件。EOG 评分条件不足时，选完整辅助事件窗内 EEG 峰峰值最大的事件展示。轴标签保留信号名与 µV，采集参考简写 Ref；标题给出原始 t₀ 与检测状态。选窗规则与候选数值保存在 figure_selection.json。α 时间曲线通过报告中的补充链接查看，用于定位能量变化时段。",
        "连续性直接检查原始样本的连续恒值，默认最短 0.1 秒；按文件校准端点、近零量化电平和其他恒值分类。门槛为项目短片段筛查参数。肌电使用 MNE 高频 Hilbert 包络 z，结合宽带功率和同步 EMG；连续性异常周围加入保护范围。全部参数与有效覆盖保存在对应方法块。",
        "[EDF 校准定义](https://www.edfplus.info/specs/guidelines.html)；[EEGLAB 连续恒值检查](https://github.com/sccn/clean_rawdata/blob/master/clean_flatlines.m)；[肌电频谱依据](https://doi.org/10.3389/fnhum.2013.00138)；[MNE 包络方法](https://mne.tools/stable/generated/mne.preprocessing.annotate_muscle_zscore.html)。",
        "PSD 主图使用 µV²/Hz 的对数坐标；拟合详情使用 dB(µV²/Hz) = 10 log10(PSD[V²/Hz]) + 120。每个频谱、候选和事件均保存原始 sample、seconds、channel_id 与输入 SHA-256。", "",
        "## 本次参数", "", "```json", json.dumps(result["manifest"]["method_config"], ensure_ascii=False, indent=2), "```", "",
        "完整公式、预处理、方法差异与软件版本保存在 paper_methods.json 各方法块。", "",
        "## 文件声明的采集滤波", "",
        "；".join(f"{_text(r['raw_name'])}: {_text(r.get('prefilter') or '未声明')}" for r in result["manifest"].get("acquisition_filter_headers", [])), "",
        "频谱描述导出文件中的能量；人工结合以上采集滤波和导出处理判断有效带宽。", ""]
    return "\n".join(lines)
