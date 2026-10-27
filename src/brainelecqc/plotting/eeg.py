"""Render supplied EEG features and decisions using the original signal clock.

The module consumes computed results. Feature extraction and decisions belong to
the QC modules, so the same figures can be called from a script or a host plugin.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from ..eeg.display import display_channel_name


BLUE = "#27718E"
ORANGE = "#C07932"
RED = "#B55551"
INK = "#253545"
GRAY = "#CAD0D6"


def _style():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS", "Heiti SC"]
    selected = next((name for name in preferred if name in fonts), "DejaVu Sans")
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [selected, "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }


def _validate(data_volts, features, keep_mask):
    data = np.asarray(data_volts)
    mask = np.asarray(keep_mask)
    if data.ndim != 2 or not data.shape[0] or data.shape[1] < 2:
        raise ValueError("data_volts must be channel × sample with at least two samples")
    if mask.shape != data.shape or mask.dtype != np.bool_:
        raise ValueError("keep_mask must be a boolean array with the same shape as data_volts")
    ids, names = features["channel_ids"], features["channel_names"]
    if len(ids) != data.shape[0] or len(names) != data.shape[0]:
        raise ValueError("Feature channel identities must match data rows")
    fs = float(features["sampling_rate_hz"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if features["stop_sample"] - features["start_sample"] != data.shape[1]:
        raise ValueError("Feature sample interval must match the supplied data length")
    frequencies = np.asarray(features["frequencies_hz"], dtype=float)
    psd = np.asarray(features["psd_v2_hz"], dtype=float)
    if frequencies.ndim != 1 or psd.shape != (data.shape[0], len(frequencies)):
        raise ValueError("PSD must have channel × frequency shape")
    return data, mask, fs


def _window(data, fs, start_seconds, display_start_seconds, display_duration_seconds):
    duration = float(display_duration_seconds)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("display_duration_seconds must be finite and positive")
    requested = start_seconds if display_start_seconds is None else float(display_start_seconds)
    if not np.isfinite(requested):
        raise ValueError("display_start_seconds must be finite")
    left = max(0, int(round((requested - start_seconds) * fs)))
    right = min(data.shape[1], int(round((requested + duration - start_seconds) * fs)))
    if right - left < 2:
        raise ValueError("Display interval must overlap at least two input samples")
    time = start_seconds + np.arange(left, right) / fs
    return left, right, time, (time[0], start_seconds + right / fs)


def _label_names(names):
    strings = [display_channel_name(name) for name in names]
    return [f"{name} [{index}]" if strings.count(name) > 1 else name
            for index, name in enumerate(strings)]


def _limits(signal):
    finite = np.abs(signal[np.isfinite(signal)])
    amplitude = max(float(np.max(finite)) if finite.size else 0.0, 1.0)
    return -amplitude * 1.08, amplitude * 1.08


def _runs(mask):
    """Half-open contiguous runs, including runs touching either endpoint."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return zip(edges[::2], edges[1::2])


def _title(fig, stage, title, subtitle):
    fig.text(0.07, 0.955, f"{stage}  {title}", size=20, weight="bold")
    fig.text(0.07, 0.915, subtitle, size=10, color="#5A6977")


def _waveform_figure(data, mask, fs, features, labels, shown, interval, time,
                     left, right, title, decisions, after=False):
    fig = plt.figure(figsize=(11.2, 7.2))
    grid = fig.add_gridspec(len(shown), 1, left=0.11, right=0.97,
                           bottom=0.16, top=0.84, hspace=0.36)
    stage = ("03 自动剔除后" if decisions.get("automatic_exclusions") else "03 待审核 · 全部保留") if after else "01 原始 EEG"
    _title(fig, stage, title,
           f"{interval[0]:g}–{interval[1]:g} s  ·  {fs:g} Hz")
    for row, channel in enumerate(shown):
        ax = fig.add_subplot(grid[row, 0])
        ax.set_gid(f"waveform:{features['channel_ids'][channel]}")
        values = data[channel, left:right] * 1e6
        if after:
            ax.plot(time, values, color=GRAY, linewidth=0.5, zorder=1)
            local_mask = mask[channel, left:right]
            for begin, end in _runs(~local_mask):
                ax.axvspan(time[begin], interval[0] + end / fs,
                           color=RED, alpha=0.11, linewidth=0)
            ax.plot(time, np.where(local_mask, values, np.nan),
                    color=BLUE, linewidth=0.65, zorder=2)
        else:
            ax.plot(time, values, color=BLUE, linewidth=0.65)
        ax.set_xlim(*interval)
        ax.set_ylim(*_limits(values))
        ax.set_ylabel(f"{labels[channel]}\nµV", size=9)
        ax.axhline(0, color="#DEE3E8", linewidth=0.45, zorder=0)
        ax.grid(axis="x", color="#EEF0F2", linewidth=0.6)
        if row == len(shown) - 1:
            ax.set_xlabel("原始时间（s）")
        else:
            ax.tick_params(labelbottom=False)

    if after:
        handles = [Line2D([], [], color=BLUE, label="保留信号"), Line2D([], [], color=GRAY, label="原始对照")]
        if decisions.get("automatic_exclusions"):
            handles.append(Rectangle((0, 0), 1, 1, color=RED, alpha=0.15, label="自动剔除"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.025), ncol=len(handles), fontsize=9)

    return fig


def _matrix(features, key):
    shape = (len(features["channel_ids"]), len(features["windows"]))
    value = features["window_features"].get(key)
    if value is None:
        return np.full(shape, np.nan)
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{key} must have channel × window shape")
    return matrix


def _review_figure(features, decisions, labels, title):
    fig = plt.figure(figsize=(11.2, 9.2))
    automatic = decisions.get("automatic_exclusions", [])
    end = features["start_seconds"] + (features["stop_sample"]-features["start_sample"])/features["sampling_rate_hz"]
    _title(fig, "02 自动特征", title,
           f"{features['start_seconds']:g}–{end:g} s  ·  {len(features['windows'])} 个分析窗")
    grid = fig.add_gridspec(3, 2, left=0.11, right=0.94, bottom=0.09, top=0.84,
                           height_ratios=[1.05, 1, 1.1], hspace=0.78, wspace=0.50)
    ax = fig.add_subplot(grid[0, :])
    ax.set_gid("features:rms")
    windows = features["windows"]
    rms = _matrix(features, "rms_uv")
    if windows:
        finite = rms[np.isfinite(rms)]
        upper = max(float(np.max(finite)) if finite.size else 1, 1)
        # Draw each interval at its recorded boundaries, preserving short final
        # windows and any gaps supplied by the feature extractor.
        for index, window in enumerate(windows):
            image = ax.pcolormesh([window["start_seconds"], window["stop_seconds"]],
                                  np.arange(len(labels) + 1) - 0.5, rms[:, index:index + 1],
                                  cmap="cividis", vmin=0, vmax=upper, shading="flat")
        ax.set_ylim(len(labels) - 0.5, -0.5)
        fig.colorbar(image, ax=ax, fraction=0.022, pad=0.016, label="AC RMS（µV）")
    channel_indices = {channel: index for index, channel in enumerate(features["channel_ids"])}
    for item in automatic:
        index = channel_indices[item["channel_id"]]
        ax.add_patch(Rectangle((item["start_seconds"], index - 0.42),
                               item["stop_seconds"] - item["start_seconds"], 0.84,
                               edgecolor="#F0AAA2", facecolor="none", linewidth=1.8))
    if automatic:
        ax.legend(handles=[Rectangle((0, 0), 1, 1, edgecolor="#F0AAA2", facecolor="none",
                                     linewidth=1.8, label="自动剔除")],
                  loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_xlabel("原始时间（s）")
    ax.set_title("a  AC RMS（去均值幅度）", loc="left", weight="bold")

    ax = fig.add_subplot(grid[1, 0])
    _psd_panel(ax, features, labels, frequency_range=(40, 70),
               gid="features:line_noise", title="b  PSD：50/60 Hz 邻域")
    for center in (50, 60):
        width = features.get("config", {}).get("line_peak_half_width_hz", 1)
        ax.axvspan(center-width, center+width, color=GRAY, alpha=0.3, zorder=0)
        ax.axvline(center, color="#808B94", linewidth=0.6, linestyle="--", zorder=0)

    ax = fig.add_subplot(grid[1, 1])
    low, high = features.get("config", {}).get("alpha_band", (8, 13))
    _psd_panel(ax, features, labels, frequency_range=(min(2, low), max(30, high+2)),
               gid="features:alpha", title=f"c  PSD：{low:g}–{high:g} Hz 与周围背景")
    ax.axvspan(low, high, color=GRAY, alpha=0.3, zorder=0)

    _reference_panel(fig.add_subplot(grid[2, 0]), features, labels)
    _psd_panel(fig.add_subplot(grid[2, 1]), features, labels)

    return fig


def _reference_panel(ax, features, labels):
    """Show summary values already calculated by the feature extractor."""
    ax.set_gid("features:references")
    keys = ["ecg_correlation", "eog_correlation", "emg_envelope_correlation"]
    values = []
    for key in keys:
        values.append([np.max(np.abs(row[np.isfinite(row)])) if np.any(np.isfinite(row)) else np.nan
                       for row in _matrix(features, key)])
    matrix = np.asarray(values, dtype=float).T
    ax.imshow(matrix, aspect="auto", interpolation="none", cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(3):
            value = matrix[row, column]
            text = "待提供参考"
            if np.isfinite(value):
                wi = int(np.nanargmax(np.abs(_matrix(features, keys[column])[row])))
                window = features["windows"][wi]
                text = f"{value:.2f}\n{window['start_seconds']:g}–{window['stop_seconds']:g} s"
            ax.text(column, row, text, ha="center", va="center", size=8,
                    color="white" if np.isfinite(value) and value > 0.65 else INK)
    ax.set_xticks([0, 1, 2], ["ECG 波形", "EOG 波形", "EMG 包络"], fontsize=9)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_title("d  参考相关：各窗 |r| 最大值", loc="left", weight="bold")


def _psd_panel(ax, features, labels, *, frequency_range=None, gid="features:psd", title="e  全分析窗 PSD"):
    ax.set_gid(gid)
    frequencies = np.asarray(features["frequencies_hz"], dtype=float)
    psd = np.asarray(features["psd_v2_hz"], dtype=float)
    lower, upper = frequency_range or (0.5, min(features["sampling_rate_hz"] / 2, 100))
    valid = (frequencies >= lower) & (frequencies <= upper)
    colors = [BLUE, ORANGE, "#677AA0", "#7E9772", "#9D7A8B", "#698B87"]
    for index, density in enumerate(psd):
        ax.plot(frequencies[valid], 10*np.log10(np.maximum(density[valid]*1e12, 1e-20)),
                color=colors[index % len(colors)], linewidth=0.9, label=labels[index])
    ax.set_title(title, loc="left", weight="bold")
    if frequency_range:
        ax.set_xlim(lower, upper)
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("dB (µV²/Hz)")
    if gid == "features:psd":
        ax.legend(fontsize=7, loc="upper right")


def make_eeg_figures(data_volts, features, decisions, keep_mask, *, title="EEG QC",
                     display_start_seconds=None, display_duration_seconds=15,
                     max_channels=6):
    """Create three report figures from channel-major volts and supplied results.

    ``display_start_seconds`` follows the original clock from ``features``.
    Selection changes only the waveform view. Feature panels use the complete
    analysis interval. Raw and retained traces have identical per-channel limits.
    The returned figures are caller-owned; :func:`save_eeg_figures` closes them.
    """
    data, mask, fs = _validate(data_volts, features, keep_mask)
    if isinstance(max_channels, bool) or int(max_channels) != max_channels or max_channels < 1:
        raise ValueError("max_channels must be a positive integer")
    shown = list(range(min(data.shape[0], int(max_channels))))
    labels = _label_names(features["channel_names"])
    left, right, time, interval = _window(data, fs, float(features["start_seconds"]),
                                         display_start_seconds, display_duration_seconds)
    with mpl.rc_context(_style()):
        original = _waveform_figure(data, mask, fs, features, labels, shown, interval,
                                   time, left, right, title, decisions)
        review = _review_figure(features, decisions, labels, title)
        after = _waveform_figure(data, mask, fs, features, labels, shown, interval,
                                time, left, right, title, decisions, after=True)
    return {"01_initial": original, "02_features_review": review, "03_auto_excluded": after}


def save_eeg_figures(figures, output_dir, *, formats=("png",)):
    """Export PNG by default, then close the supplied figures.

    A direct caller can explicitly select a nonempty sequence of distinct
    ``png``, ``svg`` or ``pdf`` formats. QC reports and plugin runs use PNG.
    """
    files = {}
    try:
        if (not isinstance(formats, (tuple, list)) or not formats
                or any(not isinstance(value, str) or value not in {"png", "svg", "pdf"} for value in formats)
                or len(set(formats)) != len(formats)):
            raise ValueError("formats must be a nonempty sequence of distinct png, svg or pdf values")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with mpl.rc_context(_style()):
            for name, figure in figures.items():
                if Path(name).name != name:
                    raise ValueError("Figure names must be plain file stems")
                files[name] = {}
                for extension in formats:
                    destination = output / f"{name}.{extension}"
                    figure.savefig(destination, dpi=180, facecolor="white")
                    files[name][extension] = str(destination.resolve())
    finally:
        for figure in figures.values():
            plt.close(figure)
    return files
