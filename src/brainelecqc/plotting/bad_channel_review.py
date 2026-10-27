"""Bad-channel evidence with shared-scale waveform and PSD comparisons."""
from numbers import Integral

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..eeg.binding import validate_result_identity
from .eeg import _style


def make_bad_channel_review_figure(result, channel_indices, *, start_seconds, duration_seconds=4,
                                   title="坏道检测 · 待人工复核"):
    """Return a figure and traceable display metadata for explicitly selected leads.

    Input indices address the selected EEG array, not the original file. The
    caller chooses a shared reference group and records that display selection.
    All measurements come from the saved detection result; raw voltage is shown
    at its original sample clock and the same vertical scale in every trace.
    """
    validate_result_identity(result)
    indices = list(channel_indices)
    features, data = result["features"], result["raw_data_volts"]
    if (not 1 <= len(indices) <= 6 or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(data) for i in indices)):
        raise ValueError("Choose one to six distinct EEG channel indices within the input")
    if not np.isfinite(start_seconds) or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Choose a finite positive waveform display interval")
    bad, spectral = result["method_results"]["bad_channels"], result["method_results"]["spectral"]
    rows = bad["channels"]
    group_ids = {rows[i]["group_id"] for i in indices}
    if len(group_ids) != 1:
        raise ValueError("Choose channels from one declared comparison group")
    group = next(g for g in bad["groups"] if g["group_id"] in group_ids)
    members = [r for r in rows if r["group_id"] in group_ids]
    fs, origin = features["sampling_rate_hz"], features["start_seconds"]
    left = round((start_seconds-origin)*fs)
    right = round((start_seconds+duration_seconds-origin)*fs)
    if not 0 <= left < right <= data.shape[1] or right-left < 2:
        raise ValueError("Choose a waveform interval inside the recording with at least two samples")
    frequencies = np.asarray(spectral["psd"]["frequencies_hz"])
    visible = (frequencies >= 2) & (frequencies <= min(80, frequencies[-1]))
    if visible.sum() < 2:
        raise ValueError("This comparison needs at least two PSD bins in the 2–80 Hz display range")
    names = features["channel_names"]
    labels = {i: f"{names[i]} [row {i}]" if names.count(names[i]) > 1 else names[i] for i in indices}
    palette = ("#B35250", "#BF7A2C", "#8A68A3", "#277898", "#458A79", "#6E7880")
    colors = dict(zip(indices, palette))
    metadata = {"data_sha256": features["data_sha256"], "sampling_rate_hz": fs,
                "group_id": group["group_id"], "thresholds_active": group["thresholds_active"],
                "group_channel_indices": [r["channel_index"] for r in members],
                "selected_channels": [rows[i] for i in indices],
                "parameters": bad["parameters"], "candidates": bad["candidates"],
                "waveform": {"start_sample": features["start_sample"]+left,
                             "stop_sample": features["start_sample"]+right,
                             "start_seconds": origin+left/fs, "stop_seconds": origin+right/fs},
                "psd": {"start_seconds": spectral["psd"]["start_seconds"],
                        "stop_seconds": spectral["psd"]["start_seconds"]+spectral["psd"]["estimator"]["covered_samples"]/fs},
                "status": "pending_review"}
    with mpl.rc_context(_style()):
        fig = plt.figure(figsize=(12.6, 8.7), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
        score, psd = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        traces = [fig.add_subplot(cell) for cell in grid[1, 0].subgridspec(len(indices), 1, hspace=.05)]
        correlation = fig.add_subplot(grid[1, 1])
        score.set_gid("bad:high_frequency_z")
        score.scatter([r["channel_index"] for r in members], [r["hf_z"] for r in members],
                      color="#BFC5C9", s=16, zorder=1)
        if group["thresholds_active"]:
            score.axhline(bad["parameters"]["hf_noise_z"], color="#555555", ls="--", lw=.8, label="候选阈值")
        for i in indices:
            if rows[i]["hf_z"] is not None:
                score.scatter(i, rows[i]["hf_z"], color=colors[i], s=36, zorder=3)
                score.annotate(labels[i], (i, rows[i]["hf_z"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
        score.set(xlabel="通道索引（本次分析顺序）", ylabel="高频噪声 z")
        score.set_title("a  高频噪声 · 全段统计", loc="left", fontweight="bold")
        score.margins(y=.2)
        if group["thresholds_active"]:
            score.legend(loc="upper left", fontsize=9)

        power = np.asarray(spectral["psd"]["power_v2_hz"])*1e12
        for i in indices:
            psd.plot(frequencies[visible], np.where(power[i, visible] > 0, power[i, visible], np.nan),
                     color=colors[i], lw=1.15, label=labels[i])
        psd.set_gid("bad:psd")
        psd.set(xlabel="频率 (Hz)", ylabel="PSD (µV²/Hz)", yscale="log",
                xlim=(frequencies[visible][0], frequencies[visible][-1]))
        psd.set_title(f"b  同轴 PSD · {metadata['psd']['start_seconds']:g}–{metadata['psd']['stop_seconds']:g} s",
                      loc="left", fontweight="bold")
        psd.legend(fontsize=9, ncols=2)

        times = origin+np.arange(left, right)/fs
        values = np.asarray(data)[indices, left:right]*1e6
        limit = max(float(np.max(np.abs(values))), 1)*1.08
        for ax, i, y in zip(traces, indices, values):
            ax.set_gid(f"bad:raw:{i}")
            ax.plot(times, y, color=colors[i], lw=.7)
            ax.set(xlim=(times[0], origin+right/fs), ylim=(-limit, limit), ylabel="µV")
            ax.text(.01, .91, labels[i], transform=ax.transAxes, va="top", color=colors[i], fontsize=9)
            if ax is not traces[-1]:
                ax.tick_params(labelbottom=False)
        traces[0].set_title("c  原始波形 · 同一幅度刻度", loc="left", fontweight="bold")
        traces[-1].set_xlabel("原始时间 (s)")

        windows = bad["windows"]
        threshold = bad["parameters"]["correlation_threshold"]
        centers = np.array([(w["start_seconds"]+w["stop_seconds"])/2 for w in windows])
        for i in indices:
            values = np.asarray(bad["window_metrics"]["max_abs_correlation"][i], dtype=float)
            correlation.plot(centers, values, color=colors[i], lw=.9)
            marked = np.array(sorted({w for c in bad["candidates"] if c["channel_index"] == i
                                      and c["kind"] == "low_correlation" for w in c["window_indices"]}), dtype=int)
            if len(marked):
                correlation.scatter(centers[marked], values[marked], color=colors[i], marker="D", s=20, zorder=4)
        correlation.axhline(threshold, color="#555555", ls="--", lw=.8)
        correlation.axvspan(origin+left/fs, origin+right/fs, color="#8799A7", alpha=.13)
        correlation.set_gid("bad:correlation")
        correlation.set(xlim=(features["start_seconds"], features["stop_seconds"]), ylim=(0, 1.04),
                        xlabel="原始时间 / 窗中心 (s)", ylabel="同组相关（|r| 的第 98 百分位）")
        correlation.set_title("d  逐秒低相关 · 菱形为候选窗", loc="left", fontweight="bold")
        fig.suptitle(title, x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig, metadata
