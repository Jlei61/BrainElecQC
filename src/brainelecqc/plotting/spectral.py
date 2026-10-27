"""PSD, absolute alpha power and relative alpha on the same selected channels."""
import matplotlib.pyplot as plt
import numpy as np

from ..eeg.spectral_summary import spectral_display_rows


COLORS = ("#1F648A", "#B96E24", "#428C76", "#925B9B", "#A34550", "#68737E")


def make_spectrum_figure(result):
    spectral = result["method_results"]["spectral"]
    rows = spectral_display_rows(result)
    if not rows:
        raise ValueError("PSD summary requires saved alpha_power measurements; regenerate the result first")
    alpha = spectral["alpha_power"]
    lo, hi = alpha["alpha_band_hz"]
    rlo, rhi = alpha["reference_band_hz"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.3, 1, 1]}, layout="constrained")
    f = np.asarray(spectral["psd"]["frequencies_hz"])
    visible_values = []
    for row, color in zip(rows, COLORS):
        power = np.asarray(spectral["psd"]["power_v2_hz"])[row["channel_index"]]*1e12
        axes[0].plot(f, np.where(power > 0, power, np.nan), color=color, lw=1.2, label=row["display_name"])
        visible_values.extend(power[(f >= rlo) & (f <= rhi) & (power > 0)])
    axes[0].set_gid("spectral:psd")
    axes[0].set_yscale("log")
    if visible_values:
        axes[0].set_ylim(min(visible_values)/1.6, max(visible_values)*1.8)
    axes[0].axvspan(lo, hi, color="#DDBB69", alpha=.18)
    axes[0].set(xlim=(max(f[0], rlo), min(rhi, f[-1])), xlabel="频率 (Hz)", ylabel="PSD (µV²/Hz)")
    axes[0].set_title(f"a  功率谱 · {rows[0]['start_seconds']:g}–{rows[0]['stop_seconds']:g} s", loc="left", fontweight="bold")
    axes[0].legend(fontsize=9, ncols=2, frameon=False)
    for ax, field, label, title in (
        (axes[1], "alpha_power_uv2", "α 功率 (µV²)", f"b  α 功率 · {lo:g}–{hi:g} Hz"),
        (axes[2], "relative_alpha_percent", "α 占比 (%)", f"c  α / {rlo:g}–{rhi:g} Hz")):
        values = [r[field] for r in rows]
        bars = ax.barh(np.arange(len(rows)), [v if v is not None else 0 for v in values],
                       color=COLORS[:len(rows)], height=.6)
        ax.set_gid(f"spectral:{field}")
        ax.set_yticks(np.arange(len(rows)), [r["display_name"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.bar_label(bars, labels=["待评" if v is None else f"{v:.1f}" for v in values], padding=4, fontsize=9)
        max_value = max([v for v in values if v is not None] or [1])
        ax.set_xlim(0, 115 if field == "relative_alpha_percent" else max(1, max_value*1.32))
        if field == "relative_alpha_percent":
            ax.set_xticks([0, 25, 50, 75, 100])
    return fig


def make_alpha_time_figure(result):
    spectral = result["method_results"]["spectral"]
    block = spectral["alpha_power"]
    rows = spectral_display_rows(result)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for row, color in zip(rows, COLORS):
        windows = [w for w in block["windows"] if w["channel_index"] == row["channel_index"]]
        times = [(w["start_seconds"]+w["stop_seconds"])/2 for w in windows]
        for ax, field in zip(axes, ("alpha_power_uv2", "relative_alpha_percent")):
            ax.plot(times, [np.nan if w[field] is None else w[field] for w in windows],
                    color=color, lw=1.1, marker=".", ms=3, label=row["display_name"])
    axes[0].set(ylabel="α 功率 (µV²)")
    # Shared physical scale exposes sustained differences and short transients.
    if any(w["alpha_power_uv2"] is not None for w in block["windows"]):
        axes[0].set_yscale("log")
    axes[0].set_title(f"a  α 功率随时间 · {block['config']['window_seconds']:g} s 窗 / {block['config']['step_seconds']:g} s 步长",
                      loc="left", fontweight="bold")
    axes[0].legend(fontsize=9, ncols=3, frameon=False)
    axes[1].set(xlabel="原始时间 / 窗中心 (s)", ylabel="α 占比 (%)", ylim=(0, 100))
    axes[1].set_title("b  α 占比随时间", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlim(block["input_interval"]["start_seconds"], block["input_interval"]["stop_seconds"])
    return fig
