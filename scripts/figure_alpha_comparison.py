"""Compare named O1/Oz/O2 alpha in two saved EEG results.

Usage: python scripts/figure_alpha_comparison.py OPEN_BUNDLE CLOSED_BUNDLE OUTPUT
The caller supplies recordings with documented eyes-open / eyes-closed states.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from brainelecqc.eeg.binding import validate_result_identity
from brainelecqc.eeg.masks import apply_eeg_decisions
from brainelecqc.eeg.spectral_summary import occipital_name
from brainelecqc.plotting import save_eeg_figures
from brainelecqc.plotting.eeg import _style


def load_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Result fingerprint mismatch: {name}")
    result = {"manifest": manifest}
    for key, name in (("features", "features.json"), ("decisions", "decisions.json"), ("method_results", "paper_methods.json")):
        result[key] = json.loads((root/name).read_text(encoding="utf-8"))
    with np.load(root/"review_signals.npz", allow_pickle=False) as signals:
        result["raw_data_volts"] = signals["raw_data_volts"]
        result["references"] = {key: signals[key] for key in signals.files if key != "raw_data_volts"}
    _, result["keep_mask"] = apply_eeg_decisions(result["raw_data_volts"], result["decisions"],
                                                 channel_ids=result["features"]["channel_ids"])
    validate_result_identity(result)
    if "alpha_power" not in result["method_results"]["spectral"]:
        raise ValueError("Regenerate this bundle with figure_eeg_qc.py to save alpha_power first")
    return result


def comparison_data(results):
    rows, indices = [], []
    settings = None
    for state, result in zip(("eyes_open", "eyes_closed"), results):
        spectral = result["method_results"]["spectral"]
        alpha = spectral["alpha_power"]
        current = {key: alpha[key] for key in ("config", "sfreq")}
        if settings is not None and current != settings:
            raise ValueError("Use matching sampling rates, duration and alpha/Welch settings for this comparison")
        settings = current
        selected = []
        for lead in ("o1", "oz", "o2"):
            found = [i for i, name in enumerate(alpha["channel_names"]) if occipital_name(name) == lead]
            if len(found) != 1:
                raise ValueError(f"Provide one explicitly named {lead} channel in each recording")
            i = found[0]
            selected.append(i)
            row = alpha["full"][i]
            if row["status"] != "computed":
                raise ValueError(f"Alpha power is unavailable for {row['channel_name']}: {row['status']}")
            rows.append({"state": state, "lead": lead, **row})
        indices.append(selected)
    # References must match; raw names are the declared channel/reference labels.
    if [rows[i]["channel_name"] for i in range(3)] != [rows[i]["channel_name"] for i in range(3, 6)]:
        raise ValueError("Match raw channel/reference names before comparing recordings")
    return rows, indices


def make_comparison(results, title):
    rows, indices = comparison_data(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1.4]}, layout="constrained")
    colors = ("#277898", "#BC7628")
    labels = ("睁眼", "闭眼")
    visible_values = []
    for state_index, (result, chosen, color, label) in enumerate(zip(results, indices, colors, labels)):
        spectral = result["method_results"]["spectral"]
        alpha = spectral["alpha_power"]
        psd = np.asarray(spectral["psd"]["power_v2_hz"])[chosen].mean(axis=0)*1e12
        axes[0].plot(spectral["psd"]["frequencies_hz"], psd, color=color, lw=1.5, label=label)
        f = np.asarray(spectral["psd"]["frequencies_hz"])
        lo, hi = alpha["reference_band_hz"]
        visible_values.extend(psd[(f >= lo) & (f <= hi) & (psd > 0)])
        values = [rows[state_index*3+i]["alpha_power_uv2"] for i in range(3)]
        bars = axes[1].bar(np.arange(3)+(state_index-.5)*.34, values, width=.32, color=color)
        axes[1].bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize=9, padding=3)
        windows = alpha["windows"]
        groups = [[w for w in windows if w["window_index"] == wi and w["channel_index"] in chosen]
                  for wi in sorted(set(w["window_index"] for w in windows))]
        centers = [(g[0]["start_seconds"]+g[0]["stop_seconds"])/2-alpha["input_interval"]["start_seconds"] for g in groups]
        means = [float(np.mean([w["alpha_power_uv2"] for w in g]))
                 if all(w["alpha_power_uv2"] is not None for w in g) else np.nan for g in groups]
        axes[2].plot(centers, means, color=color, lw=1.3, label=label)
    alpha = results[0]["method_results"]["spectral"]["alpha_power"]
    axes[0].axvspan(*alpha["alpha_band_hz"], color="#DDBB69", alpha=.18)
    axes[0].set(xlim=alpha["reference_band_hz"], xlabel="频率 (Hz)", ylabel="PSD (µV²/Hz)", yscale="log")
    axes[0].set_ylim(min(visible_values)/1.6, max(visible_values)*1.8)
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(3), [r["channel_name"] for r in rows[:3]])
    axes[1].set_ylabel("α 功率 (µV²)")
    axes[1].margins(y=.22)
    axes[2].set(xlabel="距各记录起点 (s)", ylabel="α 功率 (µV²)", yscale="log")
    for ax, heading in zip(axes, ("a  枕区平均 PSD", "b  各道 α 功率", "c  枕区平均 α 随时间")):
        ax.set_title(heading, loc="left", fontweight="bold")
    fig.suptitle(title, x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("open_bundle", type=Path)
    parser.add_argument("closed_bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="睁眼 / 闭眼 · α 对照")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or any((p/".git").exists() for p in (output, *output.parents)):
        parser.error("Choose a new output directory outside the Git checkout")
    results = [load_bundle(root) for root in (args.open_bundle, args.closed_bundle)]
    with mpl.rc_context(_style()):
        figure, rows = make_comparison(results, args.title)
        save_eeg_figures({"alpha_comparison": figure}, output)
    with (output/"alpha_comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"inputs": [{"bundle": str(root.resolve()), "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                            "data_sha256": result["features"]["data_sha256"]}
                           for root, result in zip((args.open_bundle, args.closed_bundle), results)],
                "rows": rows, "aggregation": "arithmetic mean of O1/Oz/O2, within each recording",
                "time_axis": "seconds relative to the beginning of each recording; Welch window centers",
                "alpha_config": results[0]["method_results"]["spectral"]["alpha_power"]["config"]}
    (output/"comparison.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = [f"# {args.title}", "", "两段记录各取 Welch 实际覆盖范围；PSD 和时间曲线为 O1/Oz/O2 的算术平均。", "",
             "| 通道 | 睁眼 α (µV²) | 闭眼 α (µV²) | 睁眼 α 占比 | 闭眼 α 占比 |", "| --- | ---: | ---: | ---: | ---: |"]
    for i in range(3):
        opened, closed = rows[i], rows[i+3]
        lines.append(f"| {opened['channel_name']} | {opened['alpha_power_uv2']:.1f} | {closed['alpha_power_uv2']:.1f} | "
                     f"{opened['relative_alpha_percent']:.1f}% | {closed['relative_alpha_percent']:.1f}% |")
    lines += ["", "核对：闭眼 α 是否增强；增强是否集中在枕区；时间曲线是否显示持续变化或短时突增。", "",
              "[对照图](alpha_comparison.png) · [逐道数值与时间范围](alpha_comparison.tsv)", ""]
    (output/"review.md").write_text("\n".join(lines), encoding="utf-8")
    plt.close("all")
    print(output/"review.md")


if __name__ == "__main__":
    main()
