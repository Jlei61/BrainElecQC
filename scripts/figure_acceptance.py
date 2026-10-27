"""Local QC figures with calibrated traces and illustrative candidate metrics.

Evidence includes source units, sample ranges and actual review status.
Inputs, figures and review decisions are stored outside the code repository.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.signal import welch


def configure():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def require_external(path):
    path = Path(path).expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    if path == root or root in path.parents:
        raise ValueError("Acceptance output must be outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, output, stem):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def candidate_metrics(data, fs, neural_indices, ecg_index=None):
    """Small transparent demonstration: flags remain pending, never exclusions."""
    if data.ndim != 2 or not np.isfinite(data).all() or fs <= 0:
        raise ValueError("Need finite channel x sample volts and a positive sample rate")
    rms = np.std(data, axis=1)
    pool = rms[neural_indices]
    median = float(np.median(pool[pool > 1e-12])) if np.any(pool > 1e-12) else 0
    freq, power = welch(data, fs, nperseg=min(data.shape[1], round(fs * 2)), axis=-1)
    denominator = power[:, (freq >= 1) & (freq <= min(150, fs / 2))].sum(axis=1)
    line = power[:, ((abs(freq - 50) <= 1) | (abs(freq - 60) <= 1))].sum(axis=1)
    high = power[:, (freq >= 70) & (freq <= min(150, fs / 2)) & (abs(freq - 100) > 1) & (abs(freq - 120) > 1)].sum(axis=1)
    result = []
    for index in neural_indices:
        flags = []
        if np.ptp(data[index]) < 0.2e-6:
            flags.append("flat")
        if median and rms[index] > 8 * median:
            flags.append("amplitude_outlier")
        line_ratio = float(line[index] / denominator[index]) if denominator[index] else 0
        hf_ratio = float(high[index] / denominator[index]) if denominator[index] else 0
        if line_ratio > 0.35:
            flags.append("line_noise")
        if hf_ratio > 0.30:
            flags.append("high_frequency_candidate")
        corr = None
        if ecg_index is not None and rms[index] > 0 and rms[ecg_index] > 0:
            corr = float(np.corrcoef(data[index], data[ecg_index])[0, 1])
            if abs(corr) > 0.6:
                flags.append("ECG_coupling_candidate")
        result.append({"index": index, "flags": flags, "status": "pending",
                       "rms_uV": float(rms[index] * 1e6), "line_power_fraction": line_ratio,
                       "high_frequency_fraction": hf_ratio, "ecg_correlation": corr})
    return result


def synthetic_case(modality):
    seed = {"EEG": 401, "SEEG": 402, "ECoG": 403}[modality]
    rng = np.random.default_rng(seed)
    fs, seconds = 500, 12
    t = np.arange(fs * seconds) / fs
    data = np.array([10e-6 * np.sin(2*np.pi*(8+i*.3)*t+i) + rng.normal(0, 1e-6, len(t)) for i in range(10)])
    ecg = sum(np.exp(-((t - beat) / 0.025)**2) for beat in np.arange(.5, seconds, 1.0)) * 150e-6
    emg = rng.normal(0, 70e-6, len(t)) * ((t >= 4) & (t < 6))
    data[0] = 0
    data[1] += rng.normal(0, 180e-6, len(t))
    data[2] += 60e-6*np.sin(2*np.pi*50*t)
    data[3] += ecg
    data[4] += emg
    data[8], data[9] = ecg, emg
    truth = {"flat": 0, "noisy": 1, "line": 2, "ecg": 3, "emg_like": 4,
             "clean_controls": [5, 6, 7], "seed": seed}
    return data, fs, [f"Ch {i+1:02d}" for i in range(8)] + ["ECG ref", "EMG ref"], truth


def triptych(data, fs, names, metrics, output, alias, modality, start_seconds=0, simulated=False, annotation_note=""):
    by_index = {item["index"]: item for item in metrics}
    flagged = [i for i in by_index if by_index[i]["flags"]]
    selected = list(dict.fromkeys(flagged[:5] + list(by_index)))[:6]
    if not selected:
        raise ValueError("No explicitly selected neural channels")
    t = start_seconds + np.arange(data.shape[1]) / fs
    spacing = max(50, float(np.percentile(np.abs(data[selected]*1e6), 99.5)) * 2.4)
    colors = {"raw": "#34495e", "candidate": "#b86b16", "retained": "#166b72"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    titles = ["1  Before QC", "2  Automatic candidates / before review",
              "3  After SIMULATED review" if simulated else "3  Human review: pending"]
    masked = data.copy()
    decisions = []
    if simulated:
        # This is an explicit scripted demonstration, never a real reviewer.
        for index in (0, 1):
            masked[index] = np.nan
            decisions.append({"channel_index": index, "action": "exclude_channel", "reviewer": "SIMULATION"})
        masked[4, round(4*fs):round(6*fs)] = np.nan
        decisions.append({"channel_index": 4, "action": "exclude_interval", "start_sample": round(4*fs),
                          "stop_sample": round(6*fs), "reviewer": "SIMULATION"})
    for panel, (ax, title) in enumerate(zip(axes, titles)):
        shown = masked if panel == 2 and simulated else data
        for row, index in enumerate(selected):
            base = (len(selected)-1-row)*spacing
            color = colors["candidate"] if panel == 1 and index in flagged else colors["raw"]
            if panel == 2 and simulated:
                color = colors["retained"]
            ax.plot(t, shown[index]*1e6+base, linewidth=.55, color=color)
            if panel == 1 and by_index[index]["flags"]:
                ax.text(t[0], base+spacing*.30,
                        ", ".join(by_index[index]["flags"]).replace("_", " "),
                        va="bottom", fontsize=7, color=colors["candidate"],
                        bbox={"facecolor":"white", "alpha":.85, "edgecolor":"none", "pad":1})
            if panel == 2 and simulated and index in (0, 1):
                ax.text(t[len(t)//2], base, "Excluded channel (mask only)", ha="center", fontsize=8, color="#7b7b7b")
        if panel == 2 and simulated:
            ax.axvspan(4, 6, facecolor="#166b72", alpha=.08)
        ax.set_title(title, fontsize=10, loc="left", pad=14)
        ax.set_xlabel("Source-clock time (s)")
        ax.set_ylim(-spacing*.6, (len(selected)-.4)*spacing)
        ax.set_yticks([(len(selected)-1-r)*spacing for r in range(len(selected))], [names[i] for i in selected])
        ax.grid(axis="x", alpha=.15)
    axes[0].set_ylabel(f"Calibrated amplitude; row spacing {spacing:.0f} µV")
    fig.suptitle(f"{modality} · {alias} · " + ("SYNTHETIC known-truth demonstration" if simulated else "REAL recording"), x=.07, ha="left", fontsize=14, weight="bold")
    if annotation_note:
        fig.text(.07, .87, annotation_note, fontsize=9, color="#7a581b")
    fig.text(.07, .02, "Same samples, units, scale and channel order in all panels. "
             + ("Review: scripted simulation." if simulated else "Review: awaiting a human decision."), fontsize=9)
    fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.79, wspace=.10)
    save(fig, output, alias)
    return {"selected_channel_indices": selected, "spacing_uV": spacing, "simulated_decisions": decisions,
            "review_status": "simulated_only" if simulated else "pending", "metrics": metrics}


def workflow(output):
    fig, ax = plt.subplots(figsize=(13.6, 7.0))
    ax.set(xlim=(0, 14), ylim=(0, 8)); ax.axis("off")
    labels = [
        (0.3, 5.4, "1  Input", "EEG / SEEG / ECoG\nSignal files + recording metadata"),
        (4.9, 5.4, "2  Read", "Calibrated samples + stable identity\nChannels / units / time segments"),
        (9.5, 5.4, "3  Extract features", "PSD / alpha / 50 + 60 Hz\nBad channels + artifact evidence"),
        (9.5, 2.7, "4  Report", "Summary + waveforms + spectra\nChannel / time / reason"),
        (4.9, 2.7, "5  Human review", "Confirm / reject / resize / add\nSave decisions in the host"),
        (0.3, 2.7, "6  Deliver", "Channel + interval masks\nReview version + source mapping"),
    ]
    for x,y,title,body in labels:
        ax.add_patch(FancyBboxPatch((x,y),4.2,1.75,boxstyle="round,pad=0.12",facecolor="#edf5f5",edgecolor="#166b72",linewidth=1.3))
        ax.text(x+.15,y+1.3,title,fontsize=12,weight="bold",va="top")
        ax.text(x+.15,y+.8,body,fontsize=10,va="top",linespacing=1.7)
    for a,b in [((4.55,6.3),(4.7,6.3)),((9.15,6.3),(9.3,6.3)),((11.6,5.2),(11.6,4.65)),((9.3,3.55),(9.15,3.55)),((4.7,3.55),(4.55,3.55))]:
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"-|>","color":"#34495e","lw":1.5})
    ax.text(.25,7.65,"BrainElecQC · product workflow",fontsize=19,weight="bold")
    ax.text(.3,1.7,"Python computation + reports  →  Linxi processor  →  host review + downstream analysis",fontsize=12,color="#166b72")
    ax.text(.3,.9,"Review comparison: original signals  /  automatic findings  /  human decisions",fontsize=11)
    ax.text(.3,.35,"Product design · implementation sequence: overallplan.md",fontsize=10,color="#666666")
    save(fig, output, "qc-workflow")


def real_case(case, output):
    from brainelecqc.readers import open_reader
    path = Path(case["path"])
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    reader = open_reader(path, case["alias"], case.get("format_hint"))
    desc = reader.describe()
    stream = desc["streams"][case.get("stream_index", 0)]
    segment = stream["segments"][case.get("segment_index", 0)]
    fs = stream["sampling_frequency_hz"]
    start = segment["start_sample"] + round(case.get("offset_seconds", 0)*fs)
    stop = min(start+round(case.get("duration_seconds", 12)*fs), segment["stop_sample"])
    channels = stream["channels"]
    # Neural channels are declared in the local manifest, not inferred as a fact.
    neural = case["neural_indices"]
    for index in neural + ([case["ecg_index"]] if case.get("ecg_index") is not None else []):
        if channels[index].get("unit") != "V":
            raise ValueError("The selected voltage channels need explicit V calibration")
    data = reader.read_window(stream["stream_id"], segment["segment_id"],
                              [c["channel_id"] for c in channels], start, stop)
    metrics = candidate_metrics(data, fs, neural, case.get("ecg_index"))
    names = [c.get("raw_name", c.get("label", c.get("name", f"Ch {i+1}"))) for i,c in enumerate(channels)]
    time_start = segment["recording_start_seconds"]+(start-segment["start_sample"])/fs
    note = case.get("annotation_note", "")
    result = triptych(data, fs, names, metrics, output, case["alias"], case["modality"], time_start, annotation_note=note)
    result.update(source=str(path), stream=stream, start_sample=start, stop_sample=stop,
                  unit="V", data_sha256=hashlib.sha256(data.tobytes()).hexdigest(), warnings=desc.get("warnings", []),
                  source_size_mtime_unchanged=before == (path.stat().st_size, path.stat().st_mtime_ns))
    if case.get("reference_indices"):
        indices = [case.get("focus_index", neural[0])] + case["reference_indices"]
        fig, axes = plt.subplots(len(indices), 1, figsize=(12, 6), sharex=True)
        t = time_start + np.arange(data.shape[1])/fs
        for ax, index in zip(axes, indices):
            ax.plot(t, data[index]*1e6, color="#34495e", linewidth=.7)
            ax.set_ylabel(names[index]+"\nµV")
            ax.grid(axis="x", alpha=.2)
        axes[-1].set_xlabel("Source-clock time (s)")
        fig.suptitle(case["alias"]+" · synchronous ECG / EMG reference evidence", fontsize=14)
        fig.text(.13,.02,"Calibrated; each row has its own µV scale. Synchronous reference signals for review. " + case.get("source_credit", ""),fontsize=8)
        fig.subplots_adjust(left=.14,bottom=.12,top=.9,hspace=.25)
        save(fig, output, case["alias"]+"-references")
    result["external_annotations"] = case.get("external_annotations", [])
    if hasattr(reader, "close"):
        reader.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", help="External JSON with cases; never committed")
    args = parser.parse_args()
    output = require_external(args.output)
    configure(); workflow(output)
    results = {"scope": "Illustrative candidate metrics with explicit review status", "synthetic": {}, "real": {}}
    for modality in ("EEG", "SEEG", "ECoG"):
        data, fs, names, truth = synthetic_case(modality)
        metrics = candidate_metrics(data, fs, list(range(8)), 8)
        alias = "synthetic-"+modality.lower()
        results["synthetic"][modality] = triptych(data, fs, names, metrics, output, alias, modality, simulated=True)
        results["synthetic"][modality]["truth"] = truth
    if args.manifest:
        for case in json.loads(Path(args.manifest).read_text())["cases"]:
            results["real"][case["alias"]] = real_case(case, output)
    (output / "figure-evidence.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved workflow, 3 synthetic comparisons and {len(results['real'])} real comparisons outside Git")


if __name__ == "__main__":
    main()
