"""Plot original EEG/EMG and the saved high-frequency evidence on one clock."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from numbers import Integral, Real
from .eeg import _style, BLUE, ORANGE, RED
from ..eeg.binding import validate_result_identity
from ..eeg.display import display_channel_name


def make_muscle_figure(result, *, channel_index=None, duration_seconds=12):
    validate_result_identity(result)
    if channel_index is not None and (isinstance(channel_index, bool) or not isinstance(channel_index, Integral)):
        raise ValueError("Select an original integer EEG channel index")
    if (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, Real)
            or not np.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive and finite")
    block = result["method_results"]["muscle"]
    f = result["features"]
    fs, origin = f["sampling_rate_hz"], f["start_seconds"]
    n = result["raw_data_volts"].shape[1]
    candidates = block["candidates"]
    supported = [r for r in candidates if r["kind"] == "emg_supported_burst"]
    best = max(supported or candidates, key=lambda r: (r.get("max_reference_envelope_correlation") or 0, r["peak_z_score"]), default=None)
    ci = (best["channel_index"] if best else 0) if channel_index is None else channel_index
    if not 0 <= ci < len(f["channel_ids"]):
        raise ValueError("Select an original EEG channel")
    selected = [r for r in candidates if r["channel_index"] == ci]
    if best is None or best["channel_index"] != ci:
        supported_channel = [r for r in selected if r["kind"] == "emg_supported_burst"]
        best = max(supported_channel or selected, key=lambda r: r["peak_z_score"], default=None)
    center = (best["start_seconds"]+best["stop_seconds"])/2 if best else origin+duration_seconds/2
    a = max(0, min(max(0, n-round(duration_seconds*fs)), round((center-origin-duration_seconds/2)*fs)))
    b = min(n, a+round(duration_seconds*fs))
    if b <= a:
        raise ValueError("Muscle plot requires a nonempty original interval")
    t = origin+np.arange(a,b)/fs
    meta = {"channel_id": f["channel_ids"][ci], "channel_name": f["channel_names"][ci],
            "start_sample": f["start_sample"]+a, "stop_sample": f["start_sample"]+b,
            "start_seconds": float(t[0]), "stop_seconds": origin+b/fs,
            "selection": "strongest EMG-supported candidate, then high-frequency candidate",
            "all_candidate_intervals": selected,
            "visible_candidate_intervals": [{**r, "display_start_seconds": max(t[0], r["start_seconds"]),
                                              "display_stop_seconds": min(origin+b/fs,r["stop_seconds"])}
                                             for r in selected if r["stop_seconds"]>t[0] and r["start_seconds"]<origin+b/fs],
            "reference": block["reference"]}
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True,layout="constrained")
        axes[0].plot(t,result["raw_data_volts"][ci,a:b]*1e6,color=BLUE,lw=.7)
        axes[0].set(ylabel=f"{display_channel_name(f['channel_names'][ci])}\nEEG (µV)",title="A  原始 EEG 波形")
        reference = result["references"].get("emg")
        if reference is not None:
            axes[1].plot(t,reference[a:b]*1e6,color=ORANGE,lw=.7)
        else:
            axes[1].text(.5,.5,"补充同步 EMG 导联后复核",transform=axes[1].transAxes,ha="center")
        name = block["reference"]["source_reference"].get("raw_name", "EMG")
        axes[1].set(ylabel=f"{display_channel_name(name)}\nEMG (µV)",title="B  同步 EMG 波形")
        metrics=block["sample_metrics"]
        if len(metrics["z_score"]) > ci and metrics["z_score"][ci] is not None:
            axes[2].plot(t,np.asarray(metrics["z_score"][ci])[a:b],color=BLUE,lw=1,label="EEG 高频包络 z")
        if metrics["reference_z_score"] is not None:
            axes[2].plot(t,np.asarray(metrics["reference_z_score"])[a:b],color=ORANGE,lw=1,label="EMG 高频包络 z")
        axes[2].axhline(block["parameters"]["z_threshold"],color=BLUE,lw=.7,ls="--",label="EEG 筛查阈值")
        axes[2].axhline(block["parameters"]["reference_z_threshold"],color=ORANGE,lw=.7,ls=":",label="EMG 筛查阈值")
        lo,hi=block["parameters"]["effective_filter_band_hz"]
        axes[2].set(ylabel="包络 z",xlabel="原始时间 (s)",title=f"C  {lo:g}–{hi:g} Hz 包络 · 工频及谐波已处理")
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles+[Patch(facecolor=RED,alpha=.13),Patch(facecolor="#BCC2C7",alpha=.25)],
                       labels+["自动候选", "连续性 / 边界保护区"],loc="upper right",fontsize=8,ncol=2)
        for ax in axes:
            ax.set_xlim(t[0],origin+b/fs)
            if len(block["channels"]) > ci:
                for excluded in block["channels"][ci].get("excluded_intervals", []):
                    if excluded["stop_seconds"]>t[0] and excluded["start_seconds"]<t[-1]:
                        ax.axvspan(max(t[0],excluded["start_seconds"]),min(origin+b/fs,excluded["stop_seconds"]),color="#BCC2C7",alpha=.25)
            for item in selected:
                if item["stop_seconds"]>t[0] and item["start_seconds"]<t[-1]:
                    ax.axvspan(max(t[0],item["start_seconds"]),min(origin+b/fs,item["stop_seconds"]),color=RED,alpha=.13)
        for ax in axes[1:]:
            for excluded in block["reference"].get("excluded_intervals", []):
                if excluded["stop_seconds"]>t[0] and excluded["start_seconds"]<origin+b/fs:
                    ax.axvspan(max(t[0],excluded["start_seconds"]),min(origin+b/fs,excluded["stop_seconds"]),color="#BCC2C7",alpha=.25)
        status = "EEG 高频突发与 EMG 同步" if best and best["kind"]=="emg_supported_burst" else "宽带高频候选" if best else "本段肌电相关高频未检出"
        coverage = min([r.get("valid_seconds",0)/(n/fs) for r in block["channels"]]
                       + [block["reference"].get("valid_seconds",0)/(n/fs)])
        if block["status"] != "ok" or block["reference"]["status"] != "ok" or coverage < .95:
            status = (status + " · 有效检测范围受限") if best else "肌电检测条件不足"
        fig.suptitle(status,x=.01,ha="left",fontweight="bold")
        fig._brainelecqc_display_metadata=meta
        return fig,meta
