"""Default EEG evidence figures with explicit channel and interval selection."""
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..eeg.binding import validate_result_identity
from ..eeg.display import display_channel_names
from .eeg import _style, BLUE
from .focused import (make_all_channel_psd_figure, make_focused_bad_channel_figure,
                      make_line_noise_summary_figure, make_aligned_event_figure)
from .spectral import make_spectrum_figure, make_alpha_time_figure


def _names(result):
    names = result["features"]["channel_names"]
    return display_channel_names(names)


def _priority(result, count=8):
    ids = result["features"]["channel_ids"]
    found = list(dict.fromkeys(ids.index(x["channel_id"]) for x in result["decisions"]["review_queue"]))
    return (found + [i for i in range(len(ids)) if i not in found])[:count]


def _raw(result, start, duration):
    f, data = result["features"], result["raw_data_volts"]
    indices = _priority(result, 6)
    a = round((start-f["start_seconds"])*f["sampling_rate_hz"])
    b = min(data.shape[1], round((start+duration-f["start_seconds"])*f["sampling_rate_hz"]))
    times = f["start_seconds"]+np.arange(a, b)/f["sampling_rate_hz"]
    fig, axes = plt.subplots(len(indices), 1, figsize=(11, max(3, len(indices)*1.15)),
                             squeeze=False, sharex=True, layout="constrained")
    for ax, i in zip(axes[:, 0], indices):
        ax.plot(times, data[i, a:b]*1e6, color=BLUE, lw=.7)
        ax.set(xlim=(times[0], f["start_seconds"]+b/f["sampling_rate_hz"]), ylabel="µV")
        ax.set_title(_names(result)[i], loc="left", fontsize=10)
        ax.set_gid(f"initial:raw:{i}")
    axes[-1, 0].set_xlabel("原始时间 (s)")
    fig.suptitle("原始 EEG", x=.01, ha="left", fontsize=13, fontweight="bold")
    fig._brainelecqc_display_metadata = {"data_sha256": f["data_sha256"], "channel_indices": indices,
        "start_sample": f["start_sample"]+a, "stop_sample": f["start_sample"]+b,
        "start_seconds": float(times[0]), "stop_seconds": f["start_seconds"]+b/f["sampling_rate_hz"],
        "unit": "µV", "processing": "original samples"}
    return fig


def _finite(value, default=0.0):
    return float(value) if value is not None and np.isfinite(value) else default


def _window(result, center, duration):
    f = result["features"]
    length = min(duration, f["stop_seconds"]-f["start_seconds"])
    start = min(max(f["start_seconds"], center-length/2), f["stop_seconds"]-length)
    fs = f["sampling_rate_hz"]
    a, b = round((start-f["start_seconds"])*fs), round((start+length-f["start_seconds"])*fs)
    return {"start_seconds": f["start_seconds"]+a/fs, "duration_seconds": (b-a)/fs,
            "start_sample": f["start_sample"]+a, "stop_sample": f["start_sample"]+b}


def _comparison(result, selected, window):
    rows = result["method_results"]["bad_channels"]["channels"]
    candidates = list(result["method_results"]["bad_channels"]["candidates"])
    candidates += result["method_results"].get("continuity", {}).get("candidates", [])
    candidates = [r for r in candidates if r["start_seconds"] < window["start_seconds"]+window["duration_seconds"]
                  and r["stop_seconds"] > window["start_seconds"]]
    flagged = {r["channel_index"] for r in candidates}
    group = rows[selected[0]]["group_id"]
    eligible = [r for r in rows if r["group_id"] == group and r["channel_index"] not in selected]
    clean = [r for r in eligible if r["channel_index"] not in flagged and not r["flags"]]
    def rank(row):
        name = re.sub(r"[^a-z0-9]", "", row["channel_name"].lower()).removeprefix("eeg")
        return (name != "cz", abs(_finite(row["hf_z"])), abs(_finite(row["amplitude_z"])), row["channel_index"])
    if clean:
        return min(clean, key=rank)["channel_index"], "same_reference_group_without_candidates"
    if eligible:
        return None, "no_unflagged_comparison_in_reference_group"
    return None, "no_additional_channel_in_reference_group"


def build_focus_plan(result, *, display_start_seconds=None, display_duration_seconds=15):
    """Return a JSON-ready selection plan without drawing or changing measurements.

    Bad-channel panels include all automatic bad channels, with the strongest
    high-frequency candidate first. Continuous-value observations use pages by kind and
    reference group. Each page records the representative interval, nearby
    affected channels and the available same-group comparison channel.
    """
    validate_result_identity(result)
    f, bad = result["features"], result["method_results"]["bad_channels"]
    start = f["start_seconds"] if display_start_seconds is None else display_start_seconds
    if (isinstance(start, bool) or not np.isfinite(start) or not f["start_seconds"] <= start < f["stop_seconds"]
            or isinstance(display_duration_seconds, bool) or not np.isfinite(display_duration_seconds)
            or display_duration_seconds <= 0):
        raise ValueError("Display window must start inside the input and have a finite positive duration")
    a = round((start-f["start_seconds"])*f["sampling_rate_hz"])
    b = min(result["raw_data_volts"].shape[1], round((start+display_duration_seconds-f["start_seconds"])*f["sampling_rate_hz"]))
    if b-a < 2:
        raise ValueError("Display window must contain at least two original samples")
    rows, candidates = bad["channels"], bad["candidates"]
    hf = [r["channel_index"] for r in candidates if r["kind"] == "high_frequency_noise"]
    if hf:
        index = max(hf, key=lambda i: (_finite(rows[i]["hf_z"], -np.inf), -i))
        rule = "highest_saved_high_frequency_z_candidate"
    elif candidates:
        index = min(candidates, key=lambda r: (r["start_sample"], r["channel_index"]))["channel_index"]
        rule = "earliest_saved_bad_channel_candidate"
    else:
        index, rule = 0, "first_channel_no_saved_bad_channel_candidate"
    duration = min(4.0, display_duration_seconds, f["stop_seconds"]-f["start_seconds"])
    center = start+duration/2
    if display_start_seconds is None and hf and bad["windows"]:
        values = np.asarray(bad["window_metrics"]["hf_z"][index], dtype=float)
        if np.isfinite(values).any():
            wi = int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))
            center = (bad["windows"][wi]["start_seconds"]+bad["windows"][wi]["stop_seconds"])/2
            rule += "; window with highest saved per-window high-frequency z"
    window = _window(result, center, duration)
    from ..eeg.scoring import compute_quality_score
    score = compute_quality_score(result)
    channel_score = next(r for r in score["rows"] if r["dimension_id"] == "channels")
    metrics = channel_score["score_metrics"]
    required = metrics.get("automatic_bad_channel_ids", [])+metrics.get("unlocalized_channel_ids", [])
    selected = list(dict.fromkeys([index]+[f["channel_ids"].index(cid) for cid in required]))
    groups = {}
    for i in selected:
        groups.setdefault(rows[i]["group_id"], []).append(i)
    pages = []
    for members in groups.values():
        for offset in range(0, len(members), 5):
            chosen = members[offset:offset+5]
            control, control_rule = _comparison(result, chosen, window)
            key = "02_bad_channels" if not pages else f"02_bad_channels_comparison_{len(pages)+1:02d}"
            pages.append({"figure_key": key, "kind": "channel_comparison",
                "channel_indices": chosen+([] if control is None else [control]),
                "candidate_channel_indices": chosen if candidates else [], "control_channel_index": control,
                "control_selection": control_rule, "selection_rule": rule+"; all automatic bad channels, grouped and paginated",
                **window})
    continuity = result["method_results"].get("continuity", {}).get("candidates", [])
    suffixes = {"near_zero_flatline": "dropout", "calibration_endpoint_plateau": "range_plateau",
                "constant_value": "constant", "nonfinite_input": "missing"}
    for kind, suffix in suffixes.items():
        by_group = {}
        for row in continuity:
            if row["kind"] == kind:
                by_group.setdefault(rows[row["channel_index"]]["group_id"], []).append(row)
        for page_number, group_rows in enumerate(by_group.values(), 1):
            representative = min(group_rows, key=lambda r: (-(r["stop_sample"]-r["start_sample"]), r["start_sample"], r["channel_index"]))
            center = (representative["start_seconds"]+representative["stop_seconds"])/2
            minimum_duration, margin = (2, 1) if kind == "near_zero_flatline" else (4, 2)
            window = _window(result, center, max(minimum_duration, representative["stop_seconds"]-representative["start_seconds"]+margin))
            stop = window["start_seconds"]+window["duration_seconds"]
            nearby = [r for r in group_rows if r["start_seconds"] < stop and r["stop_seconds"] > window["start_seconds"]]
            selected = list(dict.fromkeys([representative["channel_index"]]+[r["channel_index"] for r in nearby]))[:3]
            control, control_rule = _comparison(result, selected, window)
            group_id = rows[representative["channel_index"]]["group_id"]
            group_indices = [r["channel_index"] for r in rows if r["group_id"] == group_id]
            affected = sorted({r["channel_index"] for r in group_rows})
            pages.append({"figure_key": f"02_bad_channels_{suffix}"+(f"_{page_number:02d}" if len(by_group) > 1 else ""),
                          "kind": kind, "channel_indices": selected+([] if control is None else [control]),
                          "candidate_channel_indices": selected, "control_channel_index": control,
                          "group_channel_indices": group_indices, "group_affected_channel_indices": affected,
                          "all_group_channels_have_this_kind": set(group_indices) == set(affected),
                          "control_selection": control_rule, "representative_candidate": representative,
                          "selection_rule": "longest saved interval per kind/reference group; nearby affected channels then comparison",
                          **window})
    return {"schema_version": "eeg-focused-figures-1.0", "data_sha256": f["data_sha256"],
            "initial": {"start_seconds": start, "duration_seconds": display_duration_seconds},
            "bad_channel_pages": pages,
            "event_references": {kind: kind in result["references"] for kind in ("ecg", "eog")}}


def make_paper_eeg_figures(result, *, title="EEG QC", display_start_seconds=None, display_duration_seconds=15,
                           figure_keys=None):
    """Render focused panels; each figure carries private JSON-ready display metadata.

    ``figure_keys`` accepts the saved report's figure inventory when replaying a
    plot configuration. The saved measurements determine the regenerated pages.
    """
    plan = build_focus_plan(result, display_start_seconds=display_start_seconds,
                            display_duration_seconds=display_duration_seconds)
    figures = {}
    def add(name, pair, selection=None):
        figure, metadata = pair
        figure._brainelecqc_display_metadata = {**metadata, "title": title}
        if selection is not None:
            figure._brainelecqc_display_metadata["selection_plan"] = selection
        figures[name] = figure
    with mpl.rc_context(_style()):
        figures["01_initial"] = _raw(result, plan["initial"]["start_seconds"], plan["initial"]["duration_seconds"])
        for page in plan["bad_channel_pages"]:
            add(page["figure_key"], make_focused_bad_channel_figure(result, page["channel_indices"],
                start_seconds=page["start_seconds"], duration_seconds=page["duration_seconds"]), page)
            if page.get("all_group_channels_have_this_kind"):
                labels = {"near_zero_flatline": "近零恒值", "calibration_endpoint_plateau": "量程端点平台",
                          "constant_value": "恒值片段", "nonfinite_input": "缺失样本"}
                figures[page["figure_key"]].suptitle(
                    f"坏道复核 · {labels[page['kind']]} · 本组全部 {len(page['group_channel_indices'])} 道命中 · 图示 {len(page['channel_indices'])} 道",
                    x=.01, ha="left", fontsize=11)
        figures["03_psd_alpha"] = make_spectrum_figure(result)
        add("03_all_channels_psd", make_all_channel_psd_figure(result))
        add("04_line_noise", make_line_noise_summary_figure(result))
        for number, kind in enumerate(("ecg", "eog"), 5):
            if plan["event_references"][kind]:
                add(f"{number:02d}_{kind}", make_aligned_event_figure(result, kind))
        figures["08_alpha_time"] = make_alpha_time_figure(result)
        if "muscle" in result["method_results"] and "emg" in result["references"]:
            from .muscle import make_muscle_figure
            add("10_muscle", make_muscle_figure(result))
    return figures
