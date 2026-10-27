"""Focused EEG review panels drawn from saved measurements and raw samples."""
from numbers import Integral

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..eeg.binding import validate_result_identity
from ..eeg.display import display_channel_name, display_channel_names
from ..eeg.event_score import score_event_coupling
from .eeg import _style


COLORS = ("#B35250", "#BF7A2C", "#8A68A3", "#277898", "#458A79", "#6E7880")
BAD_COLORS = ("#9D2438", "#D34A4A", "#B8665D", "#802B37", "#C63B67", "#A35042")


def _seconds(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _labels(result):
    return display_channel_names(result["features"]["channel_names"])


def _indices(result, indices):
    values = list(indices)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or not 0 <= i < len(result["raw_data_volts"]) for i in values)):
        raise ValueError("Choose distinct EEG channel indices within the input")
    return [int(i) for i in values]


def _spectrum(result, maximum):
    if isinstance(maximum, bool) or not np.isfinite(maximum) or maximum <= 2:
        raise ValueError("Maximum PSD frequency must be finite and greater than 2 Hz")
    psd = result["method_results"]["spectral"]["psd"]
    frequencies = np.asarray(psd["frequencies_hz"], dtype=float)
    visible = (frequencies >= 2) & (frequencies <= maximum)
    if visible.sum() < 2:
        raise ValueError("The PSD display requires at least two bins above 2 Hz")
    power = np.asarray(psd["power_v2_hz"], dtype=float)*1e12
    power = np.where(power > 0, power, np.nan)
    fs = result["features"]["sampling_rate_hz"]
    coverage = {"start_sample": psd["start_sample"],
                "stop_sample": psd["start_sample"]+psd["estimator"]["covered_samples"],
                "start_seconds": psd["start_seconds"],
                "stop_seconds": psd["start_seconds"]+psd["estimator"]["covered_samples"]/fs,
                "frequency_range_hz": [float(frequencies[visible][0]), float(frequencies[visible][-1])],
                "unit": "µV²/Hz", "estimator": psd["estimator"]}
    return frequencies[visible], power[:, visible], coverage


def _waveform_range(result, start, duration):
    if (isinstance(start, bool) or isinstance(duration, bool) or not np.isfinite(start)
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("Choose a finite positive waveform display interval")
    f = result["features"]
    fs, origin = f["sampling_rate_hz"], f["start_seconds"]
    left, right = round((start-origin)*fs), round((start+duration-origin)*fs)
    if not 0 <= left < right <= result["raw_data_volts"].shape[1] or right-left < 2:
        raise ValueError("Choose a waveform interval inside the recording with at least two samples")
    clock = {"start_sample": f["start_sample"]+left, "stop_sample": f["start_sample"]+right,
             "start_seconds": origin+left/fs, "stop_seconds": origin+right/fs}
    return left, right, origin+np.arange(left, right)/fs, clock


def _candidate_indices(result):
    candidates = list(result["method_results"]["bad_channels"]["candidates"])
    candidates.extend(result["method_results"].get("continuity", {}).get("candidates", []))
    return sorted({int(row["channel_index"]) for row in candidates})


def _channel_colors(result):
    rows = result["method_results"]["bad_channels"]["channels"]
    highlighted = _psd_highlights(result)
    candidates = highlighted + sorted(set(_candidate_indices(result))-set(highlighted),
                                      key=lambda i: (-(rows[i].get("hf_z") or 0), i))
    colors = {i: ("#277898", "#458A79", "#6E7880")[i % 3] for i in range(len(rows))}
    colors.update({i: BAD_COLORS[rank % len(BAD_COLORS)] for rank, i in enumerate(candidates)})
    return colors


def _psd_highlights(result):
    bad = result["method_results"]["bad_channels"]
    candidates = bad["candidates"]
    priorities = {"high_frequency_noise": 0, "low_correlation": 1}
    selected = []
    for kind in ("high_frequency_noise", "low_correlation", "other"):
        indices = {r["channel_index"] for r in candidates
                   if (r["kind"] == kind if kind != "other" else r["kind"] not in priorities)}
        field = "hf_z" if kind == "high_frequency_noise" else "low_correlation_fraction" if kind == "low_correlation" else "amplitude_z"
        selected.extend(sorted(indices-set(selected),
                               key=lambda i: (-abs(bad["channels"][i].get(field) or 0), i)))
    prep_count = len(selected)
    continuity = result["method_results"].get("continuity", {}).get("candidates", [])
    for row in sorted(continuity, key=lambda r: (-(r["stop_sample"]-r["start_sample"]), r["channel_index"])):
        if row["channel_index"] not in selected:
            selected.append(row["channel_index"])
    return selected[:max(6, prep_count)]


def make_all_channel_psd_figure(result, *, max_frequency_hz=120):
    """Plot every selected EEG channel, highlighting saved bad-channel candidates."""
    validate_result_identity(result)
    frequencies, power, coverage = _spectrum(result, max_frequency_hz)
    labels, highlighted = _labels(result), _psd_highlights(result)
    colors = _channel_colors(result)
    with mpl.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
        for i, name in enumerate(labels):
            if i in highlighted:
                continue
            line, = ax.plot(frequencies, power[i], color="#BEC5CA", lw=.65, alpha=.65,
                            label="其他 EEG 通道" if not ax.lines else "_nolegend_")
            line.set_gid(f"psd:channel:{i}")
        for i in highlighted:
            line, = ax.plot(frequencies, power[i], color=colors[i], lw=1.25, label=labels[i])
            line.set_gid(f"psd:channel:{i}")
        ax.set_gid("psd:all_channels")
        ax.set(xlabel="频率 (Hz)", ylabel="PSD (µV²/Hz)", yscale="log",
               xlim=(frequencies[0], frequencies[-1]))
        ax.set_title(f"全部 {len(labels)} 道 EEG · PSD {_seconds(coverage['start_seconds'])}–{_seconds(coverage['stop_seconds'])} s",
                     loc="left", fontweight="bold")
        ax.legend(fontsize=9, ncols=min(4, max(1, (len(highlighted)+2)//2)),
                  loc="upper center", bbox_to_anchor=(.5, -.16))
    return fig, {"data_sha256": result["features"]["data_sha256"], "psd": coverage,
                 "channel_ids": list(result["features"]["channel_ids"]),
                 "channel_names": list(result["features"]["channel_names"]),
                 "highlighted_channel_indices": highlighted,
                 "all_candidate_channel_indices": _candidate_indices(result),
                 "highlight_rule": "all PREP candidates, then continuity representatives up to six; remaining EEG shown in grey"}


def make_focused_bad_channel_figure(result, channel_indices, *, start_seconds, duration_seconds=4,
                                    max_frequency_hz=120):
    """Compare full-band PSD and same-scale raw traces within one reference group."""
    validate_result_identity(result)
    indices = _indices(result, channel_indices)
    if len(indices) > 6:
        raise ValueError("Choose at most six comparison channels")
    rows = result["method_results"]["bad_channels"]["channels"]
    groups = {rows[i]["group_id"] for i in indices}
    if len(groups) != 1:
        raise ValueError("Choose channels from one declared comparison group")
    frequencies, power, coverage = _spectrum(result, max_frequency_hz)
    left, right, times, clock = _waveform_range(result, start_seconds, duration_seconds)
    labels, colors = _labels(result), _channel_colors(result)
    prep_indices = {c["channel_index"] for c in result["method_results"]["bad_channels"]["candidates"]}
    local_indices = {c["channel_index"] for c in result["method_results"].get("continuity", {}).get("candidates", [])
                     if c["start_seconds"] < clock["stop_seconds"] and c["stop_seconds"] > clock["start_seconds"]}
    red_indices = [i for i in indices if i in prep_indices or i in local_indices]
    for rank, i in enumerate(red_indices):
        colors[i] = BAD_COLORS[rank % len(BAD_COLORS)]
    for i in set(indices)-set(red_indices):
        colors[i] = "#277898"
    values = np.asarray(result["raw_data_volts"])[indices, left:right]*1e6
    continuity = [r for r in result["method_results"].get("continuity", {}).get("candidates", [])
                  if r["channel_index"] in indices and r["start_seconds"] < clock["stop_seconds"]
                  and r["stop_seconds"] > clock["start_seconds"]]
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(np.max(finite)) if finite.size else 0, 1)*1.08
    with mpl.rc_context(_style()):
        fig = plt.figure(figsize=(12, max(4.7, len(indices)*1.3)), layout="constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.35))
        psd = fig.add_subplot(grid[0, 0])
        traces = [fig.add_subplot(cell) for cell in grid[0, 1].subgridspec(len(indices), 1, hspace=.05)]
        for i in indices:
            line, = psd.plot(frequencies, power[i], color=colors[i], lw=1.15, label=labels[i])
            line.set_gid(f"bad:psd:channel:{i}")
        psd.set_gid("bad:psd")
        psd.set(xlabel="频率 (Hz)", ylabel="PSD (µV²/Hz)", yscale="log",
                xlim=(frequencies[0], frequencies[-1]))
        psd.set_title(f"B  宽频 PSD · {_seconds(coverage['start_seconds'])}–{_seconds(coverage['stop_seconds'])} s",
                      loc="left", fontweight="bold")
        psd.legend(fontsize=9)
        for ax, i, y in zip(traces, indices, values):
            ax.plot(times, y, color=colors[i], lw=.7)
            for row in continuity:
                if row["channel_index"] == i:
                    patch = ax.axvspan(max(row["start_seconds"], clock["start_seconds"]),
                                      min(row["stop_seconds"], clock["stop_seconds"]), color=colors[i], alpha=.16)
                    patch.set_gid(f"bad:continuity:{row['kind']}:{i}:{row['start_sample']}")
            ax.set_gid(f"bad:raw:{i}")
            ax.set(xlim=(clock["start_seconds"], clock["stop_seconds"]), ylim=(-limit, limit), ylabel="µV")
            ax.text(.01, .93, labels[i], transform=ax.transAxes, va="top", color=colors[i], fontsize=9)
            if ax is not traces[-1]:
                ax.tick_params(labelbottom=False)
        traces[0].set_title(f"C  原始波形 · {_seconds(clock['start_seconds'])}–{_seconds(clock['stop_seconds'])} s · 同一刻度",
                            loc="left", fontweight="bold")
        traces[-1].set_xlabel("原始时间 (s)")
    return fig, {"data_sha256": result["features"]["data_sha256"], "group_id": next(iter(groups)),
                 "channel_indices": indices, "selected_channels": [rows[i] for i in indices],
                 "channel_colors": {str(i): colors[i] for i in indices},
                 "psd": coverage, "waveform": clock, "waveform_unit": "µV", "status": "pending_review",
                 "displayed_continuity_candidates": continuity}


def make_line_noise_summary_figure(result):
    """Show comparable 50/60-Hz evidence for every EEG channel and complete window."""
    validate_result_identity(result)
    line = result["method_results"]["spectral"]["line_noise"]
    names, windows = _labels(result), line["windows"]
    matrices = {hz: np.full((len(names), len(windows)), np.nan) for hz in (50, 60)}
    summaries, rows_by_hz = {}, {}
    alpha = float(line["multiplicity"]["family_alpha"])
    threshold = -np.log10(alpha)
    for hz in matrices:
        rows = [r for r in line["observations"] if r["nominal_hz"] == hz and r["status"] == "computed"]
        rows_by_hz[hz] = rows
        for row in rows:
            p = row["p_corrected"]
            if p is None or not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Computed line-noise p values must lie in [0, 1]")
            matrices[hz][row["channel_index"], row["window_index"]] = -np.log10(max(p, 1e-300))
        flagged = [r for r in rows if r["flagged"]]
        summaries[hz] = {"nominal_hz": hz, "computed_cells": len(rows),
                         "flagged_channel_indices": sorted({r["channel_index"] for r in flagged}),
                         "flagged_observations": flagged,
                         "peak_frequencies_hz": sorted({r["peak_frequency_hz"] for r in flagged})}
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    maximum = max(threshold*1.1, float(finite.max()) if finite.size else threshold)
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4.8, len(names)*.155+1.8)), layout="constrained")
        cmap = mpl.colormaps["cividis"].copy()
        cmap.set_bad("#E5E8EB")
        image = None
        for ax, hz in zip(axes, (50, 60)):
            summary = summaries[hz]
            ax.set_gid(f"line:{hz}")
            if windows:
                starts = np.array([r["start_seconds"] for r in windows])
                step = result["method_results"]["spectral"]["config"]["line_step_seconds"]
                edges = np.r_[starts, starts[-1]+step]
                image = ax.pcolormesh(edges, np.arange(len(names)+1)-.5, matrices[hz],
                                     cmap=cmap, vmin=0, vmax=maximum, shading="flat")
                ax.set_ylim(len(names)-.5, -.5)
                for row in summary["flagged_observations"]:
                    ax.plot(row["start_seconds"]+step/2, row["channel_index"], marker="s", markersize=3,
                            markerfacecolor="none", markeredgecolor="#F08258", markeredgewidth=.6, ls="none")
                ax.set_yticks(range(len(names)), names, fontsize=7.5 if len(names) > 24 else 9)
                for tick, i in zip(ax.get_yticklabels(), range(len(names))):
                    if i in summary["flagged_channel_indices"]:
                        tick.set_color("#B35250")
                        tick.set_fontweight("bold")
            else:
                ax.text(.5, .5, "完整检测窗不足", ha="center", transform=ax.transAxes)
            hits = len(summary["flagged_channel_indices"])
            peaks = summary["peak_frequencies_hz"]
            peak_text = (f" · 峰 {peaks[0]:g} Hz" if len(peaks) == 1 else
                         f" · 峰 {min(peaks):g}–{max(peaks):g} Hz" if peaks else "")
            state = f"{hits}/{len(names)} 道命中" if rows_by_hz[hz] else "待评"
            ax.set_title(f"{hz} Hz · {state}{peak_text}", loc="left", fontweight="bold", fontsize=10)
            ax.set_xlabel("原始时间 / 检测窗起点 (s)")
        if image is not None:
            bar = fig.colorbar(image, ax=list(axes), fraction=.035, pad=.025)
            bar.set_label(f"−log₁₀(校正 p) · 橙框 p < {alpha:g}")
            bar.ax.axhline(threshold, color="#F08258", lw=1.4)
        window_duration = line.get("settings", {}).get("window_samples", 0)/result["features"]["sampling_rate_hz"]
        fig.suptitle(f"工频检测 · 全部 EEG · 每窗 {window_duration:g} s", x=.01, ha="left", fontsize=12)
    return fig, {"data_sha256": result["features"]["data_sha256"],
                 "channel_names": list(result["features"]["channel_names"]), "display_names": names,
                 "channel_ids": list(result["features"]["channel_ids"]), "families": summaries,
                 "windows": windows, "multiplicity": line["multiplicity"],
                 "color_range": [0, maximum], "color_threshold": threshold,
                 "cell_value": "-log10(p_corrected); uncomputed cells are grey; orange boxes are saved flagged cells"}


def _centered_window(result, center, duration):
    f = result["features"]
    duration = min(duration, f["stop_seconds"]-f["start_seconds"])
    start = min(max(f["start_seconds"], center-duration/2), f["stop_seconds"]-duration)
    return _waveform_range(result, start, duration)


def _event_cluster_window(result, events, event, *, max_gap_seconds=10, max_span_seconds=20,
                          context_seconds=2, minimum_seconds=8, max_events=None):
    """Choose the bounded neighboring event cluster containing a real anchor."""
    ordered = sorted(events, key=lambda row: (row["time_seconds"], row["event_id"]))
    anchor = next(i for i, row in enumerate(ordered) if row["event_id"] == event["event_id"])
    left = right = anchor
    while True:
        if max_events is not None and right-left+1 >= max_events:
            break
        options = []
        if left > 0:
            gap = ordered[left]["time_seconds"]-ordered[left-1]["time_seconds"]
            span = ordered[right]["time_seconds"]-ordered[left-1]["time_seconds"]
            if gap <= max_gap_seconds and span <= max_span_seconds:
                options.append((abs(ordered[left-1]["time_seconds"]-event["time_seconds"]), gap, 0, left-1, right))
        if right+1 < len(ordered):
            gap = ordered[right+1]["time_seconds"]-ordered[right]["time_seconds"]
            span = ordered[right+1]["time_seconds"]-ordered[left]["time_seconds"]
            if gap <= max_gap_seconds and span <= max_span_seconds:
                options.append((abs(ordered[right+1]["time_seconds"]-event["time_seconds"]), gap, 1, left, right+1))
        if not options:
            break
        _, _, _, left, right = min(options)
    selected = ordered[left:right+1]
    begin, end = selected[0]["time_seconds"], selected[-1]["time_seconds"]
    interval = _centered_window(result, (begin+end)/2, max(minimum_seconds, end-begin+2*context_seconds))
    return interval, {"rule": "expand adjacent detected events by nearest distance to anchor; bounded gaps and total cluster span",
                      "max_gap_seconds": max_gap_seconds, "max_span_seconds": max_span_seconds,
                      "context_seconds": context_seconds, "minimum_seconds": minimum_seconds,
                      "target_event_count": max_events,
                      "anchor_event_id": event["event_id"], "cluster_event_ids": [r["event_id"] for r in selected],
                      "cluster_start_seconds": begin, "cluster_stop_seconds": end}


def _largest_eeg_event_change(result, events, channel_index):
    """Choose an abnormality display anchor from complete raw EEG event windows."""
    f = result["features"]
    values = np.asarray(result["raw_data_volts"])[channel_index]
    measurements, excluded = [], []
    for event in events:
        if not event["complete_epoch"]:
            excluded.append({"event_id": event["event_id"], "reason": "incomplete_epoch"})
            continue
        left = event["epoch_start_sample"]-f["start_sample"]
        right = event["epoch_stop_sample"]-f["start_sample"]
        if not 0 <= left < right <= len(values) or right-left < 2:
            raise ValueError("Complete event display window must lie inside the input")
        epoch = values[left:right]
        if not np.isfinite(epoch).all():
            excluded.append({"event_id": event["event_id"], "reason": "nonfinite_eeg_epoch"})
            continue
        measurements.append({"event_id": event["event_id"], "sample": event["sample"],
                             "time_seconds": event["time_seconds"],
                             "start_sample": event["epoch_start_sample"],
                             "stop_sample": event["epoch_stop_sample"],
                             "start_seconds": f["start_seconds"]+left/f["sampling_rate_hz"],
                             "stop_seconds": f["start_seconds"]+right/f["sampling_rate_hz"],
                             "eeg_peak_to_peak_uv": float(np.ptp(epoch))*1e6})
    selected = min(measurements, key=lambda row: (-row["eeg_peak_to_peak_uv"], row["sample"], row["event_id"])) if measurements else None
    event = next(row for row in events if row["event_id"] == selected["event_id"]) if selected else None
    return event, {"rule": "largest_displayed_eeg_peak_to_peak_in_complete_eog_epoch",
                   "purpose": "abnormality_display_selection",
                   "channel_index": channel_index, "channel_id": f["channel_ids"][channel_index],
                   "raw_channel_name": f["channel_names"][channel_index],
                   "processing": "raw samples; no baseline correction or filtering",
                   "tie_break": "earliest original sample then event ID",
                   "selected": selected, "evaluated_events": measurements, "excluded_events": excluded}


def make_aligned_event_figure(result, kind, *, channel_index=None, overview_seconds=None, zoom_seconds=2):
    """Draw local event-locked raw EEG above ECG/EOG on their original time axis.

    One stacked pair includes neighboring events, with zoom_seconds as its
    minimum duration. overview_seconds sets an explicit duration for this same
    pair. An isolated event uses a square figure. Selection metadata retains
    raw names, event identities and half-open sample intervals.
    """
    validate_result_identity(result)
    if kind not in {"ecg", "eog"}:
        raise ValueError("kind must be ecg or eog")
    for value in (zoom_seconds, overview_seconds):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
            raise ValueError("Event display durations must be finite and positive")
    score = score_event_coupling(result, kind)
    f, block = result["features"], result["method_results"]["events"]["by_kind"][kind]
    labels = _labels(result)
    worst = score["metrics"]["worst_channel_id"]
    if channel_index is not None:
        ci = _indices(result, [channel_index])[0]
        selection = "caller_selected_channel"
    elif worst is not None:
        ci = f["channel_ids"].index(worst)
        selection = "worst_repeatability_checked_channel"
    elif block["channel_summaries"]:
        ci = max(block["channel_summaries"], key=lambda r: r["mean_peak_to_peak_uv"] or 0)["channel_index"]
        selection = "largest_locked_mean_amplitude_for_review"
    else:
        ci, selection = 0, "first_eeg_channel_no_locked_summary"
    metadata = {"data_sha256": f["data_sha256"], "kind": kind, "channel_index": ci,
                "channel_id": f["channel_ids"][ci], "channel_name": f["channel_names"][ci],
                "display_channel_name": labels[ci],
                "channel_selection": selection, "source_reference": block["source_reference"],
                "event_score": score, "raw_processing": "original samples in µV; no filter or baseline correction"}
    if kind not in result["references"]:
        with mpl.rc_context(_style()):
            fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
            ax.set_axis_off()
            ax.text(.02, .75, f"{kind.upper()} 事件锁时 · 缺少辅助导联", transform=ax.transAxes, fontsize=13, fontweight="bold")
            ax.text(.02, .45, score["reason"], transform=ax.transAxes)
        metadata.update(display_status="pending_reference", t0=None, event=None)
        return fig, metadata
    auxiliary = np.asarray(result["references"][kind])
    matching_evidence = [e for e in score["evidence"] if e["channel_id"] == f["channel_ids"][ci]]
    event_selection_measurements = None
    if matching_evidence:
        event = next(e for e in block["event_table"] if e["event_id"] == matching_evidence[0]["event_id"])
        center, rule = event["time_seconds"], "score_representative_event_for_displayed_channel"
    elif block["event_table"]:
        event = None
        if kind == "eog":
            event, event_selection_measurements = _largest_eeg_event_change(result, block["event_table"], ci)
        if event is not None:
            rule = event_selection_measurements["rule"]
        else:
            event = max(block["event_table"], key=lambda e: abs(e["raw_reference_volts_at_event"]-np.median(auxiliary)))
            rule = "largest_auxiliary_event_deviation_no_complete_finite_eeg_epoch" if kind == "eog" else "largest_auxiliary_event_deviation_for_review"
        center = event["time_seconds"]
    else:
        event = None
        center = f["start_seconds"]+int(np.argmax(np.abs(auxiliary-np.median(auxiliary))))/f["sampling_rate_hz"]
        rule = "largest_auxiliary_deviation_no_detected_event"
    if overview_seconds is not None:
        waveform = _centered_window(result, center, overview_seconds)
        window_selection = {"rule": "caller_duration_around_selected_event_or_reference_time", "requested_seconds": overview_seconds}
    elif event is not None:
        waveform, window_selection = _event_cluster_window(
            result, block["event_table"], event,
            max_gap_seconds=3 if kind == "ecg" else 6, max_span_seconds=6 if kind == "ecg" else 10,
            context_seconds=.5, minimum_seconds=zoom_seconds, max_events=3)
    else:
        waveform = _centered_window(result, center, 6 if kind == "ecg" else 8)
        window_selection = {"rule": "local_raw_context_no_detected_event"}
    left, right, times, clock = waveform
    visible = [r for r in block["event_table"]
               if clock["start_seconds"] <= r["time_seconds"] < clock["stop_seconds"]]
    count = len(visible)
    layout = "consecutive_events_pair" if count > 1 else "single_event_square" if count == 1 else "raw_context_square"
    selected_score = next((row for row in score["metrics"]["channels"] if row["channel_id"] == f["channel_ids"][ci]), None)
    channel_points = selected_score["candidate_score"] if selected_score is not None else None
    display_status = "pending" if channel_points is None else "coupled_candidate" if channel_points < 10 else "no_material_repeatable_coupling"
    state = {"pending": "事件证据不足", "coupled_candidate": "检出重复同步", "no_material_repeatable_coupling": "未见重复同步"}[display_status]
    if display_status == "pending":
        if block["status"] == "constant_reference":
            state = "辅助导联恒定"
        elif block["status"] == "insufficient_duration":
            state = "片段时长不足"
        elif not block["event_table"]:
            state = "未定位到辅助事件"
        elif score["metrics"].get("nonoverlapping_event_count", 0) < score["metrics"]["config"]["min_events"]:
            state = "事件数不足"
        elif selected_score is not None and selected_score["status"] != "computed":
            state = "事件振幅不可评"
    elif display_status == "no_material_repeatable_coupling" and selected_score["repeatability_pass"]:
        state = "同步振幅较低"
    t0 = None if event is None else {"event_id": event["event_id"], "sample": event["sample"], "time_seconds": event["time_seconds"]}
    anchor_text = f"t₀ = {_seconds(center)} s" if t0 is not None else f"参考时刻 {_seconds(center)} s · 事件未定位"
    with mpl.rc_context(_style()):
        fig, (eeg, aux) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.4) if count > 1 else (6, 6),
                                      layout="constrained", gridspec_kw={"hspace": .03})
        for ax, values, color, name, source in (
            (eeg, np.asarray(result["raw_data_volts"])[ci], "#277898", labels[ci], "eeg"),
            (aux, auxiliary, "#6E7880", display_channel_name(block["source_reference"]["raw_name"]), kind)):
            ax.plot(times, values[left:right]*1e6, color=color, lw=.75)
            ax.set_gid(f"event:{kind}:waveform:{source}")
            for row in visible:
                chosen = event is not None and row["event_id"] == event["event_id"]
                marker = ax.axvline(row["time_seconds"], color="#B35250" if chosen else "#BF7A2C",
                                   lw=1.05 if chosen else .65, alpha=.9 if chosen else .45)
                marker.set_gid(f"event-marker:{row['event_id']}")
            ax.set(ylabel=f"{name}\n(µV)", xlim=(clock["start_seconds"], clock["stop_seconds"]))
        heading = f"连续 {count} 个检出事件" if count > 1 else "单个事件" if count == 1 else "局部原始波形"
        eeg.set_title(f"{heading} · {_seconds(clock['start_seconds'])}–{_seconds(clock['stop_seconds'])} s",
                      loc="left", fontsize=11, fontweight="bold")
        aux.set_xlabel("原始时间 (s)")
        fig.suptitle(f"{kind.upper()} 事件锁时 · {anchor_text}\n{state}",
                     x=.01, ha="left", fontsize=13, fontweight="bold")
    metadata.update(display_status=display_status, display_state_text=state, event_selection=rule, event=event, t0=t0,
                    event_selection_measurements=event_selection_measurements,
                    display_center_seconds=center, window_selection=window_selection,
                    layout=layout, layout_version="event-single-pair-3.0", rendered_scopes=["waveform"],
                    waveform=clock, overview=None, zoom=clock, displayed_channel_score=selected_score,
                    displayed_event_ids={"waveform": [r["event_id"] for r in visible]})
    return fig, metadata
