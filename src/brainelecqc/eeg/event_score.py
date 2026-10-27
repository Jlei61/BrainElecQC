"""Integer review scores for repeatable ECG/EOG-locked EEG waveforms.

The bins and repeatability gates are project review rules. Event detection and
epoching are supplied by the saved event analysis; raw samples stay unchanged.
"""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


DEFAULT_CONFIG = {"min_events": 20, "repeatability_min": 0.5,
                  "ratio_edges": (0.10, 0.25, 0.50), "points": (10, 8, 5, 2)}
RUN_STATUSES = {"computed", "no_events", "no_complete_epochs"}
PENDING_REASONS = {
    "missing_reference": "缺少辅助导联",
    "constant_reference": "辅助导联恒定，事件检测待核",
    "insufficient_duration": "片段长度不足，事件检测未执行",
    "insufficient_sampling_rate": "采样率不满足事件检测条件",
}


def _config(value):
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("event score config must be a mapping")
    supplied = {} if value is None else dict(value)
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("unknown event score configuration field")
    cfg = {**DEFAULT_CONFIG, **supplied}
    if isinstance(cfg["min_events"], bool) or not isinstance(cfg["min_events"], Integral) or cfg["min_events"] < 4:
        raise ValueError("min_events must be an integer of at least four")
    repeat = cfg["repeatability_min"]
    if isinstance(repeat, bool) or not isinstance(repeat, Real) or not np.isfinite(repeat) or not 0 <= repeat <= 1:
        raise ValueError("repeatability_min must be finite and within [0, 1]")
    try:
        edges, points = tuple(cfg["ratio_edges"]), tuple(cfg["points"])
    except TypeError as exc:
        raise ValueError("ratio_edges and points must be sequences") from exc
    if len(edges) != 3 or any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v <= 0 for v in edges):
        raise ValueError("ratio_edges must contain three positive finite numbers")
    if not all(a < b for a, b in zip(edges, edges[1:])):
        raise ValueError("ratio_edges must be strictly increasing")
    if len(points) != 4 or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v <= 10 for v in points):
        raise ValueError("points must contain four integers within [0, 10]")
    if points[0] != 10 or not all(a >= b for a, b in zip(points, points[1:])):
        raise ValueError("points must start at 10 and be nonincreasing")
    return {"min_events": int(cfg["min_events"]), "repeatability_min": float(repeat),
            "ratio_edges": [float(v) for v in edges], "points": [int(v) for v in points]}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer sample index")
    return int(value)


def _correlation(left, right):
    # Scaling avoids overflow and makes the zero-variation gate unit independent.
    left, right = left - np.mean(left), right - np.mean(right)
    a, b = np.max(np.abs(left)), np.max(np.abs(right))
    if not np.isfinite(a + b) or a == 0 or b == 0:
        return None
    left, right = left/a, right/b
    return float(np.clip(np.dot(left, right) / np.sqrt(np.dot(left, left)*np.dot(right, right)), -1, 1))


def score_event_coupling(result, kind, config=None):
    """Return a 10-point preliminary score with channel/event evidence.

    R is the peak-to-peak median locked waveform divided by median single-epoch
    peak-to-peak amplitude. Both chronological-half and odd/even median waveform
    correlations must pass before R can lower the score. Epoch bounds are
    half-open original sample intervals; adjacent intervals do not overlap.
    """
    if kind not in {"ecg", "eog"}:
        raise ValueError("kind must be ecg or eog")
    cfg = _config(config)
    f, event_result = result["features"], result["method_results"]["events"]
    block = event_result["by_kind"][kind]
    data = np.asarray(result["raw_data_volts"])
    ids, names = list(f["channel_ids"]), list(f["channel_names"])
    origin, stop = _integer(f["start_sample"], "start_sample"), _integer(f["stop_sample"], "stop_sample")
    fs, start_seconds = float(f["sampling_rate_hz"]), float(f["start_seconds"])
    if (not ids or len(set(ids)) != len(ids) or len(names) != len(ids) or data.ndim != 2
            or data.shape != (len(ids), stop-origin) or stop <= origin or origin < 0
            or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)
            or not np.isfinite(fs) or fs <= 0 or not np.isfinite(start_seconds)):
        raise ValueError("event score requires matching channel/sample data and a finite clock")
    for key, expected in (("channel_ids", ids), ("channel_names", names), ("start_sample", origin),
                          ("stop_sample", stop), ("sampling_rate_hz", fs), ("start_seconds", start_seconds)):
        if key in event_result and event_result[key] != expected:
            raise ValueError(f"event result {key} does not match input")
    metrics = {"kind": kind, "config": cfg, "detector_status": block.get("status"),
               "source_reference": block.get("source_reference"), "event_ids": [],
               "overlap_excluded_count": 0, "overlap_excluded_event_ids": [],
               "noncomplete_excluded_count": 0, "channels": [], "worst_channel_id": None,
               "selection_rule": "chronological greedy nonoverlap among complete averaged events",
               "ratio_definition": "ptp(median epochs) / median(ptp(single epochs))",
               "repeatability_definition": "Pearson r of chronological-half and odd/even median epochs",
               "score_basis": "project_review_bins", "epoch_processing": "raw samples; no baseline or filter"}

    def output(score, reason, evidence=None):
        return {"score": score, "max_points": 10, "reason": reason,
                "metrics": metrics, "evidence": evidence or []}

    status = block.get("status")
    if not block.get("source_reference") or status == "missing_reference":
        return output(None, PENDING_REASONS["missing_reference"])
    if status not in RUN_STATUSES:
        return output(None, PENDING_REASONS.get(status, f"事件检测未完成：{status}"))
    table, selected = list(block["event_table"]), list(block["selected_event_ids"])
    event_ids = [row["event_id"] for row in table]
    if len(set(event_ids)) != len(event_ids) or len(set(selected)) != len(selected) or not set(selected).issubset(event_ids):
        raise ValueError("event IDs must be unique and selected events must exist in the table")
    selected_set = set(selected)
    eligible = []
    for row in table:
        if row["event_id"] not in selected_set or not row["included_in_average"]:
            continue
        if not row["complete_epoch"]:
            metrics["noncomplete_excluded_count"] += 1
            continue
        sample = _integer(row["sample"], "event sample")
        a, b = _integer(row["epoch_start_sample"], "epoch start"), _integer(row["epoch_stop_sample"], "epoch stop")
        if not origin <= a <= sample < b <= stop or b-a < 2:
            raise ValueError("complete event epoch lies outside the input or has fewer than two samples")
        for key, value in (("time_seconds", sample), ("epoch_start_seconds", a), ("epoch_stop_seconds", b)):
            expected = start_seconds + (value-origin)/fs
            if key in row and (not np.isfinite(row[key]) or not np.isclose(row[key], expected, rtol=0, atol=1e-9)):
                raise ValueError("event time does not match the original sample clock")
        eligible.append(row)
    eligible.sort(key=lambda row: (row["sample"], row["event_id"]))
    metrics.update(selected_event_count=len(selected), complete_selected_event_count=len(eligible))
    if eligible:
        offsets = {(row["epoch_start_sample"]-row["sample"], row["epoch_stop_sample"]-row["sample"]) for row in eligible}
        if len(offsets) != 1:
            raise ValueError("event epochs must share the same sample offsets")
    actual, last_stop = [], origin
    for row in eligible:
        if row["epoch_start_sample"] < last_stop:
            metrics["overlap_excluded_event_ids"].append(row["event_id"])
        else:
            actual.append(row)
            last_stop = row["epoch_stop_sample"]
    metrics.update(event_ids=[row["event_id"] for row in actual],
                   overlap_excluded_count=len(metrics["overlap_excluded_event_ids"]),
                   nonoverlapping_event_count=len(actual))
    if len(actual) < cfg["min_events"]:
        return output(None, f"有效且不重叠事件 {len(actual)} 个，评分需要至少 {cfg['min_events']} 个")

    best = None
    invalid_channels = []
    for ci, (cid, name) in enumerate(zip(ids, names)):
        epochs = np.asarray([data[ci, row["epoch_start_sample"]-origin:row["epoch_stop_sample"]-origin]
                             for row in actual], dtype=float)
        channel = {"channel_id": cid, "raw_name": name, "event_count": len(actual),
                   "locked_median_peak_to_peak_uv": None, "median_single_epoch_peak_to_peak_uv": None,
                   "ratio": None, "chronological_half_correlation": None, "odd_even_correlation": None,
                   "repeatability_pass": False, "candidate_score": None, "status": "computed"}
        metrics["channels"].append(channel)
        if not np.isfinite(epochs).all():
            channel["status"] = "nonfinite_epoch_samples"
            invalid_channels.append(cid)
            continue
        median = np.median(epochs, axis=0)
        numerator, denominator = float(np.ptp(median)), float(np.median(np.ptp(epochs, axis=1)))
        if denominator <= 0 or not np.isfinite(numerator + denominator):
            channel["status"] = "unavailable_epoch_amplitude"
            invalid_channels.append(cid)
            continue
        ratio = numerator/denominator
        if not np.isfinite(ratio):
            channel["status"] = "unavailable_amplitude_ratio"
            invalid_channels.append(cid)
            continue
        half = len(epochs)//2
        chronological = _correlation(np.median(epochs[:half], axis=0), np.median(epochs[half:], axis=0))
        odd_even = _correlation(np.median(epochs[::2], axis=0), np.median(epochs[1::2], axis=0))
        reliable = all(value is not None and value >= cfg["repeatability_min"] for value in (chronological, odd_even))
        points = cfg["points"][int(np.searchsorted(cfg["ratio_edges"], ratio, side="right"))] if reliable else 10
        channel.update(locked_median_peak_to_peak_uv=numerator*1e6, median_single_epoch_peak_to_peak_uv=denominator*1e6,
                       ratio=ratio, chronological_half_correlation=chronological, odd_even_correlation=odd_even,
                       repeatability_pass=reliable, candidate_score=int(points))
        if reliable and (best is None or (points, -ratio) < (best[0], -best[1])):
            representative = actual[int(np.argmin(np.mean(np.abs(epochs-median), axis=1)))]
            best = (int(points), ratio, cid, name, representative)
    metrics["unassessed_channel_ids"] = invalid_channels
    evidence = []
    if best is not None:
        score, ratio, cid, name, row = best
        metrics["worst_channel_id"] = cid
        evidence = [{"kind": kind, "channel_id": cid, "raw_name": name, "event_id": row["event_id"],
                     "sample": row["sample"], "time_seconds": start_seconds+(row["sample"]-origin)/fs,
                     "start_sample": row["epoch_start_sample"], "stop_sample": row["epoch_stop_sample"],
                     "start_seconds": start_seconds+(row["epoch_start_sample"]-origin)/fs,
                     "stop_seconds": start_seconds+(row["epoch_stop_sample"]-origin)/fs,
                     "representative_rule": "used epoch with smallest mean absolute distance to median waveform",
                     "ratio": ratio, "candidate_score": score}]
    if invalid_channels:
        return output(None, f"{len(invalid_channels)} 道 EEG 的事件振幅不可评，需核对对应波形", evidence)
    if best is None:
        return output(10, f"已评 {len(actual)} 个不重叠事件，未见通过重复性检查的锁时波形")
    return output(score, f"{name} 锁时振幅比 {ratio:.2f}，前后半及奇偶事件形态一致；自动 {score}/10 分", evidence)
