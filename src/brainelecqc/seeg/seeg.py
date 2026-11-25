"""Bounded, segment-preserving SEEG basic QC scan over the verified catalog.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import math
from numbers import Real

import numpy as np

from .seeg_channels import resolve_seeg_channels
from .seeg_metrics import compute_relations, compute_window_metrics
from .seeg_scoring import compute_quality_score, score_window
from .seeg_waveform import (WAVEFORM_DEFAULTS, WAVEFORM_VERSION, WaveformTracker,
                            attach_waveform_events, preflight_waveform, validate_waveform_config)


SEEG_ALGORITHM_VERSION = "seeg-basic-2"
_DEFAULTS = {**WAVEFORM_DEFAULTS, "window_seconds": 5.0, "line_frequency": 50, "flat_tolerance_v": 0.0,
             "flat_min_seconds": .5, "clipping_fraction": .01, "clipping_min_seconds": .02,
             "line_noise_db": 12.0, "line_noise_fraction": .05, "max_windows": 20000,
             "max_scanned_samples": 100000000, "max_window_bytes": 16777216,
             "max_relation_evaluations": 1000000, "ranges": None}


def _config(config):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - set(_DEFAULTS):
        raise ValueError("Unknown SEEG QC configuration fields")
    result = {**_DEFAULTS, **deepcopy(config)}
    for key, value in result.items():
        if key == "ranges":
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("{} must be a finite number".format(key))
        if value < 0 or (key != "flat_tolerance_v" and value == 0):
            raise ValueError("{} must be positive".format(key))
    if result["line_frequency"] not in (50, 60):
        raise ValueError("line_frequency must be 50 or 60")
    for key in ("clipping_fraction", "line_noise_fraction"):
        if result[key] > 1:
            raise ValueError("{} must be in (0,1]".format(key))
    for key in ("max_windows", "max_scanned_samples", "max_window_bytes", "max_relation_evaluations"):
        if int(result[key]) != result[key] or result[key] > _DEFAULTS[key]:
            raise ValueError("{} must be an integer within the supported resource limit {}".format(key, _DEFAULTS[key]))
    if result["window_seconds"] > 60:
        raise ValueError("window_seconds must be <=60; select ranges for long recordings")
    validate_waveform_config(result)
    return result


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    return value


def _plan(catalog, config):
    streams = catalog.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("No readable streams in catalog")
    lookup, spans, gaps = {}, {}, []
    total = 0.0
    for stream in streams:
        sid, fs = stream["stream_id"], stream["sampling_frequency_hz"]
        if not isinstance(sid, str) or not sid or sid in lookup:
            raise ValueError("Streams require unique nonempty IDs")
        if isinstance(fs, bool) or not isinstance(fs, Real) or not math.isfinite(fs) or fs <= 0:
            raise ValueError("sampling_frequency_hz must be finite and positive")
        if not isinstance(stream["channels"], list) or not stream["channels"]:
            raise ValueError("A stream requires channels")
        if round(config["window_seconds"] * fs) < 2:
            raise ValueError("window_seconds must contain at least two samples")
        lookup[sid] = stream
        previous_stop, previous_time_stop = None, None
        for segment in stream["segments"]:
            key = (sid, segment["segment_id"])
            start, stop = _integer(segment["start_sample"], "start_sample"), _integer(segment["stop_sample"], "stop_sample")
            onset = segment["recording_start_seconds"]
            if key in spans or start < 0 or stop <= start or (previous_stop is not None and start < previous_stop):
                raise ValueError("Segment samples must be distinct, ordered nonempty intervals")
            if isinstance(onset, bool) or not isinstance(onset, Real) or not math.isfinite(onset):
                raise ValueError("Segment recording_start_seconds must be finite")
            if previous_time_stop is not None and onset < previous_time_stop - 1e-9:
                raise ValueError("Segment clocks overlap")
            if previous_time_stop is not None and onset > previous_time_stop + 1e-9:
                gaps.append({"stream_id": sid, "start_seconds": previous_time_stop,
                             "stop_seconds": onset, "duration_seconds": onset - previous_time_stop})
            previous_stop, previous_time_stop = stop, onset + (stop - start) / fs
            spans[key] = segment
            total += (stop - start) / fs * len(stream["channels"])
    ranges = config["ranges"]
    if ranges is None:
        ranges = [{"stream_id": sid, "segment_id": segment_id, "start_sample": span["start_sample"],
                   "stop_sample": span["stop_sample"]} for (sid, segment_id), span in spans.items()]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("ranges must be a nonempty list or null")
    occupied, plan = defaultdict(list), []
    windows, samples, evaluations = 0, 0, 0
    for item in ranges:
        fields = {"stream_id", "segment_id", "start_sample", "stop_sample"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Each range requires stream_id/segment_id/start_sample/stop_sample")
        key = (item["stream_id"], item["segment_id"])
        span = spans.get(key)
        start, stop = _integer(item["start_sample"], "start_sample"), _integer(item["stop_sample"], "stop_sample")
        if span is None or not span["start_sample"] <= start < stop <= span["stop_sample"]:
            raise ValueError("Range must stay inside a known segment")
        if any(start < b and stop > a for a, b in occupied[key]):
            raise ValueError("Scan ranges must not overlap")
        occupied[key].append((start, stop))
        stream = lookup[key[0]]
        size = int(round(config["window_seconds"] * stream["sampling_frequency_hz"]))
        count = len(stream["channels"])
        nwin = math.ceil((stop - start) / size)
        windows += nwin * count
        samples += (stop - start) * count
        evaluations += nwin * count * (count - 1) // 2
        if min(size, stop - start) * count * 8 > config["max_window_bytes"]:
            raise ValueError("Scan window exceeds max_window_bytes; reduce window_seconds")
        plan.append((stream, span, start, stop, size))
    if windows > config["max_windows"] or samples > config["max_scanned_samples"] or evaluations > config["max_relation_evaluations"]:
        raise ValueError("Scan resource limit exceeded; select explicit ranges or a shorter recording")
    return plan, deepcopy(ranges), gaps, total


def scan_seeg(catalog, read_window, channel_map=None, config=None):
    """Compute JSON-safe evidence; read_window returns channel × physical sample."""
    config = _config(config)
    plan, ranges, gaps, total = _plan(catalog, config)
    preflight_waveform(plan, config)
    plan.sort(key=lambda p: (p[0]["stream_id"], p[1]["start_sample"], p[2]))
    mapping = resolve_seeg_channels(catalog, channel_map)
    by_id = {item["channel_id"]: item for item in mapping}
    windows, relations, spectra, waveform_events = [], [], {}, []
    trackers = {}
    for plan_index, (stream, segment, start, stop, size) in enumerate(plan):
        channels = stream["channels"]
        ids = [channel["channel_id"] for channel in channels]
        mapped = [by_id[cid] for cid in ids]
        frequency = stream["sampling_frequency_hz"]
        if not trackers:
            for row, channel in enumerate(channels):
                item = mapped[row]
                kind = item["channel_type"]
                auxiliary = kind == "unknown" and item["candidate"]["channel_type"] in ("ecg", "eog", "emg", "stim", "misc")
                if channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage" and kind in ("unknown", "seeg") and not auxiliary:
                    context = {"channel_id": ids[row], "stream_id": stream["stream_id"], "segment_id": segment["segment_id"],
                               "recording_start_seconds": segment["recording_start_seconds"], "segment_start_sample": segment["start_sample"]}
                    trackers[row] = WaveformTracker(frequency, context, config)
        for first in range(start, stop, size):
            last = min(stop, first + size)
            data = np.asarray(read_window(stream["stream_id"], segment["segment_id"], ids, first, last,
                                          representation="physical"), dtype=np.float64)
            if data.shape != (len(ids), last - first):
                raise ValueError("Reader shape must be channel × requested samples")
            for row, channel in enumerate(channels):
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        metrics, flags, spectrum = compute_window_metrics(data[row], frequency, channel, config)
                except FloatingPointError as exc:
                    raise ValueError("Numerical overflow in channel {}; verify physical scaling".format(ids[row])) from exc
                item = mapped[row]
                kind = item["channel_type"]
                # C1/F3-style names occur in both scalp and depth recordings. A
                # name alone may suggest modality, but cannot settle this gate.
                if kind == "unknown" and item["candidate"]["channel_type"] in ("ecg", "eog", "emg", "stim", "misc"):
                    kind = item["candidate"]["channel_type"]
                eligible = channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage" and kind in ("unknown", "seeg") and last - first >= 2
                identity = "{}\0{}\0{}\0{}\0{}".format(stream["stream_id"], segment["segment_id"], ids[row], first, last)
                window = {"window_id": "sw-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                          "channel_id": ids[row], "stream_id": stream["stream_id"], "segment_id": segment["segment_id"],
                          "start_sample": first, "stop_sample": last, "duration_seconds": (last - first) / frequency,
                          "recording_start_seconds": segment["recording_start_seconds"] + (first - segment["start_sample"]) / frequency,
                          "metrics": metrics, "flags": flags, "score": score_window(flags, eligible), "score_eligible": eligible,
                          "status": "computed" if eligible else "not_applicable" if kind not in ("unknown", "seeg") else "insufficient_data"}
                windows.append(window)
                if row in trackers:
                    trackers[row].feed(data[row], first)
                _accumulate_spectrum(spectra, ids[row], stream["stream_id"], spectrum, window["duration_seconds"])
            relations.extend(compute_relations(data, mapped, stream["stream_id"], segment["segment_id"], first, last))
        next_plan = plan[plan_index + 1] if plan_index + 1 < len(plan) else None
        contiguous = next_plan is not None and next_plan[0]["stream_id"] == stream["stream_id"] and next_plan[1]["segment_id"] == segment["segment_id"] and next_plan[2] == stop
        if not contiguous:
            for tracker in trackers.values():
                tracker.flush("segment_end" if stop == segment["stop_sample"] else "range_end")
                waveform_events.extend(tracker.events)
            trackers = {}
            if len(waveform_events) > config["max_waveform_events"]:
                raise ValueError("Waveform event limit exceeded; select a shorter scan range")
    attach_waveform_events(windows, waveform_events)
    scanned = sum(w["duration_seconds"] for w in windows)
    channels = _summarize_channels(mapping, windows)
    for spectrum in spectra.values():
        weight = spectrum.pop("_weight")
        power = spectrum.pop("_power")
        spectrum["power_v2_hz"] = (power / weight).tolist() if weight else []
    return {"schema_version": 1, "profile": "seeg_basic_v1", "algorithm_version": SEEG_ALGORITHM_VERSION,
            "detection_version": WAVEFORM_VERSION, "evidence_only_flags": ["boundary_dwell", "waveform_anomaly"],
            "config": config, "channel_map": mapping, "integrity": _integrity(catalog, mapping, gaps),
            "windows": windows, "channels": channels, "spectra": list(spectra.values()), "relations": relations,
            "waveform_events": waveform_events,
            "initial_score": compute_quality_score(windows),
            "coverage": {"scope": "full_recording" if config["ranges"] is None else "selected_ranges", "ranges": ranges,
                         "scanned_channel_seconds": scanned, "total_channel_seconds": total,
                         "scanned_fraction": scanned / total if total else None,
                         "window_count": len(windows), "clock_policy": "within_source_segments_only",
                         "denominator": "readable_stream_channel_seconds",
                         "complete_input": not bool(catalog.get("errors")),
                         "unreadable_source_count": len(catalog.get("errors", []))}}


def _accumulate_spectrum(spectra, channel_id, stream_id, result, duration):
    item = spectra.setdefault(channel_id, {"channel_id": channel_id, "stream_id": stream_id,
                                         "status": "insufficient_data", "frequency_hz": [], "window_count": 0,
                                         "duration_seconds": 0.0, "_weight": 0.0, "_power": None,
                                         "method": "duration_weighted_mean_window_welch", "reason": result.get("reason")})
    if result["status"] != "computed":
        return
    power = np.asarray(result["power_v2_hz"])
    if item["_power"] is None:
        item["_power"] = np.zeros_like(power)
        item["frequency_hz"] = result["frequency_hz"]
    item["_power"] += power * duration
    item["_weight"] += duration
    item["duration_seconds"] += duration
    item["window_count"] += 1
    item.update(status="computed", reason=None)


def _summarize_channels(mapping, windows):
    grouped = defaultdict(list)
    for window in windows:
        grouped[window["channel_id"]].append(window)
    channels = []
    for item in mapping:
        subset = grouped[item["channel_id"]]
        duration = sum(w["duration_seconds"] for w in subset)
        flagged = sum(w["duration_seconds"] for w in subset if w["flags"])
        channels.append({"channel_id": item["channel_id"], "raw_name": item["raw_name"],
                         "stream_id": item["stream_id"], "scanned_seconds": duration,
                         "flagged_seconds": flagged, "flagged_fraction": flagged / duration if duration else None,
                         "line_noise_seconds": sum(w["duration_seconds"] for w in subset if "line_noise" in w["flags"]),
                         "flags": sorted({flag for w in subset for flag in w["flags"]}),
                         "score": compute_quality_score(subset)["total"], "status": "candidate_review_required"})
    return channels


def _integrity(catalog, mapping, gaps):
    unknown = [m["channel_id"] for m in mapping if m["unit"] != "V"]
    return {"status": "computed", "gaps": gaps, "unknown_voltage_unit_channel_ids": unknown,
            "mapping_pending_channel_ids": [m["channel_id"] for m in mapping if m["status"] != "confirmed"],
            "sources": deepcopy(catalog.get("sources", [])), "source_errors": deepcopy(catalog.get("errors", [])),
            "processing_history": deepcopy(catalog.get("processing", [])),
            "streams": [{"stream_id": s["stream_id"], "sampling_frequency_hz": s["sampling_frequency_hz"],
                         "segment_count": len(s["segments"]), "channel_count": len(s["channels"]),
                         "nyquist_hz": s["sampling_frequency_hz"] / 2,
                         "prefilter_by_channel": {c["channel_id"]: c.get("prefilter") for c in s["channels"]},
                         "processing_applied": deepcopy(s.get("processing_applied", [])),
                         "reference": s.get("reference"), "clock_id": s.get("clock_id")}
                        for s in catalog["streams"]]}
