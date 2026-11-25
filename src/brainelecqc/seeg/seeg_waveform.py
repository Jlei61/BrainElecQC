"""Bounded streaming summaries; QC-window boundaries never reset morphology.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import math

import numpy as np

from .seeg_waveform_events import WAVEFORM_DEFAULTS, WAVEFORM_VERSION, detect_events


def validate_waveform_config(config):
    if not .002 <= config["waveform_bin_seconds"] <= .1:
        raise ValueError("waveform_bin_seconds must be within [.002,.1]")
    if not .05 <= config["waveform_recovery_ratio"] <= .5:
        raise ValueError("waveform_recovery_ratio must be within [.05,.5]")
    for key in ("waveform_baseline_seconds", "waveform_min_event_seconds", "waveform_drift_seconds",
                "waveform_plateau_seconds", "waveform_recovery_seconds"):
        if not 3 * config["waveform_bin_seconds"] <= config[key] <= 10:
            raise ValueError("{} must cover >=3 morphology bins and be <=10 seconds".format(key))
    for key in ("max_waveform_bins", "max_waveform_events"):
        if int(config[key]) != config[key] or not 1 <= config[key] <= WAVEFORM_DEFAULTS[key]:
            raise ValueError("{} must be within the supported waveform resource limit".format(key))


def preflight_waveform(plan, config):
    bins = 0
    for stream, _, start, stop, _ in plan:
        size = max(1, round(config["waveform_bin_seconds"] * stream["sampling_frequency_hz"]))
        bins += math.ceil((stop - start) / size) * len(stream["channels"])
    if bins > config["max_waveform_bins"]:
        raise ValueError("Waveform summary limit exceeded; select explicit shorter scan ranges")


class WaveformTracker:
    """Retain compact bin features and <one bin of samples per channel."""
    def __init__(self, frequency, context, config):
        self.frequency, self.context, self.config = frequency, context, config
        self.size = max(1, round(frequency * config["waveform_bin_seconds"]))
        self.buffer = np.empty(0)
        self.buffer_start = None
        self.previous = None
        self.expected = None
        self.parts = []
        self.events = []

    def feed(self, values, start):
        if self.expected is not None and start != self.expected:
            self.flush("range_end")
        self.expected = start + len(values)
        finite = np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
        runs = list(zip(edges[::2], edges[1::2]))
        cursor = 0
        for first, last in runs:
            if first > cursor:
                self.flush("nonfinite_gap")
            self._append(values[first:last], start + first)
            if last < len(values):
                self.flush("nonfinite_gap")
            cursor = last
        if not runs:
            self.flush("nonfinite_gap")

    def _append(self, values, start):
        if not len(self.buffer):
            self.buffer_start = start
        self.buffer = np.concatenate([self.buffer, values])
        count = len(self.buffer) // self.size
        if count:
            used = count * self.size
            self._summarize(self.buffer[:used].reshape(count, self.size), self.buffer_start)
            self.buffer = self.buffer[used:].copy()
            self.buffer_start += used

    def _summarize(self, rows, start):
        length = rows.shape[1]
        centers = np.median(rows, axis=1)
        mad = np.median(np.abs(rows - centers[:, None]), axis=1)
        preceding = np.r_[rows[0, 0] if self.previous is None else self.previous, rows[:-1, -1]]
        differences = np.diff(rows, axis=1, prepend=preceding[:, None])
        positions = np.argmax(np.abs(differences), axis=1)
        starts = start + np.arange(len(rows)) * length
        changed = np.diff(rows, axis=1) != 0
        prefix = np.where(changed.any(axis=1), np.argmax(changed, axis=1) + 1, length) if length > 1 else np.ones(len(rows))
        suffix = np.where(changed.any(axis=1), np.argmax(changed[:, ::-1], axis=1) + 1, length) if length > 1 else np.ones(len(rows))
        result = np.column_stack([starts, starts + length, centers, mad,
                                  differences[np.arange(len(rows)), positions], starts + positions,
                                  rows[:, 0], rows[:, -1], prefix, suffix])
        self.parts.append(result)
        self.previous = rows[-1, -1]

    def flush(self, termination):
        if len(self.buffer):
            self._summarize(self.buffer.reshape(1, -1), self.buffer_start)
        if self.parts:
            summary = np.concatenate(self.parts)
            self.events.extend(detect_events(summary, self.frequency, self.context, self.config, termination))
        self.buffer = np.empty(0)
        self.buffer_start, self.previous = None, None
        self.parts.clear()


def attach_waveform_events(windows, events):
    grouped = {}
    for event in events:
        key = (event["stream_id"], event["segment_id"], event["channel_id"])
        grouped.setdefault(key, []).append(event)
    for window in windows:
        key = (window["stream_id"], window["segment_id"], window["channel_id"])
        selected = [event["event_id"] for event in grouped.get(key, [])
                    if event["start_sample"] < window["stop_sample"] and event["stop_sample"] > window["start_sample"]]
        window["waveform_event_ids"] = selected
        if selected:
            window["flags"].append("waveform_anomaly")
