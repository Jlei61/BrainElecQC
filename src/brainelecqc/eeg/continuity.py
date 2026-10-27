"""Locate raw constant-value runs and retain their distinct acquisition evidence.

Inputs are calibrated volts in one continuous clock segment. Classification uses
original samples and optional file calibration; filtering and group statistics
are separate analyses. Every candidate remains an explicit review observation.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class ContinuityConfig:
    """Project screening choices, expressed in physical units and seconds."""

    min_constant_seconds: float = 0.1
    near_zero_volts: float = 0.1e-6
    endpoint_tolerance_lsb: float = 0.25

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.endpoint_tolerance_lsb >= 0.5:
            raise ValueError("endpoint_tolerance_lsb must be below half a digital step")


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _range(a, b, sfreq, start_sample, start_seconds):
    return {"start_sample": int(start_sample + a), "stop_sample": int(start_sample + b),
            "start_seconds": float(start_seconds + a / sfreq),
            "stop_seconds": float(start_seconds + b / sfreq)}


def _number(metadata, key):
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"Channel calibration {key} must be finite when supplied")
    return float(value)


def _calibration(metadata, cfg):
    if metadata is None:
        return {"status": "unavailable", "source": "not_supplied", "quantization_step_volts": None,
                "lower_volts": None, "upper_volts": None, "endpoint_tolerance_volts": None,
                "near_zero_tolerance_volts": cfg.near_zero_volts}
    if metadata.get("unit") != "V":
        raise ValueError("Channel metadata must declare calibrated input unit V")
    scale = _number(metadata, "scale_to_si")
    gain = _number(metadata, "digital_to_physical_scale")
    low, high = _number(metadata, "physical_min"), _number(metadata, "physical_max")
    if scale is not None and scale <= 0:
        raise ValueError("scale_to_si must be positive")
    if gain == 0 or (low is not None and high is not None and low == high):
        raise ValueError("Channel calibration must have a nonzero physical span")
    step = abs(scale * gain) if scale is not None and gain is not None else None
    lower, upper = sorted([scale * low, scale * high]) if None not in (scale, low, high) else (None, None)
    if step is not None and lower is not None and step > upper - lower:
        raise ValueError("Quantization step must fit within the physical calibration span")
    # Half one step includes the nearest representable zero in an asymmetric
    # signed-integer calibration. It never controls constant-run segmentation.
    zero_tol = max(cfg.near_zero_volts, step * 0.5 * (1 + 1e-12) if step else 0.0)
    endpoint_tol = max(step * cfg.endpoint_tolerance_lsb,
                       np.finfo(float).eps * max(abs(lower), abs(upper)) * 8) \
        if step is not None and lower is not None else None
    return {"status": "available" if endpoint_tol is not None else "partial",
            "source": "file_declared_calibration", "quantization_step_volts": step,
            "lower_volts": lower, "upper_volts": upper, "endpoint_tolerance_volts": endpoint_tol,
            "near_zero_tolerance_volts": zero_tol}


def _inputs(data_volts, sfreq, channel_ids, channel_names, start_sample, start_seconds, channels):
    values = np.asarray(data_volts)
    if values.dtype.kind not in "iuf" or values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("data_volts must have a nonempty real channel x sample shape")
    data = np.asarray(values, dtype=float)
    if isinstance(channel_ids, (str, bytes)) or isinstance(channel_names, (str, bytes)):
        raise ValueError("Channel IDs and names must be sequences")
    ids = list(channel_ids)
    names = ids.copy() if channel_names is None else list(channel_names)
    if len(ids) != data.shape[0] or len(names) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct channel IDs and names must match the data rows")
    if any(not isinstance(x, str) or not x for x in ids + names):
        raise ValueError("Channel IDs and names must be nonempty strings")
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, Real) or not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if channels is None:
        metadata = [None] * len(ids)
    else:
        if not isinstance(channels, (list, tuple)) or any(not isinstance(c, dict) for c in channels):
            raise ValueError("channels must be a sequence of source metadata dictionaries")
        by_id = {c.get("channel_id"): c for c in channels}
        if len(by_id) != len(channels) or set(by_id) != set(ids):
            raise ValueError("Calibration metadata must identify exactly the selected channel IDs")
        metadata = [by_id[c] for c in ids]
        if any(c.get("raw_name", name) != name for c, name in zip(metadata, names)):
            raise ValueError("Calibration raw_name must match its selected channel name")
    return data, ids, names, metadata


def compute_continuity_qc(data_volts, sfreq, channel_ids, channel_names=None, *,
                          start_sample=0, start_seconds=0.0, channels=None, config=None):
    """Return constant-run and missing-sample observations on the original clock.

    A constant run contains exactly equal finite raw samples for at least
    ``ceil(min_constant_seconds * sfreq)`` samples (minimum two). A changing
    low-amplitude signal remains variable regardless of its absolute amplitude.
    Endpoint matching uses file calibration before the near-zero classification.
    ``channels`` is the selected source metadata list returned by read_eeg_input;
    its order is resolved by stable channel ID. All output intervals are half-open.
    """
    if config is not None and not isinstance(config, (ContinuityConfig, dict)):
        raise ValueError("config must be ContinuityConfig or a parameter dictionary")
    cfg = config if isinstance(config, ContinuityConfig) else ContinuityConfig(**(config or {}))
    data, ids, names, metadata = _inputs(data_volts, sfreq, channel_ids, channel_names,
                                       start_sample, start_seconds, channels)
    minimum = max(2, int(np.ceil(cfg.min_constant_seconds * sfreq)))
    n_samples = data.shape[1]
    candidates, measurements = [], []
    kinds = ("near_zero_flatline", "calibration_endpoint_plateau", "constant_value", "nonfinite_input")
    labels = {"near_zero_flatline": "近零恒值／掉线候选", "calibration_endpoint_plateau": "文件量程端点平台",
              "constant_value": "其他恒值片段", "nonfinite_input": "非有限样本"}
    for ci, row in enumerate(data):
        calibration = _calibration(metadata[ci], cfg)
        finite = np.isfinite(row)
        # Exact equality separates different ADC codes, however small their
        # physical spacing, and avoids turning smooth low-amplitude drift flat.
        equal = finite[:-1] & finite[1:] & (row[:-1] == row[1:])
        spans = [(a, last + 1, False) for a, last in _runs(equal) if last + 1 - a >= minimum]
        spans.extend((a, b, True) for a, b in _runs(~finite))
        counts = {kind: 0 for kind in kinds}
        durations = {kind: 0 for kind in kinds}
        constant_samples = 0
        for a, b, missing in sorted(spans):
            value = None if missing else float(row[a])
            endpoint = None
            if missing:
                kind = "nonfinite_input"
            else:
                constant_samples += b - a
                if calibration["endpoint_tolerance_volts"] is not None:
                    for side in ("lower", "upper"):
                        if abs(value - calibration[f"{side}_volts"]) <= calibration["endpoint_tolerance_volts"]:
                            endpoint = side
                            break
                kind = "calibration_endpoint_plateau" if endpoint else (
                    "near_zero_flatline" if abs(value) <= calibration["near_zero_tolerance_volts"] else "constant_value")
            counts[kind] += 1
            durations[kind] += int(b - a)
            before = float(row[a - 1]) if a > 0 and finite[a - 1] else None
            after = float(row[b]) if b < n_samples and finite[b] else None
            evidence = {"value_volts": value, "value_uv": None if value is None else value * 1e6,
                        "peak_to_peak_volts": None if missing else 0.0,
                        "before_volts": before, "after_volts": after,
                        "endpoint": endpoint, "calibration": dict(calibration),
                        "constant_rule": "exact_equal_finite_raw_samples",
                        "touches_input_start": bool(a == 0), "touches_input_stop": bool(b == n_samples)}
            candidates.append({"candidate_id": f"continuity:{ids[ci]}:{start_sample+a}:{start_sample+b}:{kind}",
                "channel_index": ci, "channel_id": ids[ci], "channel_name": names[ci], "kind": kind,
                "label": labels[kind], "observation": "nonfinite_raw_samples" if missing else "constant_raw_samples",
                "reason": labels[kind], "action": "review", "review_status": "pending_review",
                "sample_count": int(b - a), "duration_seconds": float((b - a) / sfreq),
                "calibration_status": calibration["status"],
                **_range(a, b, sfreq, start_sample, start_seconds), "evidence": evidence})
        finite_values = row[finite]
        measurements.append({"channel_index": ci, "channel_id": ids[ci], "channel_name": names[ci],
            "calibration": calibration, "calibration_status": calibration["status"],
            "finite_sample_count": int(finite.sum()), "nonfinite_sample_count": int((~finite).sum()),
            "constant_sample_count": int(constant_samples),
            "finite_outside_constant_sample_count": int(finite.sum() - constant_samples),
            "changing_adjacent_pair_count": int(np.count_nonzero(finite[:-1] & finite[1:] & ~equal)),
            "peak_to_peak_volts": float(np.ptp(finite_values)) if finite_values.size else None,
            "raw_std_volts": float(np.std(finite_values)) if finite_values.size else None,
            "candidate_counts": counts, "candidate_sample_counts": durations,
            "candidate_seconds": {key: float(count / sfreq) for key, count in durations.items()},
            "status": "review_candidates" if sum(counts.values()) else "no_constant_run_at_threshold"})
    return {"schema_version": "1.0", "method": {"id": "raw_continuity_review", "version": "1.0.0",
        "scope": "raw constant-value runs with source calibration classification",
        "sources": [{"url": "https://github.com/sccn/clean_rawdata/blob/master/clean_flatlines.m",
                     "role": "successive-sample flatline measurement; project uses exact equality and local intervals"},
                    {"url": "https://www.edfplus.info/specs/guidelines.html",
                     "role": "file digital-to-physical calibration and quantization step"}],
        "threshold_source": "project screening parameters; 0.1 s locates short interruptions for review",
        "classification_order": ["nonfinite_input", "calibration_endpoint_plateau", "near_zero_flatline", "constant_value"],
        "cause_status": "requires acquisition metadata and human review"},
        "parameters": {**asdict(cfg), "min_constant_samples": minimum,
                       "effective_min_constant_seconds": float(minimum / sfreq),
                       "constant_tolerance_volts": 0.0},
        "data_sha256": sha256(np.asarray(data, dtype="<f8").tobytes()).hexdigest(),
        "channel_ids": ids, "channel_names": names, "sample_rate": float(sfreq), "sampling_rate_hz": float(sfreq),
        "unit": "V", "axes": ["channel", "sample"], "interval_convention": "half_open",
        "input_range": _range(0, n_samples, sfreq, start_sample, start_seconds),
        "preprocessing": {"filtering": "none", "input_reference_retained": True},
        "measurements": measurements, "candidates": candidates, "automatic_exclusions": []}
