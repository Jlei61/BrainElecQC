"""Traceable screening observations awaiting human review."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .masks import apply_eeg_decisions, true_runs, validate_binding


@dataclass(frozen=True)
class EEGPolicy:
    clipping_min_seconds: float = 0.1
    padding_seconds: float = 0.25
    flat_min_seconds: float = 1.0
    flat_step_uv: float = 0.01
    flat_range_uv: float = 0.2
    review_rms_uv: float = 100.0
    review_relative_rms: float = 3.0
    review_large_amplitude_fraction: float = 0.01
    review_line_db: float = 10.0
    review_correlation: float = 0.6
    review_high_frequency_fraction: float = 0.4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.clipping_min_seconds, self.flat_min_seconds) <= 0:
            raise ValueError("Minimum observation duration must be positive")
        if any(not 0 <= value <= 1 for value in (self.review_large_amplitude_fraction,
                                                self.review_correlation, self.review_high_frequency_fraction)):
            raise ValueError("Review fractions and correlation must lie in [0, 1]")


def _interval(features, cid, a, b, kind, reason, evidence, source):
    fs, origin = features["sampling_rate_hz"], features["start_sample"]
    return {"channel_id": cid, "start_sample": origin+a, "stop_sample": origin+b,
            "start_seconds": features["start_seconds"]+a/fs,
            "stop_seconds": features["start_seconds"]+b/fs,
            "kind": kind, "reason": reason, "evidence": evidence, "decision_source": source}


def _clipping(data, features, limits, cfg):
    fs, n = features["sampling_rate_hz"], data.shape[1]
    minimum, padding = math.ceil(cfg.clipping_min_seconds*fs), math.ceil(cfg.padding_seconds*fs)
    result, status = [], {}
    if set(limits) - set(features["channel_ids"]):
        raise ValueError("Calibration bounds must identify selected EEG channels")
    for cid, row in zip(features["channel_ids"], data):
        bound = limits.get(cid)
        if bound is None:
            status[cid] = "calibration_bounds_needed"
            continue
        low, high, tolerance = (bound[k] for k in ("lower_volts", "upper_volts", "tolerance_volts"))
        if (not all(np.isfinite(v) for v in (low, high, tolerance)) or not low < high
                or not 0 <= tolerance < (high-low)/2 or not bound.get("source")):
            raise ValueError("Calibration bounds require finite increasing limits, tolerance and source")
        status[cid] = "checked_declared_bounds"
        for side, value in (("lower", low), ("upper", high)):
            for a, b in true_runs(np.abs(row-value) <= tolerance):
                if b-a < minimum:
                    continue
                result.append(_interval(features, cid, a, b,
                    "file_endpoint_plateau", "文件标定端点平台：核对采集与导出原因", {
                        "boundary": side, "boundary_uv": value*1e6, "bounds_source": bound["source"],
                        "tolerance_volts": tolerance, "core_start_sample": features["start_sample"]+a,
                        "core_stop_sample": features["start_sample"]+b,
                        "duration_seconds": (b-a)/fs,
                        "context_start_sample": features["start_sample"]+max(0, a-padding),
                        "context_stop_sample": features["start_sample"]+min(n, b+padding),
                        "context_padding_seconds": cfg.padding_seconds}, "review_pending"))
    return result, status


def _flat(data, features, cfg):
    fs, n = features["sampling_rate_hz"], data.shape[1]
    minimum, padding = math.ceil(cfg.flat_min_seconds*fs), math.ceil(cfg.padding_seconds*fs)
    result = []
    for cid, row in zip(features["channel_ids"], data):
        for a, last in true_runs(np.abs(np.diff(row)) <= cfg.flat_step_uv*1e-6):
            b = last+1
            if b-a < minimum or np.ptp(row[a:b]) > cfg.flat_range_uv*1e-6:
                continue
            result.append(_interval(features, cid, a, b,
                "flatline", "低变化平台：核对通道类型、分辨率及波形", {"core_start_sample": features["start_sample"]+a,
                    "core_stop_sample": features["start_sample"]+b, "duration_seconds": (b-a)/fs,
                    "range_uv": float(np.ptp(row[a:b])*1e6),
                    "context_start_sample": features["start_sample"]+max(0, a-padding),
                    "context_stop_sample": features["start_sample"]+min(n, b+padding)}, "review_pending"))
    return result


def _review(features, cfg):
    result, matrices = [], features["window_features"]
    def val(key, row, col):
        return matrices[key][row][col]
    for i, cid in enumerate(features["channel_ids"]):
        for j, window in enumerate(features["windows"]):
            reasons = []
            rms, relative = val("rms_uv", i, j), val("relative_rms", i, j)
            large = val("large_amplitude_fraction", i, j)
            if ((rms is not None and relative is not None and rms >= cfg.review_rms_uv
                 and relative >= cfg.review_relative_rms) or (large is not None and large >= cfg.review_large_amplitude_fraction)):
                reasons.append(("amplitude", "幅值异常：复核波形、漂移及生理活动", {"rms_uv": rms,
                                "relative_rms": relative, "large_amplitude_fraction": large}))
            for hz in (50, 60):
                value = val(f"line_{hz}_db", i, j)
                if value is not None and value >= cfg.review_line_db:
                    reasons.append((f"line_{hz}", f"{hz} Hz 邻域峰值比：核对窄峰位置与持续性", {"prominence_db": value}))
            for key, kind, label in (("ecg_correlation", "ecg", "心电"), ("eog_correlation", "eog", "眼电"),
                                     ("emg_envelope_correlation", "emg", "肌电包络")):
                value = val(key, i, j)
                if value is not None and abs(value) >= cfg.review_correlation:
                    reasons.append((kind, f"与{label}参考相关：核对事件锁时波形", {key: value}))
            value = val("high_frequency_fraction", i, j)
            if value is not None and value >= cfg.review_high_frequency_fraction:
                reasons.append(("high_frequency", "高频占比升高：复核肌电与脑电活动", {"high_frequency_fraction": value}))
            for kind, reason, evidence in reasons:
                item = _interval(features, cid, window["start_sample"]-features["start_sample"],
                                 window["stop_sample"]-features["start_sample"], kind, reason, evidence, "review_pending")
                item["window_index"] = j
                result.append(item)
    return result


def propose_eeg_decisions(data_volts, features, *, calibration_bounds=None, config=None):
    """Propose review candidates. File calibration and screening cutoffs retain samples."""
    data = validate_binding(data_volts, features)
    cfg = EEGPolicy(**config) if isinstance(config, dict) else config or EEGPolicy()
    if not isinstance(cfg, EEGPolicy):
        raise ValueError("config must be EEGPolicy or a parameter dictionary")
    observations, rail_status = _clipping(data, features, calibration_bounds or {}, cfg)
    for item in _flat(data, features, cfg):
        a, b = item["evidence"]["core_start_sample"], item["evidence"]["core_stop_sample"]
        overlaps = [x for x in observations if x["channel_id"] == item["channel_id"]
                    and x["evidence"]["core_start_sample"] < b and a < x["evidence"]["core_stop_sample"]]
        if overlaps:
            primary = overlaps[0]
            # Preserve both observations while retaining the endpoint's exact boundary.
            primary["evidence"]["flat_support"] = item["evidence"]
        else:
            observations.append(item)
    queue = sorted(observations+_review(features, cfg),
                   key=lambda x: (x["start_sample"], x["channel_id"], x["kind"]))
    for index, item in enumerate(queue, 1):
        kind = item["kind"]
        item.update(candidate_id=f"R{index:04d}", threshold_basis="project_screening_parameter",
                    method_id=("endpoint_plateau" if kind == "file_endpoint_plateau" else
                               "line_neighbour_ratio" if kind.startswith("line_") else
                               "reference_correlation" if kind in {"ecg", "eog", "emg"} else
                               "amplitude_screening"))
    result = {key: features[key] for key in ("channel_ids", "channel_names", "sampling_rate_hz",
              "start_sample", "stop_sample", "start_seconds", "data_sha256")}
    result.update(schema_version=1, policy_version="eeg-review-2.0", policy_config=asdict(cfg),
                  decision_mode="review_only", calibration_status=rail_status,
                  automatic_exclusions=[], review_queue=queue, review_status="pending")
    _, keep = apply_eeg_decisions(data, result, channel_ids=features["channel_ids"])
    fs = features["sampling_rate_hz"]
    result["channel_summary"] = [{"channel_id": cid, "excluded_seconds": float((~row).sum()/fs),
        "retained_seconds": float(row.sum()/fs), "excluded_samples": int((~row).sum()),
        "status": "excluded" if not row.any() else "partly_excluded" if not row.all() else "retained"}
        for cid, row in zip(features["channel_ids"], keep)]
    result["summary"] = {"excluded_channel_seconds": float((~keep).sum()/fs),
        "retained_channel_seconds": float(keep.sum()/fs), "excluded_fraction": float((~keep).mean()),
        "excluded_channel_samples": int((~keep).sum()), "retained_channel_samples": int(keep.sum()),
        "fully_excluded_channel_ids": [cid for cid, row in zip(features["channel_ids"], keep) if not row.any()],
        "review_count": len(result["review_queue"]), "automatic_interval_count": 0}
    return result
