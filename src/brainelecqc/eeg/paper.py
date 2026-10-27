"""Compose paper-defined measurements into a sample-addressed review queue."""
from dataclasses import asdict

from .bad_channels import BadChannelConfig, compute_bad_channels
from .events import EEGEventConfig, compute_event_qc
from .spectral import SpectralQCConfig, compute_spectral_qc
from .continuity import ContinuityConfig, compute_continuity_qc
from .muscle import EEGMuscleConfig, compute_muscle_qc


def validate_method_config(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {"comparison_groups", "bad_channels", "spectral", "events", "continuity", "muscle"}:
        raise ValueError("method_config accepts comparison_groups, bad_channels, spectral, events, continuity and muscle")
    result = {"comparison_groups": value.get("comparison_groups")}
    for key, cls in (("bad_channels", BadChannelConfig), ("spectral", SpectralQCConfig), ("events", EEGEventConfig),
                     ("continuity", ContinuityConfig), ("muscle", EEGMuscleConfig)):
        parameters = value.get(key, {})
        if not isinstance(parameters, dict):
            raise ValueError(f"{key} must be a parameter object")
        try:
            result[key] = asdict(cls(**parameters))
        except TypeError as exc:
            raise ValueError(f"Invalid {key}: {exc}") from exc
    return result


def compute_paper_methods(source, config):
    data, meta, channels = source["data_volts"], source["metadata"], source["channels"]
    args = (data, meta["sampling_frequency_hz"], [c["channel_id"] for c in channels], [c["raw_name"] for c in channels])
    origin = dict(start_sample=meta["start_sample"], start_seconds=meta["recording_start_seconds"])
    continuity = compute_continuity_qc(*args, **origin, channels=channels, config=config["continuity"])
    return {
        "continuity": continuity,
        "bad_channels": compute_bad_channels(*args, **origin, groups=config["comparison_groups"], config=config["bad_channels"]),
        "spectral": compute_spectral_qc(*args, **origin, config=config["spectral"]),
        "events": compute_event_qc(*args, **origin, references=source["references"],
            reference_channels=source["reference_channels"], config=config["events"]),
        "muscle": compute_muscle_qc(*args, **origin, references=source["references"],
            reference_channels=source["reference_channels"], config=config["muscle"],
            excluded_intervals=continuity["candidates"]),
    }


def attach_paper_candidates(decisions, methods):
    """Technical observations and paper candidates all request an explicit review."""
    queue = [] if "continuity" in methods else [x for x in decisions["review_queue"] if x["kind"] in {"file_endpoint_plateau", "flatline"}]
    for key in ("continuity", "muscle"):
        for item in methods.get(key, {}).get("candidates", []):
            queue.append({**item, "method_id": key, "threshold_basis": "project_screening_parameter",
                          "decision_source": "review_pending"})
    for item in methods["bad_channels"]["candidates"]:
        queue.append({**item, "method_id": "prep_nonspatial_review", "threshold_basis": "PREP_PyPREP",
                      "decision_source": "review_pending"})
    for row in methods["spectral"]["line_noise"]["observations"]:
        if row["flagged"]:
            queue.append({k: row[k] for k in ("channel_id", "channel_name", "start_sample", "stop_sample", "start_seconds", "stop_seconds")}
                | {"kind": "sinusoidal_line", "reason": "窄带正弦候选：核对谱峰、持续性及谐波", "method_id": "multitaper_F",
                   "threshold_basis": "global_bonferroni", "decision_source": "review_pending", "window_index": row["window_index"],
                   "evidence": {k: row[k] for k in ("nominal_hz", "peak_frequency_hz", "f_statistic", "p_corrected", "amplitude_uv")}})
    queue.sort(key=lambda x: (x["start_sample"], x["channel_id"], x["kind"]))
    for index, item in enumerate(queue, 1):
        item["candidate_id"] = f"R{index:04d}"
    decisions["review_queue"] = queue
    decisions["summary"]["review_count"] = len(queue)
    decisions["policy_version"] = "eeg-paper-review-2.0" if "continuity" in methods else "eeg-paper-review-1.0"
    decisions["measurement_methods"] = ["PREP", "multitaper_F", "Welch_FOOOF", "MNE_event_locked"]
    decisions["measurement_methods"] += [key for key in ("continuity", "muscle") if key in methods]
    return decisions
