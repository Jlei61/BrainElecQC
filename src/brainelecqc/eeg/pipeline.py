"""Plain Python EEG QC entry point, returning arrays and JSON-compatible evidence."""
from importlib.metadata import version
import uuid

import numpy as np

from .features import compute_eeg_features
from .config import EEGConfig
from .input import read_eeg_input
from .masks import apply_eeg_decisions, retained_intervals
from .policy import EEGPolicy, propose_eeg_decisions
from .paper import validate_method_config, compute_paper_methods, attach_paper_candidates


def validate_qc_config(config):
    if not isinstance(config, dict) or set(config)-{"selection", "feature_config", "policy_config", "method_config"}:
        raise ValueError("QC config accepts selection, feature_config, policy_config and method_config")
    selection = config.get("selection", {})
    allowed = {"stream_id", "segment_id", "start_seconds", "duration_seconds", "eeg_indices", "reference_indices"}
    if not isinstance(selection, dict) or set(selection)-allowed:
        raise ValueError("selection must be an object with stream_id, segment_id, start_seconds, duration_seconds, eeg_indices or reference_indices")
    parameters = []
    for key, cls in (("feature_config", EEGConfig), ("policy_config", EEGPolicy)):
        value = config.get(key)
        if value is not None and not isinstance(value, (dict, cls)):
            raise ValueError(f"{key} must be a parameter object")
        try:
            parameters.append(cls(**value) if isinstance(value, dict) else value or cls())
        except TypeError as exc:
            raise ValueError(f"Invalid {key}: {exc}") from exc
    return selection.copy(), *parameters, validate_method_config(config.get("method_config"))


def run_eeg_qc(artifact_id, *, feature_config=None, policy_config=None, method_config=None, **selection):
    selection, feature_config, policy_config, method_config = validate_qc_config({"selection": selection,
        "feature_config": feature_config, "policy_config": policy_config, "method_config": method_config})
    source = read_eeg_input(artifact_id, **selection)
    data, metadata, channels = source["data_volts"], source["metadata"], source["channels"]
    ids = [c["channel_id"] for c in channels]
    features = compute_eeg_features(data, metadata["sampling_frequency_hz"], ids,
        [c["raw_name"] for c in channels], start_sample=metadata["start_sample"],
        start_seconds=metadata["recording_start_seconds"], references=source["references"], config=feature_config)
    features["reference_channels"] = source["reference_channels"]
    decisions = propose_eeg_decisions(data, features, calibration_bounds=source["calibration_bounds"], config=policy_config)
    methods = compute_paper_methods(source, method_config)
    if source.get("header_qc") is not None:
        methods["header"] = source["header_qc"]
    attach_paper_candidates(decisions, methods)
    masked, keep = apply_eeg_decisions(data, decisions, channel_ids=ids)
    for row, original, retained in zip(decisions["channel_summary"], data, keep):
        row["before_ac_rms_uv"] = float(np.std(original)*1e6)
        row["retained_ac_rms_uv"] = float(np.std(original[retained])*1e6) if retained.any() else None
    provenance = {key: metadata[key] for key in ("recording_id", "artifact_id", "content_sha256", "source_sha256",
        "stream_id", "segment_id", "clock_id", "clock_alignment", "reader_version", "sample_index_origin")}
    provenance.update(schema_version=1, run_id=str(uuid.uuid4()), modality="eeg", unit="V", axes=["channel", "sample"],
        channel_ids=ids, start_sample=features["start_sample"], stop_sample=features["stop_sample"],
        start_seconds=features["start_seconds"], sampling_rate_hz=features["sampling_rate_hz"],
        data_sha256=features["data_sha256"], interval_convention="half_open", selection=source["selection"],
        review_status="pending", decision_source="review_pending", decision_mode=decisions["decision_mode"],
        mask_semantics="True retains the original sample; False excludes it",
        method_config=method_config,
        acquisition_filter_headers=[{k: c.get(k) for k in ("channel_id", "raw_name", "prefilter")} for c in channels],
        software_versions={name: version(name) for name in ("brainelecqc", "numpy", "scipy", "mne", "fooof")},
        feature_version=features["feature_version"], policy_version=decisions["policy_version"])
    return {"manifest": provenance, "features": features, "method_results": methods, "decisions": decisions, "channels": channels,
            "raw_data_volts": data, "masked_data_volts": masked, "keep_mask": keep,
            "references": source["references"], "retained_intervals": list(retained_intervals(keep, features)),
            "verification": {"retained_values_exact": bool(np.array_equal(data[keep], masked[keep])),
                "excluded_values_nan": bool(np.all(np.isnan(masked[~keep]))), "shape": list(data.shape),
                "original_clock_preserved": True}}
