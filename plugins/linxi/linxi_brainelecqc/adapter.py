"""Validate task parameters and bind file-based QC to an isolated Linxi task."""
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from brainelecqc.config import storage_context
from brainelecqc.contracts import descriptor
from brainelecqc.db import init_db
from brainelecqc.eeg.export import export_eeg_qc
from brainelecqc.eeg.pipeline import run_eeg_qc, validate_qc_config
from brainelecqc.recordings import import_recording


def _json_copy(value):
    """Detach constructor parameters from later caller mutations."""
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Configuration object keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)
            or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be a finite number in its allowed range")


def normalize_options(qc_config, figures, title, display_start_seconds,
                      display_duration_seconds, recording_info):
    config = _json_copy({} if qc_config is None else qc_config)
    selection, _, _, methods = validate_qc_config(config)
    for key in ("stream_id", "segment_id"):
        if selection.get(key) is not None and (not isinstance(selection[key], str) or not selection[key]):
            raise ValueError(f"{key} must be a nonempty string")
    if selection.get("start_seconds") is not None:
        _number(selection["start_seconds"], "start_seconds")
    if selection.get("duration_seconds") is not None:
        _number(selection["duration_seconds"], "duration_seconds", positive=True)
    indices = selection.get("eeg_indices")
    if indices is not None and (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("eeg_indices must contain distinct nonnegative integers")
    refs = selection.get("reference_indices")
    if refs is not None and (not isinstance(refs, dict) or set(refs) - {"ecg", "eog", "emg"}
            or any(type(index) is not int or index < 0 or index in (indices or []) for index in refs.values())
            or len(set(refs.values())) != len(refs)):
        raise ValueError("reference_indices must map auxiliary kinds to distinct non-EEG indices")
    groups = methods["comparison_groups"]
    if groups is not None and (not isinstance(groups, list)
            or any(not isinstance(group, str) or not group for group in groups)
            or (indices is not None and len(groups) != len(indices))):
        raise ValueError("comparison_groups must supply one nonempty group name per selected EEG channel")
    if type(figures) is not bool or not isinstance(title, str) or not title.strip():
        raise ValueError("figures must be boolean and title must be a nonempty string")
    if display_start_seconds is not None:
        _number(display_start_seconds, "display_start_seconds", nonnegative=True)
    _number(display_duration_seconds, "display_duration_seconds", positive=True)
    info = _json_copy({} if recording_info is None else recording_info)
    if not isinstance(info, dict) or set(info) - {"label", "participant_id", "session_id", "metadata"}:
        raise ValueError("recording_info accepts label, participant_id, session_id and metadata")
    descriptor({"label": "EEG recording", **info, "modality": "eeg"})
    return {"qc_config": config, "figures": figures, "title": title,
            "display_start_seconds": display_start_seconds,
            "display_duration_seconds": display_duration_seconds, "recording_info": info}


def _paths(input_path, work_dir):
    if not input_path:
        raise ValueError("Set context.run_state.input_path to an EEG file or companion-file directory")
    source = Path(input_path).expanduser().absolute()
    if source.is_symlink():
        raise ValueError("Choose an original input path; symlink input is unsupported")
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise ValueError("input_path must identify an existing file or directory")
    source = source.resolve()
    work = Path(work_dir).expanduser().resolve()
    base = work / "brainelecqc"
    if any((parent / ".git").exists() for parent in (work, *work.parents)):
        raise ValueError("Choose a Linxi work_dir outside the Git repository")
    for owned in (base, base / "store", base / "runs"):
        if owned.is_symlink() or (owned.exists() and not owned.is_dir()):
            raise ValueError(f"QC task directory must be an ordinary directory: {owned.name}")
    if (source.is_dir() and work.is_relative_to(source)) or source.is_relative_to(base):
        raise ValueError("Keep the input and QC work directories separate to avoid recursive input copies")
    return source, base


def process_file(input_path, work_dir, options, *, progress):
    source, base = _paths(input_path, work_dir)
    info = descriptor({"label": source.name, **options["recording_info"], "modality": "eeg"})
    identity = json.dumps({"source": str(source), "descriptor": info}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
    info["import_key"] = "linxi-eeg:" + hashlib.sha256(identity.encode()).hexdigest()
    selection, features, policy, methods = validate_qc_config(options["qc_config"])
    with storage_context(base / "store"):
        init_db()
        record, reused = import_recording(info, paths=[source])
        progress("analyze")
        result = run_eeg_qc(record["input_artifact_id"], feature_config=features,
                            policy_config=policy, method_config=methods, **selection)
        result["manifest"]["host_integration"] = {
            "processor": "BrainElecQCProcessor", "plugin_version": version("linxi-brainelecqc"),
            "linxi_version": version("linxi"), "parameters": options}
        progress("export")
        output = base / "runs" / result["manifest"]["run_id"]
        exported = export_eeg_qc(result, output, **{key: options[key] for key in
            ("figures", "title", "display_start_seconds", "display_duration_seconds")})
    score = json.loads((output / "quality_score.json").read_text(encoding="utf-8"))
    return {"schema_version": 1, "recording_id": record["recording_id"],
        "input_artifact_id": record["input_artifact_id"], "run_id": result["manifest"]["run_id"],
        "data_sha256": result["manifest"]["data_sha256"], "bundle_dir": str(output),
        "manifest_path": str(output / "manifest.json"), "report_path": exported["report"],
        "score_path": str(output / "quality_score.json"),
        "score_report_path": exported["score_report"],
        "score_100": score["score_100"], "assessment_coverage": score["assessment_coverage"],
        "auto_total": score["auto_total"], "assessed_max": score["assessed_max"],
        "pending_max": score["pending_max"], "review_status": "pending", "review_id": None,
        "figures": exported["figures"], "source_reused": reused}
