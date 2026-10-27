"""Run three existing EEG examples through the installed official Linxi host.

All input paths are supplied by the caller; all artifacts go outside Git.
An optional previous bundle root verifies unchanged integer scoring and samples.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
import logging
from pathlib import Path

# Linxi configures its numerical runtime before the scientific libraries.
from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

import mne
import numpy as np
import yaml

from brainelecqc.eeg.bad_channels import compare_with_pyprep
from brainelecqc.eeg.events import compare_events_to_mne
from brainelecqc.eeg.spectral import compare_spectral_reference


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    stat = path.stat()
    return {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def event_checks(comparison):
    checks = {}
    for kind, block in comparison["by_kind"].items():
        if block["status"] == "compared":
            checks[f"{kind}_indices"] = block["event_indices_equal"]
            for key, value in block.items():
                if "max_abs_error" in key and value is not None:
                    checks[f"{kind}_{key}"] = value <= (1e-14 if key.endswith("seconds") else 1e-18)
    return checks


def run_case(case, output, baseline):
    source, name, selection, groups = case
    before = fingerprint(source)
    directory = output / name
    directory.mkdir()
    params = {"figures": True, "title": f"{name} EEG QC", "display_start_seconds": selection["start_seconds"],
              "qc_config": {"selection": selection, "method_config": {"comparison_groups": groups}},
              "recording_info": {"label": name, "metadata": {
                  "comparison_group_basis": "original M1/M2 labels" if name.startswith("HMC")
                      else "prespecified same-recording group; acquisition reference requires review"}}}
    pipeline_path = directory / "pipeline.yml"
    pipeline_path.write_text(yaml.safe_dump({"name": f"eeg-qc-{name}", "linxi_plugin": ["linxi_brainelecqc"],
        "steps": [{"stage": "quality", "processor_name": "BrainElecQCProcessor", "params": params}]}), encoding="utf-8")
    context = LinxiContext(file_create_date=datetime.now(timezone.utc))
    context.run_state.input_path = str(source)
    context.run_state.work_dir = directory / "task"
    print(f"{name}: official TaskRunner, default dry run, EEG QC and figures", flush=True)
    returned = TaskRunner(PipelineDefinition.from_file(pipeline_path)).run(
        context, skip_dry_run=False, allow_errors=False)
    summary = context.get_metric("brainelecqc")
    write_json(directory / "context_summary.json", summary)
    bundle = Path(summary["bundle_dir"])
    manifest, score = read_json(bundle / "manifest.json"), read_json(bundle / "quality_score.json")
    features, methods = read_json(bundle / "features.json"), read_json(bundle / "paper_methods.json")
    channels = read_json(bundle / "channels.json")
    with np.load(bundle / "review_signals.npz", allow_pickle=False) as archive:
        data = archive["raw_data_volts"]
        references = {key: archive[key] for key in archive.files if key != "raw_data_volts"}
    checks = {"task_returned_context": returned == [context], "task_succeeded": not context.run_state.failed,
              "history_recorded": len(context.history) == 1,
              "complete_status": context.get_metric("brainelecqc_status")["status"] == "completed",
              "source_unchanged": fingerprint(source) == before,
              "unit_volts": manifest["unit"] == "V", "axes_channel_sample": manifest["axes"] == ["channel", "sample"],
              "source_indices": [channel["source_index"] for channel in channels] == selection["eeg_indices"],
              "bundle_fingerprints": all(fingerprint(bundle / name)["sha256"] == entry["sha256"]
                                         for name, entry in manifest["files"].items()),
              "integer_score": all(type(row["auto_points"]) is int for row in score["rows"] if row["auto_points"] is not None),
              "score_summary": all(summary[key] == score[key] for key in ("auto_total", "assessed_max", "pending_max", "score_100", "assessment_coverage")),
              "score_report_exists": Path(summary["score_report_path"]).is_file(),
              "percentage_score": score["score_100"] == (200*score["auto_total"]+score["assessed_max"])//(2*score["assessed_max"]),
              "human_adjusts_evaluated_only": all(row["human_adjustable"] == (row["auto_points"] is not None) for row in score["rows"]),
              "score_sum": sum(row["auto_points"] for row in score["rows"] if row["auto_points"] is not None) == score["auto_total"],
              "pending_review": summary["review_status"] == "pending" and score["human_total"] is None,
              "original_samples_retained": bool(np.load(bundle / "keep_mask.npy", allow_pickle=False).all()),
              "retained_data_exact": bool(np.array_equal(data, np.load(bundle / "masked_data_volts.npy", allow_pickle=False))),
              "figure_formats": bool(summary["figures"]) and all(Path(p).suffix == ".png" and Path(p).is_file()
                                    and Path(p).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                                    for p in summary["figures"].values())
                                    and not any(p.suffix.lower() in {".svg", ".pdf"} for p in bundle.iterdir())}
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    try:
        start, stop = manifest["start_sample"], manifest["stop_sample"]
        reference_data = raw.get_data(picks=selection["eeg_indices"], start=start, stop=stop)
        reader_error = float(np.max(np.abs(data - reference_data)))
        checks["mne_voltage_exact"] = reader_error == 0
        checks["source_names"] = [channel["raw_name"] for channel in channels] == [raw.ch_names[i] for i in selection["eeg_indices"]]
        checks["sampling_rate"] = manifest["sampling_rate_hz"] == raw.info["sfreq"]
        checks["sample_range"] = [start, stop] == [round(selection["start_seconds"] * raw.info["sfreq"]),
            round((selection["start_seconds"] + selection["duration_seconds"]) * raw.info["sfreq"])]
        checks["original_clock"] = manifest["start_seconds"] == selection["start_seconds"]
        for kind, index in selection["reference_indices"].items():
            checks[f"reference_{kind}_exact"] = bool(np.array_equal(references[kind], raw.get_data(picks=[index], start=start, stop=stop)[0]))
    finally:
        raw.close()
    print(f"{name}: compare saved results with PyPREP, MNE and author FOOOF", flush=True)
    bad = compare_with_pyprep(data, features["sampling_rate_hz"], features["channel_ids"], features["channel_names"], groups=groups)
    spectral = compare_spectral_reference(data, methods["spectral"])
    events = compare_events_to_mne(data, references, methods["events"])
    checks.update(pyprep_measurements_match=bad["all_compared_measurements_match"], spectral_reference_passed=spectral["passed"],
                  **event_checks(events))
    if baseline:
        previous = baseline / name
        old_score = read_json(previous / "quality_score.json")
        checks["previous_integer_scores"] = [(r["dimension_id"], r["auto_points"]) for r in score["rows"]] == [
            (r["dimension_id"], r["auto_points"]) for r in old_score["rows"]]
        with np.load(previous / "review_signals.npz", allow_pickle=False) as prior:
            checks["previous_input_exact"] = bool(np.array_equal(data, prior["raw_data_volts"]))
    write_json(directory / "reference_comparison.json", {"mne_voltage_max_abs_error_volts": reader_error,
        "bad_channels": bad, "spectral": spectral, "events": events})
    result = {"case": name, "input": str(source), "source_fingerprint": before,
        "shape": list(data.shape), "sampling_rate_hz": manifest["sampling_rate_hz"],
        "start_seconds": manifest["start_seconds"], "duration_seconds": selection["duration_seconds"],
        "score": {key: score[key] for key in ("auto_total", "assessed_max", "pending_max", "rubric_version", "score_100", "assessment_coverage")},
        "score_rows": [{key: row[key] for key in ("dimension_id", "label", "auto_points", "max_points", "score_reason")} for row in score["rows"]],
        "figures": summary["figures"], "report_path": summary["report_path"], "bundle_dir": str(bundle),
        "mne_voltage_max_abs_error_volts": reader_error, "checks": checks, "passed": all(checks.values())}
    write_json(directory / "acceptance.json", result)
    print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}; {score['score_100']} / 100", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eegmmidb-dir", type=Path, required=True)
    parser.add_argument("--hmc", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or any((p / ".git").exists() for p in (output, *output.parents)):
        parser.error("Choose a fresh output directory outside Git")
    sources = args.eegmmidb_dir.expanduser().resolve()
    cases = [(sources / f"{name}.edf", name, {"eeg_indices": list(range(64)), "reference_indices": {},
              "start_seconds": 0, "duration_seconds": 61}, ["analysis_same_recording"] * 64) for name in ("S001R01", "S003R02")]
    cases.append((args.hmc.expanduser().resolve(), "HMC100-200", {"eeg_indices": [0, 1, 2, 3],
        "reference_indices": {"ecg": 7, "eog": 5, "emg": 4}, "start_seconds": 100, "duration_seconds": 100}, ["M1", "M1", "M1", "M2"]))
    for source, *_ in cases:
        if not source.is_file():
            parser.error(f"Supply the existing authentic EEG source: {source}")
    output.mkdir(parents=True)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    results = [run_case(case, output, args.baseline) for case in cases]
    acceptance = {"schema_version": 1, "completed_at": datetime.now(timezone.utc).isoformat(),
        "execution": "PipelineDefinition.from_file -> TaskRunner.run(skip_dry_run=False, allow_errors=False)",
        "software": {name: version(name) for name in ("linxi", "linxi-brainelecqc", "brainelecqc", "numpy", "scipy", "mne", "pyprep", "fooof")},
        "module_paths": {name: importlib.import_module(name).__file__ for name in ("linxi", "linxi_brainelecqc", "brainelecqc")},
        "cases": results, "passed": all(result["passed"] for result in results)}
    write_json(output / "acceptance.json", acceptance)
    lines = ["# BrainElecQCProcessor 真实 EEG 验收", "", "三份已有 EEG 经 Linxi 官方 YAML 和 TaskRunner 生成自动整数评分、复核表及图。",
        "", "| 记录 | 自动初评 | 检测覆盖权重 | 数值与接入检查 | 报告 |", "| --- | ---: | ---: | --- | --- |"]
    for result in results:
        score = result["score"]
        report = Path(result["report_path"]).relative_to(output).as_posix()
        lines.append(f"| {result['case']} | {score['score_100']} / 100 | {score['assessment_coverage']['evaluated_weight']}/100 | {'通过' if result['passed'] else '查看差异'} | [自动报告]({report}) |")
    lines += ["", "初评为已评项目按固定权重折算的整数；覆盖字段记录实际检查范围。人工核对调整已有分项得分。",
              "", "[运行参数、输入指纹和检查结果](acceptance.json)", ""]
    (output / "index.md").write_text("\n".join(lines), encoding="utf-8")
    if not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
