"""Reproduce EEG QC reference comparisons on seven prespecified real recordings.

Inputs are caller-provided, authentic HMC and EEGMMIDB files. Source selection
is fixed before analysis: S001--S003, resting runs R01/R02, and one HMC input.
Outputs and the immutable input store are required to live outside Git.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import sys
import traceback

import numpy as np


EEGMMIDB_CASES = tuple(f"S{subject:03d}R{run:02d}" for subject in (1, 2, 3) for run in (1, 2))


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external(path):
    path = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError(f"Select a path outside Git for data and results: {path}")
    return path


def _sources(directory, hmc):
    selected = []
    for case in EEGMMIDB_CASES:
        choices = [directory / f"{case}.edf", directory / case[:4] / f"{case}.edf"]
        found = [path for path in choices if path.is_file()]
        if len(found) != 1:
            raise ValueError(f"Provide exactly one {case}.edf in source-dir or its {case[:4]} subdirectory")
        selected.append({"case": case, "path": found[0], "dataset": "EEG Motor Movement/Imagery 1.0.0",
                         "dataset_url": "https://physionet.org/content/eegmmidb/1.0.0/",
                         "condition": "eyes_open" if case.endswith("01") else "eyes_closed",
                         "duration_seconds": 61.0, "eeg_indices": list(range(64)), "reference_indices": {},
                         "groups": ["analysis_same_recording"] * 64,
                         "group_metadata": {"comparison_group_basis": "prespecified same-recording comparison group",
                                            "acquisition_reference": "not reported in source page",
                                            "group_assumption": "shared acquisition reference; review assumption"}})
    if not hmc.is_file():
        raise ValueError("Provide the existing HMC EDF file with --hmc")
    selected.append({"case": "HMC", "path": hmc, "dataset": "HMC sleep staging 1.1",
                     "dataset_url": "https://physionet.org/content/hmc-sleep-staging/1.1/",
                     "condition": "sleep_recording; per-epoch sleep state requires scoring annotations",
                     "duration_seconds": 300.0, "eeg_indices": [0, 1, 2, 3],
                     "reference_indices": {"ecg": 7, "eog": 5, "emg": 4}, "groups": ["M1", "M1", "M1", "M2"],
                     "group_metadata": {"comparison_group_basis": "reference labels in the original channel names",
                                        "acquisition_reference": "F4-M1, C4-M1, O2-M1, C3-M2"}})
    return selected


def _event_checks(comparison):
    checks = {}
    for kind, block in comparison["by_kind"].items():
        if block["status"] != "compared":
            continue
        checks[f"{kind}_indices"] = block["event_indices_equal"]
        for key, value in block.items():
            if "max_abs_error" in key and value is not None:
                checks[f"{kind}_{key}"] = value <= 1e-14 if key.endswith("seconds") else value <= 1e-18
    return checks


def _case_summary(case, result, checks, exported):
    methods = result["method_results"]
    events = methods["events"]["by_kind"]
    return {"case": case["case"], "source_filename": case["path"].name,
            "dataset": case["dataset"], "dataset_url": case["dataset_url"], "condition": case["condition"],
            "source_sha256": case["source_sha256"], "group_metadata": case["group_metadata"],
            "recording_id": result["manifest"]["recording_id"], "artifact_id": result["manifest"]["artifact_id"],
            "input_data_sha256": result["manifest"]["data_sha256"],
            "shape": list(result["raw_data_volts"].shape), "sampling_rate_hz": result["manifest"]["sampling_rate_hz"],
            "start_seconds": result["manifest"]["start_seconds"], "duration_seconds": case["duration_seconds"],
            "bad_channel_candidate_categories": dict(Counter(row["kind"] for row in methods["bad_channels"]["candidates"])),
            "bad_channel_candidate_channels": sorted(set(row["channel_name"] for row in methods["bad_channels"]["candidates"])),
            "line_flagged_channel_windows": sum(row["flagged"] for row in methods["spectral"]["line_noise"]["observations"]),
            "full_spectrum_alpha_channels": sum(row["alpha_status"] == "peak_present"
                                                for row in methods["spectral"]["parameterization"]["full"]),
            "events": {kind: {key: block[key] for key in ("status", "detected_event_count", "averaged_event_count",
                                                         "boundary_excluded_count")}
                       for kind, block in events.items()},
            "checks": checks, "passed": all(checks.values()),
            "report": f"{case['case']}/review_report.md", "export_run_id": exported["run_id"],
            "reference_comparison": f"reference_comparisons/{case['case']}.json"}


def _index(manifest):
    lines = ["# EEG 论文方法验收", "", "固定样例：S001–S003 的睁眼/闭眼记录各一份，以及 HMC 睡眠记录。",
             "每份报告提供具体通道、原始样本与时间段；原始数值全部保留。", "",
             "| 记录 | 时长 | 坏道候选通道 | 工频候选窗 | 全程 α 峰通道 | ECG / EOG 候选 | 数值对照 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    displayed_cases = list(manifest["cases"])
    if manifest.get("event_sensitivity", {}).get("case_summary"):
        displayed_cases.append(manifest["event_sensitivity"]["case_summary"])
    for case in displayed_cases:
        event_count = " / ".join(str(case["events"][kind]["detected_event_count"])
                                 if case["events"][kind]["status"] != "missing_reference" else "无辅助导联"
                                 for kind in ("ecg", "eog"))
        lines.append(f"| [{case['case']}]({case['report']}) | {case['duration_seconds']:g} s | "
                     f"{len(case['bad_channel_candidate_channels'])} | {case['line_flagged_channel_windows']} | "
                     f"{case['full_spectrum_alpha_channels']} | {event_count} | {'通过' if case['passed'] else '查看差异'} |")
    lines.extend(["", "对照范围：PREP 逐通道与逐窗测量对 PyPREP；多窗工频统计对 MNE；"
                  "PSD 对 SciPy；周期/非周期拟合对作者 FOOOF；事件位置与锁时汇总对 MNE。",
                  "", "EEGMMIDB 的同记录比较组按本次分析假设指定，人工复核时核对采集参考。"
                  "HMC 按原始导联中的 M1/M2 分组。",
                  "", "[完整验收及源文件指纹](acceptance.json) · [HMC EOG 阈值敏感性](hmc_event_sensitivity.json)",
                  "", "HMC 对照使用同一默认事件方法分别分析 0–300 s 与 100–200 s；"
                  "核对辅助导联上的大幅波动如何改变范围阈值和候选数量。",
                  "", "人工复核：确认采集参考与记录状态；查看候选原波形、频谱或锁时片段；记录保留、标注或处理决定。",
                  "数值验收记录实现之间的一致性；事件和坏道的准确率由独立人工标签评估。", ""])
    if manifest.get("failure"):
        lines.extend(["运行状态：" + manifest["failure"]["message"], ""])
    return "\n".join(lines)


def _sensitivity(record, source_case, original, output, *, figures):
    from brainelecqc.eeg.bad_channels import compare_with_pyprep
    from brainelecqc.eeg.events import compare_events_to_mne
    from brainelecqc.eeg.export import export_eeg_qc
    from brainelecqc.eeg.pipeline import run_eeg_qc
    from brainelecqc.eeg.spectral import compare_spectral_reference

    case = {**source_case, "case": "HMC100-200", "duration_seconds": 100}
    print("HMC100-200: interval sensitivity and full report", flush=True)
    result = run_eeg_qc(record["input_artifact_id"], start_seconds=100, duration_seconds=100,
                        eeg_indices=case["eeg_indices"], reference_indices=case["reference_indices"],
                        method_config={"comparison_groups": case["groups"]})
    data, features, methods = result["raw_data_volts"], result["features"], result["method_results"]
    comparison = compare_events_to_mne(data, result["references"], methods["events"])
    bad = compare_with_pyprep(data, features["sampling_rate_hz"], features["channel_ids"],
                              features["channel_names"], groups=case["groups"])
    spectral = compare_spectral_reference(data, methods["spectral"])
    fs = features["sampling_rate_hz"]
    expected = original["raw_data_volts"][:, round(100 * fs):round(200 * fs)]
    checks = {"source_unchanged": _hash(case["path"]) == case["source_sha256"],
              "reader_matches_validated_parent_interval": bool(np.array_equal(data, expected)),
              "all_input_samples_retained": bool(result["keep_mask"].all()),
              "output_values_exact": bool(np.array_equal(data, result["masked_data_volts"])),
              "pyprep_measurements_match": bad["all_compared_measurements_match"],
              "spectral_reference_passed": spectral["passed"], **_event_checks(comparison)}
    exported = export_eeg_qc(result, output / case["case"], figures=figures,
                              title="HMC100-200 EEG QC", display_start_seconds=100, display_duration_seconds=15)
    _json(output / "reference_comparisons" / f"{case['case']}.json",
          {"source_sha256": case["source_sha256"], "reader_max_abs_error_volts": float(np.max(np.abs(data - expected))),
           "bad_channels": bad, "spectral": spectral, "events": comparison, "checks": checks})
    keys = ("status", "detected_event_count", "averaged_event_count", "detector", "event_time_span_seconds", "event_count_windows")
    payload = {"recording_id": record["recording_id"], "input_artifact_id": record["input_artifact_id"],
               "source_sha256": source_case["source_sha256"],
               "comparison": "same reference detector defaults on two explicitly selected intervals",
               "whole_0_300_seconds": {kind: {key: block.get(key) for key in keys}
                                       for kind, block in original["method_results"]["events"]["by_kind"].items()},
               "subset_100_200_seconds": methods["events"], "reference_comparison": comparison,
               "checks": checks, "report": f"{case['case']}/review_report.md"}
    _json(output / "hmc_event_sensitivity.json", payload)
    return {"passed": all(checks.values()), "checks": checks, "case_summary": _case_summary(case, result, checks, exported)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Directory containing the six prespecified EEGMMIDB EDFs")
    parser.add_argument("--hmc", required=True, type=Path, help="Existing HMC EDF with four EEG, ECG and EOG channels")
    parser.add_argument("--output", required=True, type=Path, help="New result directory outside Git")
    parser.add_argument("--storage-root", required=True, type=Path, help="Independent immutable-input store outside Git")
    parser.add_argument("--no-figures", action="store_true", help="Run numeric comparisons and write reports without figures")
    args = parser.parse_args(argv)
    output, storage = _external(args.output), _external(args.storage_root)
    if output.exists():
        raise ValueError("Choose a new output directory; existing acceptance evidence is preserved")
    cases = _sources(args.source_dir.expanduser().resolve(), args.hmc.expanduser().resolve())
    for case in cases:
        if case["path"].is_symlink():
            raise ValueError("Provide regular source files instead of symlinks")
        case["source_sha256"] = _hash(case["path"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference_comparisons").mkdir()
    import mne
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    from brainelecqc.config import storage_context
    from brainelecqc.db import init_db
    from brainelecqc.recordings import import_recording, verify_recording
    from brainelecqc.eeg.bad_channels import compare_with_pyprep
    from brainelecqc.eeg.events import compare_events_to_mne
    from brainelecqc.eeg.export import export_eeg_qc
    from brainelecqc.eeg.pipeline import run_eeg_qc
    from brainelecqc.eeg.spectral import compare_spectral_reference

    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "selection": "prespecified S001-S003 R01/R02 plus supplied HMC; source hashes recorded before analysis",
                "comparison_scope": "real source input, numerical reference agreement, traceable channel/time mapping",
                "software": {name: version(name) for name in ("brainelecqc", "numpy", "scipy", "mne", "pyprep", "fooof")},
                "cases": [], "passed": False}
    try:
        with storage_context(storage):
            init_db()
            for case in cases:
                print(f"{case['case']}: import and calculate", flush=True)
                payload = {"label": f"EEG paper methods {case['case']}", "modality": "eeg",
                           "participant_id": case["case"][:4],
                           "import_key": f"paper-eeg:{case['case']}:{case['source_sha256']}",
                           "metadata": {"dataset": case["dataset"], "dataset_url": case["dataset_url"],
                                        "condition": case["condition"], **case["group_metadata"]}}
                record, _ = import_recording(payload, paths=[case["path"]])
                result = run_eeg_qc(record["input_artifact_id"], start_seconds=0,
                                    duration_seconds=case["duration_seconds"], eeg_indices=case["eeg_indices"],
                                    reference_indices=case["reference_indices"],
                                    method_config={"comparison_groups": case["groups"]})
                data, features = result["raw_data_volts"], result["features"]
                print(f"{case['case']}: reference comparisons", flush=True)
                bad = compare_with_pyprep(data, features["sampling_rate_hz"], features["channel_ids"],
                                          features["channel_names"], groups=case["groups"])
                spectral = compare_spectral_reference(data, result["method_results"]["spectral"])
                events = compare_events_to_mne(data, result["references"], result["method_results"]["events"])
                reference = mne.io.read_raw_edf(case["path"], preload=False, verbose=False)
                try:
                    expected = reference.get_data(picks=case["eeg_indices"], start=0, stop=data.shape[1])
                    same_names = [reference.ch_names[i] for i in case["eeg_indices"]] == features["channel_names"]
                finally:
                    reference.close()
                checks = {"immutable_input_verified": verify_recording(record["recording_id"])["valid"],
                          "source_unchanged": _hash(case["path"]) == case["source_sha256"],
                          "reader_matches_mne": bool(np.array_equal(data, expected)), "channel_order_matches_mne": same_names,
                          "all_input_samples_retained": bool(result["keep_mask"].all()),
                          "output_values_exact": bool(np.array_equal(data, result["masked_data_volts"])),
                          "pyprep_measurements_match": bad["all_compared_measurements_match"],
                          "spectral_reference_passed": spectral["passed"], **_event_checks(events)}
                for group in bad["groups"]:
                    comparison = group.get("candidate_comparison", {})
                    for name, matches in comparison.get("common_window_matches", {}).items():
                        checks[f"pyprep_candidates_{group['group_id']}_{name}"] = matches
                exported = export_eeg_qc(result, output / case["case"], figures=not args.no_figures,
                                          title=f"{case['case']} EEG QC", display_start_seconds=32 if case["case"] == "HMC" else 0,
                                          display_duration_seconds=16 if case["case"] == "HMC" else 15)
                _json(output / "reference_comparisons" / f"{case['case']}.json",
                      {"source_sha256": case["source_sha256"], "reader_max_abs_error_volts": float(np.max(np.abs(data - expected))),
                       "bad_channels": bad, "spectral": spectral, "events": events, "checks": checks})
                manifest["cases"].append(_case_summary(case, result, checks, exported))
                if case["case"] == "HMC":
                    manifest["event_sensitivity"] = _sensitivity(record, case, result, output, figures=not args.no_figures)
                _json(output / "acceptance.json", manifest)
                print(f"{case['case']}: {'passed' if all(checks.values()) else 'comparison differences'}", flush=True)
        manifest["passed"] = all(case["passed"] for case in manifest["cases"]) and manifest["event_sensitivity"]["passed"]
    except Exception as error:
        manifest["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["files"] = {path.relative_to(output).as_posix(): {"sha256": _hash(path), "size_bytes": path.stat().st_size}
                             for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"acceptance.json", "index.md"}}
        _json(output / "acceptance.json", manifest)
        (output / "index.md").write_text(_index(manifest), encoding="utf-8")
    print(json.dumps({"passed": manifest["passed"], "cases": len(manifest["cases"]),
                      "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
