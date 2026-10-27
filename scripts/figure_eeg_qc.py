"""Regenerate EEG reports and figures from a saved result bundle using Python.

Usage: python scripts/figure_eeg_qc.py /path/to/qc-result /path/to/new-figures
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from brainelecqc.plotting import make_eeg_figures, save_eeg_figures
from brainelecqc.eeg.masks import apply_eeg_decisions
from brainelecqc.eeg.report import render_eeg_review_report, render_eeg_review_items
from brainelecqc.eeg.methods import render_measurement_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Update scorecards and reuse saved figures")
    parser.add_argument("--header-source", type=Path, help="Inspect the original input header after verifying its saved SHA256")
    args = parser.parse_args()
    root, output = args.result_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if output.exists() or any((p/".git").exists() for p in (output, *output.parents)):
        parser.error("Choose a new output directory outside the Git checkout")
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, fingerprint in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != fingerprint["sha256"]:
            parser.error(f"Result fingerprint mismatch: {name}")
    features = json.loads((root/"features.json").read_text(encoding="utf-8"))
    decisions = json.loads((root/"decisions.json").read_text(encoding="utf-8"))
    with np.load(root/"review_signals.npz", allow_pickle=False) as signals:
        data = signals["raw_data_volts"]
        references = {key: signals[key] for key in signals.files if key != "raw_data_volts"}
    _, keep = apply_eeg_decisions(data, decisions, channel_ids=features["channel_ids"])
    if "paper_methods.json" in manifest["files"]:
        from brainelecqc.plotting.paper_eeg import make_paper_eeg_figures
        from brainelecqc.eeg.paper_report import render_paper_review, render_paper_items, render_paper_methods
        from brainelecqc.eeg.score_report import write_score_files, render_score_review
        import shutil
        methods = json.loads((root/"paper_methods.json").read_text(encoding="utf-8"))
        result = {"features": features, "decisions": decisions, "manifest": manifest, "method_results": methods,
                  "raw_data_volts": data, "keep_mask": keep, "references": references}
        from brainelecqc.eeg.binding import validate_result_identity
        from brainelecqc.eeg.spectral_summary import write_alpha_files
        validate_result_identity(result)
        added_measurements = []
        if args.header_source:
            from brainelecqc.eeg.header_qc import inspect_input_header
            methods["header"] = inspect_input_header(args.header_source, expected_sha256=manifest["source_sha256"])
            added_measurements.append(methods["header"]["schema_version"])
            validate_result_identity(result)
        if not args.report_only and "alpha_power" not in methods["spectral"]:
            from brainelecqc.eeg.alpha_power import compute_alpha_power
            methods["spectral"]["alpha_power"] = compute_alpha_power(data, methods["spectral"])
            added_measurements.append(methods["spectral"]["alpha_power"]["schema_version"])
        if args.report_only:
            output.mkdir(parents=True)
            for name in manifest["files"]:
                if Path(name).suffix == ".png":
                    shutil.copy2(root/name, output/name)
            with_figures = "01_initial.png" in manifest["files"]
            manifest["plot_config"]["figure_keys"] = [Path(name).stem for name in manifest["files"] if name.endswith(".png")]
        else:
            figs = make_paper_eeg_figures(result, **manifest["plot_config"])
            save_eeg_figures(figs, output)
            manifest["plot_config"]["figure_keys"] = list(figs)
            (output/"figure_selection.json").write_text(json.dumps({k:getattr(fig,"_brainelecqc_display_metadata",{})
                for k,fig in figs.items()},ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            with_figures = True
        review = write_score_files(result, output)
        previous_review = json.loads((root/"quality_score.json").read_text(encoding="utf-8")) if "quality_score.json" in manifest["files"] else {}
        if ("event_impact" in review["scoring_policy"] and review["scoring_policy"]["event_impact"]["version"] !=
                previous_review.get("scoring_policy", {}).get("event_impact", {}).get("version")):
            added_measurements.append(review["scoring_policy"]["event_impact"]["version"])
        write_alpha_files(result, output)
        (output/"paper_methods.json").write_text(json.dumps(methods, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (output/"review_report.md").write_text(render_paper_review(result, manifest["plot_config"], with_figures=with_figures), encoding="utf-8")
        (output/"score_report.md").write_text(render_score_review(result, manifest["plot_config"], with_figures=with_figures), encoding="utf-8")
        (output/"review_items.md").write_text(render_paper_items(result), encoding="utf-8")
        (output/"methods.md").write_text(render_paper_methods(result), encoding="utf-8")
        regenerated = {"review_report.md", "score_report.md", "review_items.md", "methods.md", "quality_score.json",
                       "score_review.tsv", "score_evidence.tsv", "quality_overview.tsv", "paper_methods.json", "alpha_power.tsv", "spectral_display.json"}
        if not args.report_only:
            regenerated.add("figure_selection.json")
        for name in manifest["files"]:
            if Path(name).suffix not in {".png", ".svg", ".pdf"} and name not in regenerated:
                shutil.copy2(root/name, output/name)
        new_manifest = {**manifest, "parent_bundle": str(root),
                        "report_derivation": {"added_measurements": added_measurements},
                        "parent_manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
                                  for p in sorted(output.iterdir()) if p.is_file()}}
        (output/"manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(output/"review_report.md")
        return
    if args.report_only:
        parser.error("Scorecards require a saved paper_methods.json result")
    figures = make_eeg_figures(data, features, decisions, keep, **manifest["plot_config"])
    save_eeg_figures(figures, output)
    (output/"review_report.md").write_text(render_eeg_review_report(features, decisions, manifest["plot_config"]), encoding="utf-8")
    (output/"review_items.md").write_text(render_eeg_review_items(features, decisions), encoding="utf-8")
    if features.get("measurement_methods"):
        (output/"methods.md").write_text(render_measurement_methods(features, decisions), encoding="utf-8")
    print(output/"review_report.md")


if __name__ == "__main__":
    main()
