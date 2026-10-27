"""Export a new reproducible QC result directory with optional Python figures."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .report import render_eeg_review_report, render_eeg_review_items
from .methods import render_measurement_methods


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _table(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_eeg_qc(result, output_dir, *, figures=True, title="EEG QC", display_start_seconds=None,
                   display_duration_seconds=15):
    """Publish to a fresh directory. All output files remain outside Git checkouts."""
    if "method_results" in result:
        from .binding import validate_result_identity
        validate_result_identity(result)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new output directory to preserve the existing QC result")
    if any((p/".git").exists() for p in (output, *output.parents)):
        raise ValueError("Choose an output directory outside the Git checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".eeg-qc-", dir=output.parent))
    try:
        for key in ("features", "decisions", "verification"):
            _json(staging/f"{key}.json", result[key])
        _json(staging/"channels.json", result["channels"])
        np.save(staging/"keep_mask.npy", result["keep_mask"], allow_pickle=False)
        np.save(staging/"masked_data_volts.npy", result["masked_data_volts"], allow_pickle=False)
        # Raw data + references support exact report regeneration and side-by-side review.
        np.savez(staging/"review_signals.npz", raw_data_volts=result["raw_data_volts"], **result["references"])
        summaries = {r["channel_id"]: r for r in result["decisions"]["channel_summary"]}
        rows = [{**c, **summaries[c["channel_id"]]} for c in result["channels"]]
        _table(staging/"channels.tsv", rows, ["channel_id", "raw_name", "source_index", "unit", "status",
            "excluded_seconds", "retained_seconds", "before_ac_rms_uv", "retained_ac_rms_uv"])
        fields = ["channel_id", "start_sample", "stop_sample", "start_seconds", "stop_seconds", "kind", "reason", "decision_source"]
        _table(staging/"intervals.tsv", result["decisions"]["automatic_exclusions"], fields)
        _table(staging/"review_queue.tsv", result["decisions"]["review_queue"], fields)
        _table(staging/"retained_intervals.tsv", result["retained_intervals"], fields[:5]+["status"])
        plot_config = {"title": title, "display_start_seconds": display_start_seconds,
                       "display_duration_seconds": display_duration_seconds}
        figure_keys = []
        if "method_results" in result:
            from .paper_report import render_paper_review, render_paper_items, render_paper_methods
            from .score_report import write_score_files, render_score_review
            from .spectral_summary import write_alpha_files
            write_score_files(result, staging)
            write_alpha_files(result, staging)
            methods = result["method_results"]
            _json(staging/"paper_methods.json", methods)
            for kind in ("continuity", "muscle"):
                if kind in methods:
                    _table(staging/f"{kind}_intervals.tsv", methods[kind]["candidates"],
                        ["candidate_id", "channel_id", "channel_name", "kind", "start_sample", "stop_sample",
                         "start_seconds", "stop_seconds", "peak_z_score", "max_power_ratio", "max_reference_envelope_correlation"])
            if "muscle" in methods:
                muscle_rows = [{"channel_id": c["channel_id"], "channel_name": c["channel_name"], **r}
                               for c in methods["muscle"]["channels"] for r in c["window_metrics"]]
                _table(staging/"muscle_windows.tsv", muscle_rows, ["channel_id", "channel_name", "start_sample", "stop_sample",
                    "start_seconds", "stop_seconds", "high_frequency_power_uv2", "peak_z_score", "power_ratio",
                    "reference_peak_z_score", "reference_envelope_correlation", "gates", "flagged", "kind"])
            _table(staging/"bad_channels.tsv", methods["bad_channels"]["channels"], ["channel_id", "channel_name", "group_id",
                "robust_amplitude_uv", "amplitude_z", "hf_ratio", "hf_z", "low_correlation_fraction", "dropout_fraction", "flags"])
            _table(staging/"line_windows.tsv", methods["spectral"]["line_noise"]["observations"], ["channel_id", "channel_name",
                "start_sample", "stop_sample", "start_seconds", "stop_seconds", "nominal_hz", "peak_frequency_hz", "f_statistic",
                "p_value", "p_corrected", "amplitude_uv", "flagged", "status"])
            fits = methods["spectral"]["parameterization"]
            fit_rows = [{**r, "fit_scope": scope} for scope in ("full", "windows") for r in fits[scope]]
            _table(staging/"spectral_fits.tsv", fit_rows, ["channel_id", "channel_name", "fit_scope", "start_sample", "stop_sample",
                "start_seconds", "stop_seconds", "status", "alpha_status", "alpha_peaks", "aperiodic_parameters", "r_squared", "mae_log10_power"])
            for kind, block in methods["events"]["by_kind"].items():
                _table(staging/f"events_{kind}.tsv", block["event_table"], ["event_id", "sample", "time_seconds",
                    "epoch_start_sample", "epoch_stop_sample", "epoch_start_seconds", "epoch_stop_seconds", "complete_epoch",
                    "included_in_average", "interval_from_previous_seconds"])
            if figures:
                from ..plotting.paper_eeg import make_paper_eeg_figures
                from ..plotting import save_eeg_figures
                figs = make_paper_eeg_figures(result, **plot_config)
                figure_keys = list(figs)
                _json(staging/"figure_selection.json", {k: getattr(fig, "_brainelecqc_display_metadata", {}) for k, fig in figs.items()})
                save_eeg_figures(figs, staging)
            plot_config["figure_keys"] = figure_keys
            (staging/"score_report.md").write_text(render_score_review(result, plot_config, with_figures=figures), encoding="utf-8")
            (staging/"review_report.md").write_text(render_paper_review(result, plot_config, with_figures=figures), encoding="utf-8")
            (staging/"review_items.md").write_text(render_paper_items(result), encoding="utf-8")
            (staging/"methods.md").write_text(render_paper_methods(result), encoding="utf-8")
        elif figures:
            from ..plotting import make_eeg_figures, save_eeg_figures
            figs = make_eeg_figures(result["raw_data_volts"], result["features"], result["decisions"],
                                   result["keep_mask"], **plot_config)
            save_eeg_figures(figs, staging)
            figure_keys = list(figs)
        if "method_results" not in result:
            (staging/"review_report.md").write_text(render_eeg_review_report(
                result["features"], result["decisions"], plot_config, with_figures=figures), encoding="utf-8")
            (staging/"review_items.md").write_text(render_eeg_review_items(
                result["features"], result["decisions"]), encoding="utf-8")
            (staging/"methods.md").write_text(render_measurement_methods(
                result["features"], result["decisions"]), encoding="utf-8")
        manifest = {**result["manifest"], "plot_config": plot_config,
            "summary": result["decisions"]["summary"], "files": {p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
                for p in sorted(staging.iterdir()) if p.is_file()}}
        _json(staging/"manifest.json", manifest)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return {"output_dir": str(output), "run_id": manifest["run_id"], "summary": manifest["summary"],
            "report": str(output/"review_report.md"),
            "score_report": str(output/"score_report.md") if "method_results" in result else None,
            "figures": {key: str(output/f"{key}.png") for key in figure_keys}}
