"""Plots consume the same physical values, identities and clocks as the report."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.eeg.export import export_eeg_qc
from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.plotting.paper_eeg import make_paper_eeg_figures, build_focus_plan
from brainelecqc.eeg.spectral_summary import spectral_display_rows
from tests.test_eeg_pipeline import _import, _source


def test_paper_figures_display_physical_psd_original_samples_and_focused_panels(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact, eeg_indices=[2, 0])
    figures = make_paper_eeg_figures(result, display_start_seconds=2, display_duration_seconds=3)
    try:
        before = figures["01_initial"].axes[0]
        ci = int(before.get_gid().split(":")[-1])
        np.testing.assert_array_equal(before.lines[0].get_xdata(), np.arange(512, 1280)/256)
        np.testing.assert_array_equal(before.lines[0].get_ydata(), result["raw_data_volts"][ci, 512:1280]*1e6)
        assert before.lines[0].get_xdata()[0] == 2
        assert "07_retained" not in figures and "09_spectral_fit" not in figures
        assert len(figures["03_all_channels_psd"].axes[0].lines) == 2
        psd = result["method_results"]["spectral"]["psd"]
        axis = figures["03_psd_alpha"].axes[0]
        rows = spectral_display_rows(result)
        for line, row in zip(axis.lines, rows):
            expected = np.asarray(psd["power_v2_hz"][row["channel_index"]])*1e12
            np.testing.assert_allclose(line.get_ydata(), np.where(expected > 0, expected, np.nan))
            assert line.get_label() == row["display_name"]
        assert "0–8 s" in axis.get_title(loc="left")
        assert axis.get_ylabel() == "PSD (µV²/Hz)"
        assert rows[0]["channel_name"] == "EEG Oz"
        for ax, field in zip(figures["03_psd_alpha"].axes[1:], ("alpha_power_uv2", "relative_alpha_percent")):
            np.testing.assert_allclose([patch.get_width() for patch in ax.patches], [r[field] for r in rows])
        alpha = result["method_results"]["spectral"]["alpha_power"]
        windows = [w for w in alpha["windows"] if w["channel_index"] == rows[0]["channel_index"]]
        np.testing.assert_allclose(figures["08_alpha_time"].axes[0].lines[0].get_xdata(),
                                   [(w["start_seconds"]+w["stop_seconds"])/2 for w in windows])
        np.testing.assert_allclose(figures["08_alpha_time"].axes[0].lines[0].get_ydata(), [w["alpha_power_uv2"] for w in windows])
        assert {a.get_gid() for a in figures["02_bad_channels"].axes} == {"bad:psd", "bad:raw:0", "bad:raw:1"}
        assert {a.get_gid() for a in figures["04_line_noise"].axes if a.get_gid()} == {"line:50", "line:60"}
        for key in ("02_bad_channels", "03_all_channels_psd", "04_line_noise", "05_ecg", "06_eog"):
            json.dumps(figures[key]._brainelecqc_display_metadata, allow_nan=False)
    finally:
        for figure in figures.values():
            plt.close(figure)


@pytest.mark.parametrize("legacy", [False, True])
def test_saved_bundle_redraw_is_self_contained_and_detects_tampering(tmp_path, legacy):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    if legacy:
        del result["method_results"]["spectral"]["alpha_power"]
    bundle, redrawn = tmp_path/"bundle", tmp_path/"redrawn"
    export_eeg_qc(result, bundle, figures=False)
    script = Path(__file__).resolve().parents[1]/"scripts/figure_eeg_qc.py"
    command = [sys.executable, str(script), str(bundle), str(redrawn)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((redrawn/"manifest.json").read_text())
    assert manifest["parent_bundle"] == str(bundle)
    assert {"paper_methods.json", "review_signals.npz", "01_initial.png", "03_psd_alpha.png"} <= manifest["files"].keys()
    assert {"quality_score.json", "score_review.tsv", "score_evidence.tsv"} <= manifest["files"].keys()
    assert {"alpha_power.tsv", "spectral_display.json", "08_alpha_time.png", "03_all_channels_psd.png"} <= manifest["files"].keys()
    assert not {"07_retained.png", "09_spectral_fit.png"} & manifest["files"].keys()
    assert not any(Path(name).suffix in {".svg", ".pdf"} for name in manifest["files"])
    assert manifest["report_derivation"]["added_measurements"] == (["eeg-alpha-power-1.0"] if legacy else [])
    original_methods = json.loads((bundle/"paper_methods.json").read_text())
    assert ("alpha_power" in original_methods["spectral"]) != legacy
    assert json.loads((bundle/"quality_score.json").read_text()) == json.loads((redrawn/"quality_score.json").read_text())
    assert (redrawn/"03_psd_alpha.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report_only = tmp_path/"report-only"
    completed = subprocess.run([sys.executable, str(script), str(bundle), str(report_only), "--report-only"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert ".png" not in (report_only/"review_report.md").read_text()
    assert json.loads((bundle/"quality_score.json").read_text()) == json.loads((report_only/"quality_score.json").read_text())
    (bundle/"paper_methods.json").write_text("{}")
    command[-1] = str(tmp_path/"tampered")
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "fingerprint mismatch" in completed.stderr
    assert not (tmp_path/"tampered").exists()


def test_focus_plan_adds_endpoint_page_and_preserves_saved_measurements(tmp_path):
    source, _, _ = _source(tmp_path, clipped=True)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    saved = copy.deepcopy(result["method_results"])
    plan = build_focus_plan(result)
    page = next(p for p in plan["bad_channel_pages"] if p["kind"] == "calibration_endpoint_plateau")
    assert page["representative_candidate"]["start_sample"] == 600
    assert page["representative_candidate"]["stop_sample"] == 650
    assert page["start_sample"] <= 600 < 650 <= page["stop_sample"]
    assert page["channel_indices"] == [0, 1]
    assert page["control_selection"] == "same_reference_group_without_candidates"
    json.dumps(plan, allow_nan=False)
    assert result["method_results"] == saved
    figures = make_paper_eeg_figures(result, figure_keys=[])
    try:
        page_figure = figures[page["figure_key"]]
        assert len(page_figure.axes) == 3
        assert page_figure._brainelecqc_display_metadata["selection_plan"] == page
    finally:
        for figure in figures.values():
            plt.close(figure)
