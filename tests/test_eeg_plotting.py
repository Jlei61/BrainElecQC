"""Figure contracts: unchanged input, original clock, matched scales and exports."""

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.plotting import make_eeg_figures, save_eeg_figures


def _example():
    fs = 100
    time = np.arange(1000) / fs
    data = np.array([40e-6 * np.sin(2 * np.pi * 10 * time),
                     800e-6 * np.sin(2 * np.pi * 3 * time)])
    windows = [{"window_index": i, "start_sample": 10000 + i * 200,
                "stop_sample": 10200 + i * 200, "start_seconds": 100 + i * 2,
                "stop_seconds": 102 + i * 2} for i in range(5)]
    features = {
        "channel_ids": ["c0", "c1"], "channel_names": ["EEG A", "EEG B"],
        "sampling_rate_hz": fs, "start_sample": 10000, "stop_sample": 11000,
        "start_seconds": 100, "windows": windows,
        "frequencies_hz": [1, 10, 30, 50],
        "psd_v2_hz": [[1e-12, 2e-10, 5e-13, 1e-14], [2e-9, 1e-10, 1e-11, 1e-12]],
        "window_features": {"rms_uv": [[28] * 5, [565] * 5],
                            "line_50_db": [[1] * 5, [2] * 5],
                            "line_60_db": [[None] * 5, [None] * 5],
                            "alpha_peak_hz": [[10] * 5, [11] * 5],
                            "alpha_prominence_db": [[12] * 5, [2] * 5]},
        "channel_features": [], "reference_status": {},
    }
    exclusion = {"channel_id": "c1", "start_sample": 10200, "stop_sample": 10500,
                 "start_seconds": 102, "stop_seconds": 105, "kind": "clipping",
                 "reason": "反复达到标定边界", "evidence": {}, "decision_source": "automatic"}
    review = {"channel_id": "c0", "start_sample": 10000, "stop_sample": 10200,
              "start_seconds": 100, "stop_seconds": 102, "kind": "line_noise",
              "reason": "核对工频峰和背景活动", "evidence": {}, "decision_source": "review_pending"}
    mask = np.ones(data.shape, dtype=bool)
    mask[1, 200:500] = False
    decisions = {"automatic_exclusions": [exclusion], "review_queue": [review],
                 "summary": {"excluded_channel_seconds": 3, "retained_channel_seconds": 17,
                             "excluded_fraction": .15, "fully_excluded_channel_ids": [], "review_count": 1},
                 "channel_summary": [{"channel_id": "c0", "excluded_seconds": 0, "retained_seconds": 10},
                                     {"channel_id": "c1", "excluded_seconds": 3, "retained_seconds": 7}],
                 "policy_config": {}}
    return data, features, decisions, mask


def _waveform_axes(fig):
    return {ax.get_gid(): ax for ax in fig.axes if (ax.get_gid() or "").startswith("waveform:")}


def test_matching_clock_scales_and_unchanged_inputs():
    data, features, decisions, mask = _example()
    before = data.copy(), copy.deepcopy(features), copy.deepcopy(decisions), mask.copy()
    figures = make_eeg_figures(data, features, decisions, mask,
                               display_start_seconds=101, display_duration_seconds=6)
    try:
        assert list(figures) == ["01_initial", "02_features_review", "03_auto_excluded"]
        initial = _waveform_axes(figures["01_initial"])
        after = _waveform_axes(figures["03_auto_excluded"])
        assert list(initial) == list(after) == ["waveform:c0", "waveform:c1"]
        for key in initial:
            assert initial[key].get_xlim() == after[key].get_xlim() == (101, 107)
            assert initial[key].get_ylim() == after[key].get_ylim()
            assert initial[key].get_ylim()[1] != 0
        assert after["waveform:c1"].get_ylim()[1] > 800
        kept = after["waveform:c1"].lines[1].get_ydata()
        assert np.isnan(kept[100:400]).all()
        assert np.isfinite(kept[:100]).all()
        np.testing.assert_array_equal(after["waveform:c1"].lines[0].get_ydata(), data[1, 100:700] * 1e6)
        np.testing.assert_array_equal(data, before[0])
        np.testing.assert_array_equal(mask, before[3])
        assert features == before[1]
        assert decisions == before[2]
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_export_defaults_to_png_and_closes_figures(tmp_path):
    figures = make_eeg_figures(*_example(), display_duration_seconds=8)
    numbers = [fig.number for fig in figures.values()]
    output = tmp_path/"figures"
    files = save_eeg_figures(figures, output)
    assert len(files) == 3
    for formats in files.values():
        assert set(formats) == {"png"}
        for path in formats.values():
            assert Path(path).is_absolute()
            assert Path(path).stat().st_size > 1000
        assert Path(formats["png"]).read_bytes().startswith(b"\x89PNG")
    assert {p.suffix for p in output.iterdir()} == {".png"}
    assert all(not plt.fignum_exists(number) for number in numbers)


def test_explicit_format_selection_exports_only_requested_formats(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 2])
    axis.set_xlabel("Time (s)")
    output = tmp_path/"figures"
    files = save_eeg_figures({"explicit": figure}, output, formats=["svg", "pdf"])
    assert set(files["explicit"]) == {"svg", "pdf"}
    assert {p.suffix for p in output.iterdir()} == {".svg", ".pdf"}
    assert ET.parse(files["explicit"]["svg"]).findall(".//{http://www.w3.org/2000/svg}text")
    assert Path(files["explicit"]["pdf"]).read_bytes().startswith(b"%PDF")
    assert not plt.fignum_exists(figure.number)


@pytest.mark.parametrize("formats", [[], (), "png", ["png", "png"], ["jpeg"], ["PNG"], [None], None])
def test_invalid_export_formats_fail_before_creating_output_and_close_figures(tmp_path, formats):
    figure = plt.figure()
    output = tmp_path/"not-created"
    with pytest.raises(ValueError, match="formats"):
        save_eeg_figures({"test": figure}, output, formats=formats)
    assert not output.exists()
    assert not plt.fignum_exists(figure.number)


@pytest.mark.parametrize("mutation,message", [
    (lambda data, features, decisions, mask: features.update(stop_sample=10999), "sample interval"),
    (lambda data, features, decisions, mask: features.update(channel_ids=["only_one"]), "channel identities"),
    (lambda data, features, decisions, mask: features.update(psd_v2_hz=[[1, 2]]), "PSD"),
])
def test_reject_inconsistent_inputs(mutation, message):
    args = _example()
    mutation(*args)
    with pytest.raises(ValueError, match=message):
        make_eeg_figures(*args)


def test_reject_mask_shape_and_outside_window():
    data, features, decisions, mask = _example()
    with pytest.raises(ValueError, match="keep_mask"):
        make_eeg_figures(data, features, decisions, mask.astype(int))
    with pytest.raises(ValueError, match="overlap"):
        make_eeg_figures(data, features, decisions, mask, display_start_seconds=500)


def test_heatmap_preserves_gap_and_short_final_window():
    data, features, decisions, mask = _example()
    features["windows"][-1].update(start_seconds=109, stop_seconds=110)
    figures = make_eeg_figures(data, features, decisions, mask)
    try:
        heatmap = next(ax for ax in figures["02_features_review"].axes if ax.get_gid() == "features:rms")
        last_bounds = heatmap.collections[-1].get_coordinates()[:, :, 0]
        np.testing.assert_array_equal(last_bounds, [[109, 110]] * 3)
        assert heatmap.collections[-2].get_coordinates()[0, -1, 0] == 108
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_figures_keep_data_and_move_review_text_to_report():
    data, features, decisions, mask = _example()
    base_review = decisions["review_queue"][0]
    kinds = ["eog", "ecg", "emg", "line_50", "line_60", "amplitude", "high_frequency"]
    decisions["review_queue"] = [dict(base_review, kind=kind) for kind in kinds]
    decisions["summary"]["review_count"] = len(kinds)
    features["window_features"]["ecg_correlation"] = [[-.8, .2, .1, 0, 0], [.1] * 5]
    features["reference_status"] = {"ecg": {"status": "provided"}}
    features["reference_channels"] = {"ecg": {"raw_name": "ECG reference"}}
    figures = make_eeg_figures(data, features, decisions, mask)
    try:
        fig = figures["02_features_review"]
        axes = {ax.get_gid(): ax for ax in fig.axes if ax.get_gid()}
        assert "features:review_queue" not in axes
        assert "features:reference_sources" not in axes
        assert all(not ax.tables and ax.axison for figure in figures.values() for ax in figure.axes)
        assert len(figures["01_initial"].axes) == len(features["channel_ids"])
        assert len(figures["03_auto_excluded"].axes) == len(features["channel_ids"])
        psd = axes["features:psd"].lines[0]
        np.testing.assert_array_equal(psd.get_xdata(), features["frequencies_hz"])
        np.testing.assert_allclose(psd.get_ydata(), 10*np.log10(np.array(features["psd_v2_hz"][0])*1e12))
        refs = axes["features:references"].images[0].get_array()
        assert refs[0, 0] == .8
        assert refs[1, 0] == .1
        assert any("100–102 s" in text.get_text() for text in axes["features:references"].texts)
        # Local spectrum panels show the measured PSD, rather than threshold-derived peak summaries.
        local = axes["features:line_noise"].lines[0]
        np.testing.assert_array_equal(local.get_xdata(), [50])
        np.testing.assert_allclose(local.get_ydata(), [-20])
        local_alpha = axes["features:alpha"].lines[0]
        np.testing.assert_array_equal(local_alpha.get_xdata(), [10, 30])
        np.testing.assert_allclose(local_alpha.get_ydata(), 10*np.log10([200, .5]))
        patch = axes["features:rms"].patches[0]
        assert patch.get_x() == 102
        assert patch.get_width() == 3
        assert patch.get_y() == pytest.approx(1 - .42)
    finally:
        for figure in figures.values():
            plt.close(figure)
