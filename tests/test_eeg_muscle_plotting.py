"""Figures preserve native values, absolute clock, candidate extent and eligibility."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.plotting.muscle import make_muscle_figure
from brainelecqc.signals import describe_artifact
from tests.edf_fixtures import tal, write_edf_bdf
from tests.test_eeg_muscle import signals
from tests.test_eeg_pipeline import _import, _source


@pytest.fixture
def burst_result(tmp_path):
    data, emg, rate = signals(rate=160)
    channels = []
    for label, row in zip(("EEG Temporal", "EEG Central", "EMG chin"), [*data, emg]):
        values = np.rint(row*1e6*65535/1600-.5).astype(int)
        channels.append({"label": label, "records": values.reshape(40, rate).tolist(), "unit": "uV",
                         "physical_min": -800, "physical_max": 800})
    source = write_edf_bdf(tmp_path/"burst.edf", channels, reserved="EDF+D",
                          annotation_records=[tal(90.25+s) for s in range(40)])
    _, artifact = _import(source)
    segment = describe_artifact(artifact)["streams"][0]["segments"][0]
    return run_eeg_qc(artifact, segment_id=segment["segment_id"], start_seconds=90.25,
                      duration_seconds=40, eeg_indices=[1, 0], reference_indices={"emg": 2})


def test_raw_eeg_emg_score_values_and_discontinuous_clock_are_exact(burst_result):
    result = burst_result
    original = result["raw_data_volts"].copy()
    emg = result["references"]["emg"].copy()
    figure, display = make_muscle_figure(result, channel_index=1, duration_seconds=8)
    try:
        f = result["features"]
        a, b = display["start_sample"]-f["start_sample"], display["stop_sample"]-f["start_sample"]
        t = f["start_seconds"]+np.arange(a,b)/f["sampling_rate_hz"]
        assert len(figure.axes) == 3
        assert display["channel_name"] == "EEG Temporal"
        assert display["start_sample"]/f["sampling_rate_hz"] != display["start_seconds"]
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), t)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_xdata(), t)
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), original[1,a:b]*1e6)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), emg[a:b]*1e6)
        metrics = result["method_results"]["muscle"]["sample_metrics"]
        np.testing.assert_array_equal(figure.axes[2].lines[0].get_ydata(), metrics["z_score"][1][a:b])
        np.testing.assert_array_equal(figure.axes[2].lines[1].get_ydata(), metrics["reference_z_score"][a:b])
        assert all(axis.get_xlim() == figure.axes[0].get_xlim() for axis in figure.axes)
        assert figure.axes[0].get_position().y0 > figure.axes[1].get_position().y0
        np.testing.assert_array_equal(result["raw_data_volts"], original)
        np.testing.assert_array_equal(result["references"]["emg"], emg)
    finally:
        plt.close(figure)


def test_visible_candidates_correspond_to_shading_and_saved_sample_range(burst_result):
    figure, display = make_muscle_figure(burst_result, channel_index=1, duration_seconds=.5)
    try:
        visible = display["visible_candidate_intervals"]
        assert visible
        for item in visible:
            assert item["start_sample"] < display["stop_sample"]
            assert item["stop_sample"] > display["start_sample"]
        patches = figure.axes[0].patches
        assert len(patches) >= len(visible)
        boundaries = [(p.get_x(),p.get_x()+p.get_width()) for p in patches]
        for item in visible:
            start = max(item["start_seconds"], display["start_seconds"])
            stop = min(item["stop_seconds"], display["stop_seconds"])
            assert item["display_start_seconds"] == start
            assert item["display_stop_seconds"] == stop
            assert any(np.allclose([a,b],[start,stop],rtol=0,atol=1e-9) for a,b in boundaries)
    finally:
        plt.close(figure)


def test_uncomputed_short_record_is_labelled_pending_not_healthy(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact, duration_seconds=2)
    figure, _ = make_muscle_figure(result)
    try:
        assert "检测条件不足" in figure._suptitle.get_text()
        assert "未检出" not in figure._suptitle.get_text()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("option", [{"channel_index": True}, {"channel_index": 1.2},
    {"duration_seconds": float("nan")}, {"duration_seconds": float("inf")}, {"duration_seconds": True}])
def test_plot_rejects_ambiguous_channel_or_time_inputs(tmp_path, option):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact, duration_seconds=2)
    with pytest.raises(ValueError):
        make_muscle_figure(result, **option)


def test_reference_and_candidate_clock_changes_fail_identity_before_plot(burst_result):
    changed = copy.deepcopy(burst_result)
    changed["references"]["emg"][0] += 1e-6
    with pytest.raises(ValueError, match="reference_fingerprint"):
        make_muscle_figure(changed)
    changed = copy.deepcopy(burst_result)
    changed["method_results"]["muscle"]["candidates"][0]["start_seconds"] += .5
    with pytest.raises(ValueError, match="clock"):
        make_muscle_figure(changed)
