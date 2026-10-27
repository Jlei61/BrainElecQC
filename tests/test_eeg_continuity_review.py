"""Continuity figures preserve source samples and separate observations from decisions."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.plotting.continuity_review import make_continuity_review_figure
from tests.test_eeg_pipeline import _import, _source
from tests.test_eeg_bad_channel_review import result  # discontinuous clock fixture


def test_constant_run_preserves_full_sample_bounds_when_display_crops_it(tmp_path):
    source, _, fs = _source(tmp_path, clipped=True)
    _, artifact = _import(source)
    data = run_eeg_qc(artifact, eeg_indices=[2, 0])
    before, decisions = data["raw_data_volts"].copy(), copy.deepcopy(data["decisions"])
    fig, meta = make_continuity_review_figure(data, [1, 0], start_seconds=610/fs,
                                             duration_seconds=30/fs, min_constant_seconds=.1)
    try:
        assert len(meta["constant_runs"]) == 1
        run = meta["constant_runs"][0]
        assert (run["start_sample"], run["stop_sample"], run["sample_count"]) == (600, 650, 50)
        assert run["channel_name"] == "EEG Oz"
        assert run["value_uv"] == pytest.approx(800)
        assert run["observation"] == "identical_raw_samples"
        assert meta["saved_continuity_candidates"] == []
        axes = {ax.get_gid(): ax for ax in fig.axes}
        for i in (1, 0):
            np.testing.assert_array_equal(axes[f"continuity:raw:{i}"].lines[0].get_ydata(), before[i, 610:640]*1e6)
            np.testing.assert_array_equal(axes[f"continuity:difference:{i}"].lines[0].get_ydata(),
                                           np.diff(before[i, 610:640]*1e6))
        assert axes["continuity:raw:0"].get_ylim() == axes["continuity:raw:1"].get_ylim()
        assert data["decisions"] == decisions
        np.testing.assert_array_equal(data["raw_data_volts"], before)
    finally:
        plt.close(fig)


def test_oscillation_is_not_a_constant_run_and_discontinuous_clock_is_kept(result):
    fig, meta = make_continuity_review_figure(result, [1, 0], start_seconds=11, duration_seconds=1)
    try:
        assert meta["constant_runs"] == []
        fs = result["features"]["sampling_rate_hz"]
        assert meta["display"]["start_sample"] == result["features"]["start_sample"]+round(.5*fs)
        assert meta["display"]["start_seconds"] == 11
        assert meta["display"]["start_sample"]/fs != 11
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), 11+np.arange(round(fs))/fs)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("indices,start,duration,minimum", [([0, 0], 11, 1, .1),
    ([True], 11, 1, .1), ([2], 11, 1, .1), ([0], 9, 1, .1), ([0], 12, 1, .1),
    ([0], 11, 0, .1), ([0], 11, 1, 0), ([0], 11, 1, float("nan"))])
def test_invalid_selection_and_duration_raise(result, indices, start, duration, minimum):
    with pytest.raises(ValueError):
        make_continuity_review_figure(result, indices, start_seconds=start,
                                     duration_seconds=duration, min_constant_seconds=minimum)
