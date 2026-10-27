"""Detailed bad-channel figures preserve values, groups and discontinuous clocks."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.plotting.bad_channel_review import make_bad_channel_review_figure
from brainelecqc.signals import describe_artifact
from tests.test_eeg_pipeline import _import, _source


@pytest.fixture
def result(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False, discontinuous=True)
    _, artifact = _import(source)
    segment = describe_artifact(artifact)["streams"][0]["segments"][1]
    return run_eeg_qc(artifact, segment_id=segment["segment_id"], start_seconds=10.5,
                      duration_seconds=2, eeg_indices=[2, 0])


def test_detail_figure_retains_voltage_clock_order_and_threshold_gating(result):
    before = result["raw_data_volts"].copy()
    figure, display = make_bad_channel_review_figure(result, [1, 0], start_seconds=11, duration_seconds=1)
    try:
        axes = {a.get_gid(): a for a in figure.axes}
        fs = result["features"]["sampling_rate_hz"]
        left, right = round(.5*fs), round(1.5*fs)
        for i in (1, 0):
            np.testing.assert_array_equal(axes[f"bad:raw:{i}"].lines[0].get_ydata(), before[i, left:right]*1e6)
            np.testing.assert_array_equal(axes[f"bad:raw:{i}"].lines[0].get_xdata(), 11+np.arange(right-left)/fs)
        assert display["waveform"]["start_sample"] == result["features"]["start_sample"]+left
        assert display["waveform"]["start_sample"]/fs != display["waveform"]["start_seconds"]
        assert axes["bad:raw:0"].get_ylim() == axes["bad:raw:1"].get_ylim()
        assert [c["channel_name"] for c in display["selected_channels"]] == ["EEG Oz", "EEG Cz"]
        assert not display["thresholds_active"]
        assert not axes["bad:high_frequency_z"].lines
        np.testing.assert_array_equal(result["raw_data_volts"], before)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("indices,start,duration", [([0, 0], 11, 1), ([2], 11, 1),
                                                   ([True], 11, 1), ([0], 9, 1),
                                                   ([0], 12, 1), ([0], 11, 0)])
def test_bad_channel_plot_rejects_ambiguous_selection_or_outside_clock(result, indices, start, duration):
    with pytest.raises(ValueError):
        make_bad_channel_review_figure(result, indices, start_seconds=start, duration_seconds=duration)


def test_bad_channel_plot_rejects_mixed_reference_groups(result):
    result = copy.deepcopy(result)
    bad = result["method_results"]["bad_channels"]
    bad["channels"][1]["group_id"] = "another_reference"
    with pytest.raises(ValueError, match="one declared comparison group"):
        make_bad_channel_review_figure(result, [0, 1], start_seconds=11, duration_seconds=1)
