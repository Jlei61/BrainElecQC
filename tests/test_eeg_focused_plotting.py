"""Figure contracts preserve every channel, physical samples and event clocks."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.plotting.focused import (make_all_channel_psd_figure, make_focused_bad_channel_figure,
                                        make_line_noise_summary_figure, make_aligned_event_figure, _psd_highlights,
                                        _channel_colors, _event_cluster_window, _largest_eeg_event_change, _labels)
from brainelecqc.plotting.eeg import _style
from brainelecqc.signals import describe_artifact
from tests.test_eeg_pipeline import _import, _source
from tests.edf_fixtures import write_edf_bdf


@pytest.fixture
def result(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    return run_eeg_qc(artifact, eeg_indices=[2, 0])


def test_all_channel_psd_keeps_every_selected_channel_and_voltage_units(result):
    fig, metadata = make_all_channel_psd_figure(result)
    try:
        psd = result["method_results"]["spectral"]["psd"]
        f = np.asarray(psd["frequencies_hz"])
        mask = (f >= 2) & (f <= 120)
        assert len(fig.axes[0].lines) == len(result["raw_data_volts"])
        for line in fig.axes[0].lines:
            ci = int(line.get_gid().split(":")[-1])
            np.testing.assert_array_equal(line.get_xdata(), f[mask])
            expected = np.asarray(psd["power_v2_hz"])[ci, mask]*1e12
            np.testing.assert_array_equal(line.get_ydata(), np.where(expected > 0, expected, np.nan))
        assert metadata["channel_names"] == ["EEG Cz", "EEG Oz"]
        assert metadata["psd"]["frequency_range_hz"] == [2, 120]
    finally:
        plt.close(fig)


def test_psd_highlights_prioritize_noise_over_recordwide_continuity_candidates():
    result = {"method_results": {"bad_channels": {"channels": [
        {"hf_z": 20 if i == 9 else i, "low_correlation_fraction": 0, "amplitude_z": 0} for i in range(10)],
        "candidates": [{"kind": "high_frequency_noise", "channel_index": 9},
                       {"kind": "low_correlation", "channel_index": 8}]},
        "continuity": {"candidates": [{"channel_index": i, "start_sample": 0, "stop_sample": 100-i} for i in range(10)]}}}
    highlighted = _psd_highlights(result)
    assert len(highlighted) == 6
    assert highlighted[:2] == [9, 8]
    assert highlighted[2:] == [0, 1, 2, 3]
    colors = _channel_colors(result)
    assert len({colors[i] for i in highlighted}) == len(highlighted)


def test_bad_channel_figure_contains_only_psd_and_shared_scale_original_waveforms(result):
    fig, metadata = make_focused_bad_channel_figure(result, [1, 0], start_seconds=2, duration_seconds=3)
    try:
        axes = {a.get_gid(): a for a in fig.axes}
        assert set(axes) == {"bad:psd", "bad:raw:1", "bad:raw:0"}
        assert axes["bad:raw:0"].get_ylim() == axes["bad:raw:1"].get_ylim()
        for i in (1, 0):
            line = axes[f"bad:raw:{i}"].lines[0]
            np.testing.assert_array_equal(line.get_ydata(), result["raw_data_volts"][i, 512:1280]*1e6)
            np.testing.assert_array_equal(line.get_xdata(), np.arange(512, 1280)/256)
        assert metadata["waveform"]["start_sample"] == 512
        assert metadata["psd"]["stop_seconds"] == 8
    finally:
        plt.close(fig)


def test_all_automatic_bad_channels_enter_plan_and_share_red_family(result, monkeypatch):
    from brainelecqc.eeg import scoring
    from brainelecqc.plotting.paper_eeg import build_focus_plan
    from brainelecqc.plotting.focused import BAD_COLORS
    f, bad = result["features"], result["method_results"]["bad_channels"]
    for i in range(2):
        bad["candidates"].append({"channel_id": f["channel_ids"][i], "channel_index": i,
            "kind": "high_frequency_noise", **{k: f[k] for k in
                ("start_sample", "stop_sample", "start_seconds", "stop_seconds")}})
    monkeypatch.setattr(scoring, "compute_quality_score", lambda r: {"rows": [{"dimension_id": "channels",
        "score_metrics": {"automatic_bad_channel_ids": f["channel_ids"], "unlocalized_channel_ids": []}}]})
    page = build_focus_plan(result)["bad_channel_pages"][0]
    assert set(page["candidate_channel_indices"]) == {0, 1}
    fig, metadata = make_focused_bad_channel_figure(result, page["channel_indices"],
        start_seconds=page["start_seconds"], duration_seconds=page["duration_seconds"])
    try:
        assert len(set(metadata["channel_colors"].values())) == 2
        assert set(metadata["channel_colors"].values()) <= set(BAD_COLORS)
    finally:
        plt.close(fig)


def test_continuity_candidates_highlight_both_psd_and_exact_bad_waveform_interval(tmp_path):
    source, _, _ = _source(tmp_path, clipped=True)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    all_fig, all_meta = make_all_channel_psd_figure(result)
    fig, metadata = make_focused_bad_channel_figure(result, [0, 1], start_seconds=2, duration_seconds=2)
    try:
        assert 0 in all_meta["highlighted_channel_indices"]
        raw = next(a for a in fig.axes if a.get_gid() == "bad:raw:0")
        assert len(raw.patches) == 1
        assert raw.patches[0].get_x() == 600/256
        assert raw.patches[0].get_width() == 50/256
        assert metadata["displayed_continuity_candidates"][0]["stop_sample"] == 650
    finally:
        plt.close(fig)
        plt.close(all_fig)


def test_line_noise_heatmaps_use_all_channels_and_identical_color_scale(result):
    fig, metadata = make_line_noise_summary_figure(result)
    try:
        axes = {a.get_gid(): a for a in fig.axes if a.get_gid()}
        assert set(axes) == {"line:50", "line:60"}
        assert axes["line:50"].collections[0].get_clim() == axes["line:60"].collections[0].get_clim()
        for hz in (50, 60):
            ax = axes[f"line:{hz}"]
            assert [t.get_text() for t in ax.get_yticklabels()] == result["features"]["channel_names"]
            values = ax.collections[0].get_array()
            assert values.shape[0] == 2
            for row in result["method_results"]["spectral"]["line_noise"]["observations"]:
                if row["nominal_hz"] == hz and row["status"] == "computed":
                    assert values[row["channel_index"], row["window_index"]] == pytest.approx(-np.log10(max(row["p_corrected"], 1e-300)))
        assert metadata["color_threshold"] == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["ecg", "eog"])
def test_event_pairs_have_eeg_above_auxiliary_with_exact_raw_values_and_original_clock(result, kind):
    fig, metadata = make_aligned_event_figure(result, kind)
    try:
        axes = {a.get_gid(): a for a in fig.axes}
        for scope in metadata["rendered_scopes"]:
            eeg, aux = axes[f"event:{kind}:{scope}:eeg"], axes[f"event:{kind}:{scope}:{kind}"]
            assert eeg.get_position().y0 > aux.get_position().y0
            assert eeg.get_shared_x_axes().joined(eeg, aux)
            a, b = metadata[scope]["start_sample"], metadata[scope]["stop_sample"]
            np.testing.assert_array_equal(eeg.lines[0].get_ydata(), result["raw_data_volts"][metadata["channel_index"], a:b]*1e6)
            np.testing.assert_array_equal(aux.lines[0].get_ydata(), result["references"][kind][a:b]*1e6)
            np.testing.assert_array_equal(eeg.lines[0].get_xdata(), aux.lines[0].get_xdata())
            assert metadata["display_channel_name"] in eeg.get_ylabel()
            assert result["method_results"]["events"]["by_kind"][kind]["source_reference"]["raw_name"] in aux.get_ylabel()
            assert "µV" in eeg.get_ylabel() and "µV" in aux.get_ylabel()
            assert not eeg.texts and not aux.texts
        assert metadata["display_status"] == "pending"
        assert len(fig.axes) == 2
        assert metadata["rendered_scopes"] == ["waveform"]
        assert metadata["overview"] is None
        assert all(not ax.patches for ax in fig.axes)
        assert "事件锁时" in fig._suptitle.get_text()
        assert "待评" not in fig._suptitle.get_text()
        if kind == "eog":
            assert metadata["waveform"]["start_seconds"] == 0
            assert metadata["waveform"]["stop_seconds"] == 8
            if any(e["complete_epoch"] for e in result["method_results"]["events"]["by_kind"][kind]["event_table"]):
                assert metadata["event_selection"] == "largest_displayed_eeg_peak_to_peak_in_complete_eog_epoch"
                measured = metadata["event_selection_measurements"]
                assert measured["selected"]["event_id"] == metadata["event"]["event_id"]
                assert measured["selected"]["eeg_peak_to_peak_uv"] == max(row["eeg_peak_to_peak_uv"] for row in measured["evaluated_events"])
        elif kind == "ecg":
            assert metadata["waveform"]["stop_seconds"]-metadata["waveform"]["start_seconds"] == 6
    finally:
        plt.close(fig)


def test_bad_comparison_preserves_discontinuous_record_clock(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False, discontinuous=True)
    _, artifact = _import(source)
    segment = describe_artifact(artifact)["streams"][0]["segments"][1]
    result = run_eeg_qc(artifact, segment_id=segment["segment_id"], start_seconds=10.5, duration_seconds=2)
    fig, metadata = make_focused_bad_channel_figure(result, [0, 1], start_seconds=11, duration_seconds=1)
    try:
        assert metadata["waveform"]["start_sample"] == 5*256
        assert metadata["waveform"]["start_seconds"] == 11
        assert fig.axes[1].lines[0].get_xdata()[0] == 11
    finally:
        plt.close(fig)


def test_eog_figure_keeps_local_large_change_in_one_pair(tmp_path):
    source, _, _ = _source(tmp_path, clipped=True)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    fig, metadata = make_aligned_event_figure(result, "eog", channel_index=0)
    try:
        assert metadata["waveform"]["start_sample"] <= 600 < metadata["waveform"]["stop_sample"]
        axes = {a.get_gid(): a for a in fig.axes}
        scope = metadata["rendered_scopes"][0]
        eeg, aux = axes[f"event:eog:{scope}:eeg"], axes[f"event:eog:{scope}:eog"]
        assert eeg.get_shared_x_axes().joined(eeg, aux)
        np.testing.assert_array_equal(eeg.lines[0].get_xdata(), aux.lines[0].get_xdata())
        assert len(axes) == 2*len(metadata["rendered_scopes"])
        if metadata["event"] is None:
            assert metadata["t0"] is None
            assert "事件未定位" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def _clock_result(start_seconds=0, duration_seconds=300):
    return {"features": {"sampling_rate_hz": 64, "start_sample": 90000,
                         "start_seconds": start_seconds, "stop_seconds": start_seconds+duration_seconds},
            "raw_data_volts": np.zeros((1, 64*duration_seconds))}


def test_event_cluster_covers_related_burst_and_excludes_distant_events():
    events = [{"event_id": str(t), "time_seconds": t} for t in [33, 34, 42, 43, 44, 45, 100]]
    interval, selection = _event_cluster_window(_clock_result(), events, events[1])
    assert interval[3]["start_seconds"] == 31
    assert interval[3]["stop_seconds"] == 47
    assert interval[3]["start_sample"] == 90000+31*64
    assert selection["cluster_event_ids"] == ["33", "34", "42", "43", "44", "45"]
    assert selection["anchor_event_id"] == "34"


def test_dense_event_cluster_stays_local_instead_of_expanding_to_whole_record():
    events = [{"event_id": str(t), "time_seconds": t} for t in range(100)]
    interval, selection = _event_cluster_window(_clock_result(), events, events[50])
    assert interval[3]["stop_seconds"]-interval[3]["start_seconds"] <= 24
    assert interval[3]["start_seconds"] <= 50 < interval[3]["stop_seconds"]
    assert len(selection["cluster_event_ids"]) <= 21
    assert selection["cluster_start_seconds"] <= 45
    assert selection["cluster_stop_seconds"] >= 55


@pytest.mark.parametrize("times,anchor,expected", [
    ([100.2, 101.4, 102.6, 103.8, 108], 1, [100.2, 101.4, 102.6]),
    ([101, 130, 131], 0, [101]),
    ([198, 199, 199.9], 2, [198, 199, 199.9]),
])
def test_detail_shows_neighboring_events_without_bridging_long_empty_gaps(times, anchor, expected):
    events = [{"event_id": str(t), "time_seconds": t} for t in times]
    result = _clock_result(100, 100)
    interval, selection = _event_cluster_window(
        result, events, events[anchor], max_gap_seconds=3, max_span_seconds=6,
        context_seconds=.5, minimum_seconds=2, max_events=3)
    assert selection["cluster_event_ids"] == [str(t) for t in expected]
    clock = interval[3]
    assert 100 <= clock["start_seconds"] <= min(expected) <= max(expected) < clock["stop_seconds"] <= 200
    assert clock["stop_seconds"]-clock["start_seconds"] <= 7
    assert clock["start_sample"] == 90000+round((clock["start_seconds"]-100)*64)
    reversed_interval, reversed_selection = _event_cluster_window(
        result, list(reversed(events)), events[anchor], max_gap_seconds=3, max_span_seconds=6,
        context_seconds=.5, minimum_seconds=2, max_events=3)
    assert reversed_selection == selection
    assert reversed_interval[3] == clock


def test_eog_display_anchor_uses_eeg_change_not_largest_auxiliary_amplitude():
    result = _clock_result(start_seconds=100, duration_seconds=5)
    result["features"].update(channel_ids=["eeg-0"], channel_names=["EEG O2-M1"])
    raw = result["raw_data_volts"][0]
    raw[64:128] = np.linspace(-1e-6, 1e-6, 64)
    raw[128:192] = np.linspace(-40e-6, 60e-6, 64)
    raw[192:256] = np.linspace(-500e-6, 500e-6, 64)
    events = [{"event_id": f"eog:{i}", "sample": 90000+i*64+32,
               "time_seconds": 100+i+.5, "epoch_start_sample": 90000+i*64,
               "epoch_stop_sample": 90000+(i+1)*64, "complete_epoch": i != 3,
               "raw_reference_volts_at_event": 800e-6 if i == 1 else 20e-6}
              for i in (1, 2, 3)]
    event, metadata = _largest_eeg_event_change(result, events, 0)
    assert max(events, key=lambda row: row["raw_reference_volts_at_event"])["event_id"] == "eog:1"
    assert event["event_id"] == "eog:2"
    assert metadata["purpose"] == "abnormality_display_selection"
    assert metadata["selected"]["eeg_peak_to_peak_uv"] == pytest.approx(100)
    assert metadata["selected"]["start_sample"] == 90128
    assert metadata["selected"]["start_seconds"] == 102
    assert metadata["selected"]["stop_seconds"] == 103
    assert metadata["raw_channel_name"] == "EEG O2-M1"
    assert metadata["excluded_events"] == [{"event_id": "eog:3", "reason": "incomplete_epoch"}]
    raw[64:128] = raw[128:192]
    event, _ = _largest_eeg_event_change(result, list(reversed(events)), 0)
    assert event["event_id"] == "eog:1"
    raw[64:192] = np.nan
    event, metadata = _largest_eeg_event_change(result, events, 0)
    assert event is None
    assert metadata["selected"] is None
    assert len(metadata["excluded_events"]) == 3


@pytest.mark.parametrize("event_time", [100, 199.984375])
def test_cluster_clamps_recording_edges_and_preserves_discontinuous_sample_origin(event_time):
    event = {"event_id": "one", "time_seconds": event_time}
    interval, _ = _event_cluster_window(_clock_result(100, 100), [event], event)
    clock = interval[3]
    assert 100 <= clock["start_seconds"] <= event_time < clock["stop_seconds"] <= 200
    assert clock["stop_seconds"]-clock["start_seconds"] == 8
    assert clock["start_sample"] == 90000+round((clock["start_seconds"]-100)*64)


def test_reference_display_aliases_keep_raw_names_and_distinguish_collisions():
    names = ["EEG C4-M1", "EEG C4-M2", "EEG Cz"]
    result = {"features": {"channel_names": names.copy()}}
    assert _labels(result) == ["EEG C4-Ref [row 0]", "EEG C4-Ref [row 1]", "EEG Cz"]
    assert result["features"]["channel_names"] == names


@pytest.mark.parametrize("coupled", [False, True])
def test_event_title_distinguishes_repeated_coupling_and_keeps_real_event_identity(tmp_path, coupled):
    fs, seconds = 256, 40
    t = np.arange(fs*seconds)/fs
    ecg = sum(15000*np.exp(-.5*((t-event)/.015)**2) for event in np.arange(1, 39, 1.2))
    eeg = np.random.default_rng(20261027).normal(0, 300, len(t))+(ecg*.15 if coupled else 0)
    channels = [{"label": label, "records": np.rint(values).astype(int).reshape(seconds, fs).tolist(),
                 "unit": "uV", "physical_min": -800, "physical_max": 800}
                for label, values in (("EEG C4-M1", eeg), ("ECG", ecg))]
    source = write_edf_bdf(tmp_path/"event-display.edf", channels)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    fig, metadata = make_aligned_event_figure(result, "ecg")
    try:
        assert metadata["event_score"]["metrics"]["nonoverlapping_event_count"] >= 20
        assert metadata["display_status"] == ("coupled_candidate" if coupled else "no_material_repeatable_coupling")
        if coupled:
            assert metadata["event_selection"] == "score_representative_event_for_displayed_channel"
            assert metadata["event_selection_measurements"] is None
        assert ("检出重复同步" if coupled else "未见重复同步") in fig._suptitle.get_text()
        assert "t₀ =" in fig._suptitle.get_text()
        assert metadata["channel_name"] == "EEG C4-M1"
        assert metadata["display_channel_name"] == "EEG C4-Ref"
        assert metadata["t0"]["sample"] == metadata["event"]["sample"]
        assert metadata["t0"]["time_seconds"] == metadata["t0"]["sample"]/fs
        assert len(metadata["displayed_event_ids"]["waveform"]) >= 3
        assert metadata["t0"]["event_id"] in metadata["displayed_event_ids"]["waveform"]
        assert metadata["layout"] == "consecutive_events_pair"
        with matplotlib.rc_context(_style()):
            fig.canvas.draw()
        axes = {ax.get_gid(): ax for ax in fig.axes}
        assert len(axes) == 2
        assert axes["event:ecg:waveform:eeg"].get_position().bounds[0] == axes["event:ecg:waveform:ecg"].get_position().bounds[0]
        assert all(not ax.patches for ax in fig.axes)
        assert all(not ax.get_title(loc="left").startswith(("A ", "B ")) for ax in fig.axes)
        for ax in fig.axes:
            marker = next(line for line in ax.lines if line.get_gid() == f"event-marker:{metadata['t0']['event_id']}")
            np.testing.assert_array_equal(marker.get_xdata(), [metadata["t0"]["time_seconds"]]*2)
        assert "EEG C4-Ref" in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)


def test_one_detected_event_uses_one_square_figure_with_original_traces(tmp_path):
    fs, seconds = 256, 20
    t = np.arange(fs*seconds)/fs
    eog = 12000*np.exp(-.5*((t-10)/.06)**2)
    eeg = np.random.default_rng(121).normal(0, 50, len(t))+.2*eog
    channels = [{"label": label, "records": np.rint(values).astype(int).reshape(seconds, fs).tolist(),
                 "unit": "uV", "physical_min": -800, "physical_max": 800}
                for label, values in (("EEG F4-M1", eeg), ("EOG", eog))]
    source = write_edf_bdf(tmp_path/"single-event.edf", channels)
    _, artifact = _import(source)
    # Explicit threshold isolates the pulse from the detector filter's side lobes.
    result = run_eeg_qc(artifact, method_config={"events": {"eog_threshold_volts": 100e-6}})
    fig, metadata = make_aligned_event_figure(result, "eog")
    try:
        assert len(result["method_results"]["events"]["by_kind"]["eog"]["event_table"]) == 1
        assert metadata["layout"] == "single_event_square"
        assert metadata["rendered_scopes"] == ["waveform"]
        assert metadata["overview"] is None
        assert len(fig.axes) == 2
        assert fig.get_figwidth() == fig.get_figheight()
        assert metadata["displayed_event_ids"]["waveform"] == [metadata["t0"]["event_id"]]
        a, b = metadata["waveform"]["start_sample"], metadata["waveform"]["stop_sample"]
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), result["raw_data_volts"][0, a:b]*1e6)
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), result["references"]["eog"][a:b]*1e6)
    finally:
        plt.close(fig)


def test_missing_auxiliary_has_no_fabricated_event_time(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact, reference_indices={})
    fig, metadata = make_aligned_event_figure(result, "eog")
    try:
        assert metadata["display_status"] == "pending_reference"
        assert metadata["event"] is None and metadata["t0"] is None
        assert any("缺少辅助导联" in text.get_text() for text in fig.axes[0].texts)
        assert not fig.axes[0].lines
    finally:
        plt.close(fig)


@pytest.mark.parametrize("indices,start,duration", [([0, 0], 1, 2), ([True], 1, 2), ([2], 1, 2), ([0], -1, 2), ([0], 7, 2)])
def test_invalid_selection_and_clock_fail(result, indices, start, duration):
    with pytest.raises(ValueError):
        make_focused_bad_channel_figure(result, indices, start_seconds=start, duration_seconds=duration)


def test_mixed_reference_comparison_is_rejected(result):
    result = copy.deepcopy(result)
    result["method_results"]["bad_channels"]["channels"][1]["group_id"] = "other"
    with pytest.raises(ValueError, match="one declared comparison group"):
        make_focused_bad_channel_figure(result, [0, 1], start_seconds=1)


@pytest.mark.parametrize("function", [make_all_channel_psd_figure, make_line_noise_summary_figure,
                                    lambda r: make_aligned_event_figure(r, "ecg")])
def test_figures_reject_unbound_raw_samples(result, function):
    result = copy.deepcopy(result)
    result["raw_data_volts"][0, 0] += 1e-6
    with pytest.raises(ValueError):
        function(result)
