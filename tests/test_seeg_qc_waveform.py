"""Contextual SEEG waveform events, including counterexamples and censoring.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import json

import numpy as np
import pytest

from brainelecqc.seeg.seeg import scan_seeg
from tests.test_seeg_qc_metrics import fixture


def baseline(seconds=12, fs=500):
    return np.random.default_rng(73).normal(0, 8e-6, int(seconds * fs))


def scan(values, seconds=5, segments=None, config=None):
    catalog, read, calls = fixture(values, segments=segments)
    return scan_seeg(catalog, read, config={"window_seconds": seconds, **(config or {})}), calls


def event_signature(report):
    return [(e["start_sample"], e["stop_sample"], e["recovery_status"],
             [(s["kind"], s["start_sample"], s["stop_sample"]) for s in e["stages"]])
            for e in report["waveform_events"]]


def test_step_plateau_recovery_is_one_event_across_qc_windows():
    x = baseline()
    x[2400:4100] += .001
    report, _ = scan(x)
    events = report["waveform_events"]
    assert len(events) == 1
    event = events[0]
    assert event["start_sample"] == 2400
    assert event["stop_sample"] >= 4250
    assert event["recovery_status"] == "confirmed"
    assert {s["kind"] for s in event["stages"]} >= {"jump", "plateau", "recovery"}
    assert event["score_effect"] == "none"
    assert len([w for w in report["windows"] if event["event_id"] in w["waveform_event_ids"]]) == 2
    assert all(w["score"] == 100 for w in report["windows"])
    smaller, _ = scan(x, seconds=.5)
    assert event_signature(smaller) == event_signature(report)
    json.dumps(report, allow_nan=False)


def test_ramp_into_plateau_has_drift_and_recovery_stages():
    x = baseline()
    x[1500:2000] += np.linspace(0, .0015, 500, endpoint=False)
    x[2000:3250] += .0015
    x[3250:3750] += np.linspace(.0015, 0, 500, endpoint=False)
    report, _ = scan(x)
    assert len(report["waveform_events"]) == 1
    event = report["waveform_events"][0]
    assert event["recovery_status"] == "confirmed"
    assert {s["kind"] for s in event["stages"]} >= {"drift", "plateau", "recovery"}
    assert event["start_sample"] <= 1600
    assert event["stop_sample"] >= 3750


def test_no_return_before_tail_is_right_censored_not_fake_recovery():
    x = baseline(seconds=7)
    x[2400:] += .001
    report, _ = scan(x)
    event = report["waveform_events"][0]
    assert event["stop_sample"] == len(x)
    assert event["recovery_status"] == "right_censored"
    assert event["termination_reason"] == "segment_end"
    assert not any(s["kind"] == "recovery" for s in event["stages"])


def test_gap_and_unscanned_ranges_break_context():
    x = baseline(seconds=10)
    x[1500:2500] += .001
    segments = [{"segment_id": "a", "start_sample": 0, "stop_sample": 2500, "recording_start_seconds": 0},
                {"segment_id": "b", "start_sample": 2500, "stop_sample": 5000, "recording_start_seconds": 20}]
    report, _ = scan(x, segments=segments)
    assert len(report["waveform_events"]) == 1
    event = report["waveform_events"][0]
    assert event["segment_id"] == "a" and event["stop_sample"] == 2500
    assert event["recovery_status"] == "right_censored"
    selected, _ = scan(x, config={"ranges": [{"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 2000},
                                            {"stream_id": "s", "segment_id": "p", "start_sample": 2300, "stop_sample": 5000}]})
    assert selected["waveform_events"][0]["stop_sample"] == 2000
    assert selected["waveform_events"][0]["termination_reason"] == "range_end"


def test_adjacent_explicit_ranges_preserve_same_event_context():
    x = baseline()
    x[2400:4100] += .001
    report, _ = scan(x)
    adjacent, _ = scan(x, config={"ranges": [{"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 2500},
                                            {"stream_id": "s", "segment_id": "p", "start_sample": 2500, "stop_sample": len(x)}]})
    assert event_signature(adjacent) == event_signature(report)


def test_nonfinite_breaks_continuity_and_does_not_bridge_recovery():
    x = baseline()
    x[1500:3500] += .001
    x[2500:2550] = np.nan
    report, _ = scan(x)
    first = report["waveform_events"][0]
    assert first["stop_sample"] == 2500
    assert first["termination_reason"] == "nonfinite_gap"
    assert first["recovery_status"] == "right_censored"


@pytest.mark.parametrize("kind", ["single_spike", "oscillation", "slow_wave", "constant"])
def test_physiological_counterexamples_do_not_emit_joint_artifact(kind):
    fs = 500
    x = baseline(seconds=20)
    t = np.arange(len(x)) / fs
    if kind == "single_spike":
        x[2400:2403] += [0.002, -.003, .001]
    elif kind == "oscillation":
        x += .0004 * np.sin(2 * np.pi * 12 * t)
    elif kind == "slow_wave":
        x += .0006 * np.sin(2 * np.pi * .2 * t)
    else:
        x[:] = .001
    report, _ = scan(x)
    assert report["waveform_events"] == []


def test_brief_baseline_crossing_does_not_confirm_recovery():
    x = baseline()
    x[1500:] += .001
    x[3000:3020] -= .001
    report, _ = scan(x)
    assert len(report["waveform_events"]) == 1
    assert report["waveform_events"][0]["recovery_status"] == "right_censored"


def test_waveform_resource_budget_is_preflight_not_silent_sampling():
    with pytest.raises(ValueError, match="waveform|Waveform"):
        scan(baseline(), config={"max_waveform_bins": 10})


def test_large_event_residual_offset_does_not_count_as_recovery():
    x = baseline(seconds=20)
    x[1500:2500] += .003
    x[2500:3500] += np.linspace(.003, .0003, 1000, endpoint=False)
    x[3500:] += .0003
    report, _ = scan(x)
    assert len(report["waveform_events"]) == 1
    event = report["waveform_events"][0]
    assert event["recovery_status"] == "right_censored"
    assert event["features"]["recovery_tolerance_v"] < .0001


def test_exact_subsecond_plateau_boundaries_survive_bin_and_qc_edges():
    x = baseline()
    x[2400:3000] += .002
    x[2423:2544] = .002
    x[3000:3500] += np.linspace(.002, 0, 500, endpoint=False)
    report, _ = scan(x)
    exact = [s for e in report["waveform_events"] for s in e["stages"] if s["kind"] == "plateau"
             and s["features"]["method"] == "exact_constant_run"]
    assert any(s["start_sample"] == 2423 and s["stop_sample"] == 2544 for s in exact)


def test_isolated_prior_spike_is_not_joined_to_later_connected_drift():
    x = baseline()
    x[2200:2203] += [.003, -.003, .001]
    x[2500:2530] += np.linspace(0, .0008, 30, endpoint=False)
    x[2530:3500] += .003
    report, _ = scan(x)
    assert len(report["waveform_events"]) == 1
    event = report["waveform_events"][0]
    assert 2500 <= event["start_sample"] < 2530
    assert event["stages"][0]["kind"] == "drift"
    assert all(s["stop_sample"] > s["start_sample"] >= event["start_sample"] for s in event["stages"])
    assert all(s["stop_sample"] <= event["stop_sample"] for s in event["stages"])
