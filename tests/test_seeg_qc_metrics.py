"""SEEG QC counterexamples: candidate evidence must not erase real activity.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import json

import numpy as np
import pytest

from brainelecqc.seeg.seeg import scan_seeg
from brainelecqc.seeg.seeg_channels import resolve_seeg_channels
from brainelecqc.seeg.seeg_scoring import compute_quality_score


def fixture(data, fs=500, names=None, segments=None, units=None):
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    names = names or ["A{}".format(i + 1) for i in range(len(data))]
    channels = [{"channel_id": "c{}".format(i), "raw_name": name,
                 "unit": (units or ["V"] * len(data))[i], "channel_type": "unknown",
                 "quantity": "voltage" if (units or ["V"] * len(data))[i] == "V" else "unknown"}
                for i, name in enumerate(names)]
    segments = segments or [{"segment_id": "p", "start_sample": 0,
                            "stop_sample": data.shape[1], "recording_start_seconds": 0}]
    catalog = {"streams": [{"stream_id": "s", "source_id": "src", "clock_id": "src",
                           "sampling_frequency_hz": fs, "channels": channels,
                           "segments": segments, "sample_count": data.shape[1]}], "sources": []}
    calls = []

    def read(stream, segment, ids, start, stop, representation="physical"):
        calls.append((stream, segment, ids, start, stop))
        return data[[int(c[1:]) for c in ids], start:stop]

    return catalog, read, calls


def test_local_flat_window_does_not_discard_whole_channel_or_other_channels():
    rng = np.random.default_rng(28)
    x = rng.normal(0, 20e-6, (2, 7500))
    x[0, 2500:5000] = 0
    catalog, read, calls = fixture(x)
    report = scan_seeg(catalog, read)
    flat = [w for w in report["windows"] if "flat" in w["flags"]]
    assert [(w["channel_id"], w["start_sample"], w["stop_sample"]) for w in flat] == [("c0", 2500, 5000)]
    assert report["channels"][0]["flagged_seconds"] == 5
    assert report["channels"][1]["flagged_seconds"] == 0
    assert max(stop - start for _, _, _, start, stop in calls) == 2500
    assert report["initial_score"]["total"] == 83


def test_segments_and_partial_tail_keep_real_clock_and_coverage():
    x = np.random.default_rng(2).normal(0, 1e-5, 3100)
    segments = [{"segment_id": "p", "start_sample": 0, "stop_sample": 1500, "recording_start_seconds": 0},
                {"segment_id": "q", "start_sample": 1500, "stop_sample": 3100, "recording_start_seconds": 10}]
    catalog, read, calls = fixture(x, segments=segments)
    report = scan_seeg(catalog, read, config={"window_seconds": 2})
    assert all(not start < 1500 < stop for _, _, _, start, stop in calls)
    assert [w["duration_seconds"] for w in report["windows"]] == [2, 1, 2, 1.2]
    assert report["integrity"]["gaps"][0]["duration_seconds"] == 7
    assert report["coverage"]["scanned_channel_seconds"] == pytest.approx(6.2)
    partial = scan_seeg(catalog, read, config={"ranges": [{"stream_id": "s", "segment_id": "q",
                                                        "start_sample": 1500, "stop_sample": 2000}]})
    assert partial["coverage"]["scanned_fraction"] == pytest.approx(500 / 3100)
    assert partial["coverage"]["scope"] == "selected_ranges"


@pytest.mark.parametrize("frequency", [50, 60])
def test_line_noise_uses_configured_frequency_and_neighbor_background(frequency):
    fs = 500
    t = np.arange(fs * 5) / fs
    x = 1e-4 * np.sin(2 * np.pi * frequency * t) + np.random.default_rng(9).normal(0, 1e-5, len(t))
    catalog, read, _ = fixture(x)
    report = scan_seeg(catalog, read, config={"line_frequency": frequency})
    assert report["windows"][0]["flags"] == ["line_noise"]
    assert report["windows"][0]["metrics"]["line_noise_db"] > 12
    assert report["spectra"][0]["status"] == "computed"


def test_high_frequency_sharp_events_and_absent_alpha_are_not_badness():
    fs = 500
    t = np.arange(fs * 5) / fs
    x = 1e-4 * np.sin(2 * np.pi * 83 * t)
    x[1234:1240] += np.array([0, .001, .003, .001, 0, -.0005])
    catalog, read, _ = fixture(x)
    report = scan_seeg(catalog, read)
    assert report["windows"][0]["flags"] == []
    assert report["initial_score"]["total"] == 100
    assert report["windows"][0]["metrics"]["high_frequency_ratio"] > .3


def test_identity_parser_preserves_zero_padding_primes_and_existing_bipolar():
    names = ["A1", "A01", "A'1", "A1-A2", "A3", "A1A3", "ECG", "Fp1"]
    catalog, _, _ = fixture(np.ones((len(names), 10)), names=names)
    mapped = resolve_seeg_channels(catalog)
    assert [c["raw_name"] for c in mapped] == names
    assert len({c["channel_id"] for c in mapped}) == len(names)
    assert mapped[2]["candidate"]["shaft"] == "A'"
    assert mapped[3]["candidate"]["contact_names"] == ["A1", "A2"]
    assert mapped[3]["measurement_kind"] == "unknown"
    assert mapped[5]["candidate"]["shaft"] is None
    assert mapped[7]["channel_type"] == "unknown"
    assert all(not c.get("bipolar_pairs") for c in mapped)
    assert "possible_alias" in mapped[0]["issues"]
    assert "missing_contact_numbers" in mapped[4]["issues"]


def test_confirmed_mapping_is_explicit_and_cannot_hide_cross_gap_pairing():
    catalog, _, _ = fixture(np.ones((2, 10)), names=["A1", "A3"])
    mapping = [{"channel_id": "c0", "channel_type": "seeg", "status": "confirmed",
                "measurement_kind": "bipolar", "contact_names": ["A1", "A3"], "reference": "A3"}]
    resolved = resolve_seeg_channels(catalog, mapping)
    assert resolved[0]["contact_names"] == ["A1", "A3"]
    assert "nonadjacent_bipolar_contacts" in resolved[0]["issues"]
    assert resolved[0]["status"] == "confirmed"


def test_duplicate_signals_are_only_relation_candidates():
    x = np.random.default_rng(4).normal(0, 1e-5, 2500)
    catalog, read, _ = fixture(np.vstack([x, x]))
    report = scan_seeg(catalog, read)
    assert report["initial_score"]["total"] == 100
    assert any(r["kind"] == "duplicate_signal_candidate" for r in report["relations"])


def test_unknown_units_have_no_voltage_metrics_or_false_full_score():
    catalog, read, _ = fixture(np.zeros(2500), units=["Ohm"])
    report = scan_seeg(catalog, read)
    assert report["windows"][0]["metrics"]["rms_v"] is None
    assert report["windows"][0]["score"] is None
    assert report["initial_score"]["total"] is None
    json.dumps(report, allow_nan=False)


def test_unconfirmed_file_boundary_is_review_evidence_without_deduction():
    x = np.random.default_rng(5).normal(0, 1e-5, 2500)
    x[500:600] = 1e-3
    catalog, read, _ = fixture(x)
    report = scan_seeg(catalog, read)
    assert "boundary_dwell" not in report["windows"][0]["flags"]
    catalog["streams"][0]["channels"][0].update(physical_min=-1000, physical_max=1000, scale_to_si=1e-6,
                                                digital_min=-32768, digital_max=32767)
    original = x.copy()
    report = scan_seeg(catalog, read)
    window = report["windows"][0]
    assert "boundary_dwell" in window["flags"]
    assert "clipping" not in window["flags"]
    assert window["metrics"]["boundary_dwell_fraction"] == pytest.approx(.04)
    assert window["metrics"]["longest_boundary_dwell_seconds"] == pytest.approx(.2)
    assert window["metrics"]["clipping_fraction"] == pytest.approx(.04)
    assert window["metrics"]["adc_range_status"] == "unconfirmed"
    assert window["metrics"]["boundary_dwell_score_effect"] == "none"
    assert window["score"] == report["initial_score"]["total"] == 100
    clipping = next(c for c in report["initial_score"]["components"] if c["flag"] == "clipping")
    assert clipping["evaluated_channel_seconds"] == 0
    np.testing.assert_array_equal(x, original)


def test_boundary_evidence_does_not_suppress_independent_flat_detection():
    catalog, read, _ = fixture(np.full(2500, 1e-3))
    catalog["streams"][0]["channels"][0].update(physical_min=-1000, physical_max=1000, scale_to_si=1e-6,
                                                digital_min=-32768, digital_max=32767)
    window = scan_seeg(catalog, read)["windows"][0]
    assert "boundary_dwell" in window["flags"]
    assert "flat" in window["flags"]
    assert "clipping" not in window["flags"]
    assert window["score"] == 0


def test_nonfinite_and_constant_have_no_spectral_fabrication():
    catalog, read, _ = fixture(np.vstack([np.full(2500, np.nan), np.zeros(2500)]))
    report = scan_seeg(catalog, read)
    assert all(s["status"] == "insufficient_data" for s in report["spectra"])
    assert report["windows"][0]["flags"] == ["nonfinite"]
    assert report["windows"][1]["flags"] == ["flat"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("config", [{"window_seconds": 0}, {"window_seconds": True}, {"line_frequency": 55},
                                     {"unknown_option": 1}, {"ranges": []}, {"max_windows": 0}])
def test_invalid_configuration_is_rejected(config):
    catalog, read, _ = fixture(np.ones(100))
    with pytest.raises(ValueError):
        scan_seeg(catalog, read, config=config)


def test_score_uses_max_penalty_and_review_is_duration_weighted():
    windows = [{"window_id": "a", "duration_seconds": 2, "flags": ["flat", "clipping"],
                "score": 0, "score_eligible": True},
               {"window_id": "b", "duration_seconds": 8, "flags": [], "score": 100, "score_eligible": True}]
    assert compute_quality_score(windows)["total"] == 80
    assert compute_quality_score(windows, {"a": "rejected"})["total"] == 100
    assert compute_quality_score(windows, {"b": "accepted"})["total"] == 0


def test_vendor_prefix_does_not_set_modality_and_auxiliary_is_not_scored():
    rng = np.random.default_rng(41)
    names = ["EEG A1-Ref", "POL A3", "POL C1", "ECG", "TRIG"]
    catalog, read, _ = fixture(rng.normal(0, 1e-5, (5, 2500)), names=names)
    report = scan_seeg(catalog, read)
    assert report["channel_map"][0]["raw_name"] == "EEG A1-Ref"
    assert report["channel_map"][0]["candidate"]["shaft"] == "A"
    assert report["channel_map"][0]["candidate"]["reference"] == "Ref"
    assert report["channel_map"][0]["channel_type"] == "unknown"
    assert [w["score"] for w in report["windows"]] == [100, 100, 100, None, None]


def test_explicit_seeg_type_overrides_name_candidate_without_modifying_signal():
    x = np.random.default_rng(3).normal(0, 1e-5, (2, 2500))
    catalog, read, _ = fixture(x, names=["ECG1", "C3"])
    report = scan_seeg(catalog, read, channel_map=[{"channel_id": "c0", "channel_type": "seeg", "status": "confirmed"}])
    assert report["windows"][0]["score"] == 100
    assert report["windows"][1]["score"] == 100


def test_bad_ranges_and_resources_fail_before_reading():
    catalog, read, calls = fixture(np.ones(6000))
    span = {"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 1000}
    for config in ({"ranges": [span, span]}, {"ranges": [{**span, "stop_sample": 7000}]},
                   {"max_scanned_samples": 100}, {"max_windows": 1}):
        with pytest.raises(ValueError):
            scan_seeg(catalog, read, config=config)
    assert calls == []


def test_review_cannot_turn_unknown_unit_into_a_full_score():
    catalog, read, _ = fixture(np.zeros(2500), units=["unknown"])
    report = scan_seeg(catalog, read)
    score = compute_quality_score(report["windows"], {report["windows"][0]["window_id"]: "rejected"})
    assert score["total"] is None
    assert score["unassessed_channel_seconds"] == 5


def test_numeric_overflow_is_explicit_and_not_a_json_nan_result():
    catalog, read, _ = fixture(np.full(2500, 1e200))
    with pytest.raises(ValueError, match="numeric|Numerical"):
        scan_seeg(catalog, read)
