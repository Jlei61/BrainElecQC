"""Format truth tests: known bytes, no cleaning and no inferred continuous time."""
import base64

import numpy as np
import pytest

from brainelecqc.readers.edf import EdfBdfReader, EphysReaderError
from tests.edf_fixtures import tal, write_edf_bdf


def window(reader, stream=0, segment=0, channels=None, start=None, stop=None, representation="physical"):
    info = reader.describe()["streams"][stream]
    span = info["segments"][segment]
    return reader.read_window(info["stream_id"], span["segment_id"],
                              channels or [c["channel_id"] for c in info["channels"]],
                              span["start_sample"] if start is None else start,
                              span["stop_sample"] if stop is None else stop,
                              representation=representation)


@pytest.mark.parametrize("bdf", [False, True])
def test_signed_codes_offsets_and_channel_order(tmp_path, bdf):
    limit = 2 ** (23 if bdf else 15)
    codes = [-limit, -1, 0, 1, limit - 1]
    path = write_edf_bdf(tmp_path / "signal.dat", [
        {"label": "duplicate", "records": [codes]},
        {"label": "duplicate", "records": [codes], "physical_min": -10,
         "physical_max": 30, "digital_min": -100, "digital_max": 100, "unit": "mV"},
        {"label": "A1-A2", "records": [[1] * 5], "unit": "Ohm"}], bdf=bdf)
    reader = EdfBdfReader(path, "input")
    channels = reader.describe()["streams"][0]["channels"]
    assert [c["raw_name"] for c in channels] == ["duplicate", "duplicate", "A1-A2"]
    assert len({c["channel_id"] for c in channels}) == 3
    assert [c["source_index"] for c in channels] == [0, 1, 2]
    np.testing.assert_array_equal(window(reader, representation="digital")[0], codes)
    np.testing.assert_allclose(window(reader)[0], np.array(codes) * 1e-6, atol=1e-15)
    np.testing.assert_allclose(window(reader)[1], (np.array(codes) * .2 + 10) * .001)
    assert channels[2]["quantity"] == "unknown"
    assert channels[2]["unit"] == "Ohm"
    np.testing.assert_array_equal(window(reader, channels=[channels[2]["channel_id"], channels[0]["channel_id"]],
                                        representation="digital"), [[1] * 5, codes])


def test_equivalent_volt_and_microvolt_and_negative_gain(tmp_path):
    path = write_edf_bdf(tmp_path / "units.edf", [
        {"label": "V", "records": [[-10, 0, 10]], "unit": "V", "physical_min": -.001,
         "physical_max": .001, "digital_min": -10, "digital_max": 10},
        {"label": "uV", "records": [[-10, 0, 10]], "unit": "uV", "physical_min": -1000,
         "physical_max": 1000, "digital_min": -10, "digital_max": 10},
        {"label": "inverted", "records": [[-10, 0, 10]], "unit": "uV", "physical_min": 1000,
         "physical_max": -1000, "digital_min": -10, "digital_max": 10}])
    data = window(EdfBdfReader(path, "units"))
    np.testing.assert_array_equal(data[0], data[1])
    np.testing.assert_array_equal(data[0], -data[2])


def test_mixed_sampling_rates_auxiliaries_are_not_resampled(tmp_path):
    path = write_edf_bdf(tmp_path / "mixed.edf", [
        {"label": "EEG C3", "records": [[0, 1, 2, 3], [4, 5, 6, 7]]},
        {"label": "ECG", "records": [[90, 91], [92, 93]], "unit": ""},
        {"label": "TRIG", "records": [[7, 6, 5, 4], [3, 2, 1, 0]], "unit": ""}])
    reader = EdfBdfReader(path, "mixed")
    streams = reader.describe()["streams"]
    assert [x["sampling_frequency_hz"] for x in streams] == [4, 2]
    assert [[c["source_index"] for c in s["channels"]] for s in streams] == [[0, 2], [1]]
    np.testing.assert_array_equal(window(reader, start=3, stop=6, representation="digital"), [[3, 4, 5], [4, 3, 2]])
    np.testing.assert_array_equal(window(reader, stream=1, representation="digital"), [[90, 91, 92, 93]])


@pytest.mark.parametrize("bdf", [False, True])
def test_nonzero_origin_gaps_events_and_utf8(tmp_path, bdf):
    annotations = [tal("0.25") + tal("0.75", "刺激", "0.5"), tal("1.25"),
                   tal("4.25") + tal("4.75", "again") + tal("4.75", "again")]
    path = write_edf_bdf(tmp_path / "gaps.dat", [{"label": "A1-A2", "records": [[1, 2], [3, 4], [5, 6]]}],
                         bdf=bdf, reserved=("BDF" if bdf else "EDF") + "+D", annotation_records=annotations)
    reader = EdfBdfReader(path, "gaps")
    description = reader.describe()
    spans = description["streams"][0]["segments"]
    assert [(x["start_sample"], x["stop_sample"], x["recording_start_seconds"]) for x in spans] == [(0, 4, .25), (4, 6, 4.25)]
    np.testing.assert_array_equal(window(reader, segment=1, representation="digital"), [[5, 6]])
    with pytest.raises(EphysReaderError):
        window(reader, start=3, stop=5)
    events = description["events"]
    assert [x["text"] for x in events] == ["刺激", "again", "again"]
    assert events[0]["duration_seconds"] == .5
    assert len({x["event_id"] for x in events}) == 3
    assert events[0]["sample_mappings"][0]["sample_index"] == 1
    assert events[1]["sample_mappings"][0]["sample_index"] == 5


def test_continuous_flag_cannot_erase_gap(tmp_path):
    path = write_edf_bdf(tmp_path / "bad_c.edf", [{"label": "C3", "records": [[1], [2]]}],
                         reserved="EDF+C", annotation_records=[tal(0), tal(4)])
    description = EdfBdfReader(path, "bad_c").describe()
    assert len(description["streams"][0]["segments"]) == 2
    assert "continuous_flag_has_gaps" in {w["code"] for w in description["warnings"]}


def test_real_vendor_missing_nul_and_non_utf8_are_explicit(tmp_path):
    raw = b"+0.000000\x14\x14+0.000000\x14Segment:\xd6\xd0\xce\xc4\x14\x00"
    path = write_edf_bdf(tmp_path / "vendor.edf", [{"label": "EEG", "records": [[1, 2]]}],
                         reserved="EDF+D", annotation_records=[raw])
    description = EdfBdfReader(path, "vendor").describe()
    assert {w["code"] for w in description["warnings"]} >= {"timekeeping_missing_nul_separator", "annotation_non_utf8"}
    event = description["events"][0]
    assert event["text"] is None
    assert event["encoding_status"] == "undecoded_non_utf8"
    assert base64.b64decode(event["raw_text_base64"]) == b"Segment:\xd6\xd0\xce\xc4"
    assert event["interpretation_status"] == "vendor_recovered_ambiguous"
    assert base64.b64decode(event["raw_source_tal_base64"]) == raw
    assert event["alternative_standard_interpretation"]["onset_seconds"] == 0


def test_timestamp_like_annotation_outside_record_is_not_auto_repaired(tmp_path):
    raw = b"+0\x14\x14+123\x14literal text\x14\0"
    path = write_edf_bdf(tmp_path / "counterexample.edf", [{"label": "EEG", "records": [[1, 2]]}],
                         reserved="EDF+D", annotation_records=[raw])
    description = EdfBdfReader(path, "counterexample").describe()
    assert [event["text"] for event in description["events"]] == ["+123", "literal text"]
    assert [event["onset_seconds"] for event in description["events"]] == [0, 0]
    assert all(event["interpretation_status"] == "standard" for event in description["events"])
    assert "timekeeping_missing_nul_separator" not in {warning["code"] for warning in description["warnings"]}


@pytest.mark.parametrize("stop", [4, 100])
def test_tail_window_never_zero_pads(tmp_path, stop):
    path = write_edf_bdf(tmp_path / "short.edf", [{"label": "C3", "records": [[1, 2, 3]]}])
    reader = EdfBdfReader(path, "short")
    assert window(reader, start=2, stop=3).shape == (1, 1)
    with pytest.raises(EphysReaderError):
        window(reader, start=2, stop=stop)


@pytest.mark.parametrize("kind", ["truncated", "extra", "zero_calibration", "missing_tal", "backward_time"])
def test_invalid_input_is_not_repaired_silently(tmp_path, kind):
    channel = {"label": "C3", "records": [[1, 2], [3, 4]]}
    options = {}
    if kind == "zero_calibration":
        channel.update(physical_min=1, physical_max=1)
    if kind in ("missing_tal", "backward_time"):
        options = {"reserved": "EDF+D", "annotation_records": [tal(1), b"" if kind == "missing_tal" else tal(0)]}
    path = write_edf_bdf(tmp_path / "bad.edf", [channel], **options)
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    if kind == "extra":
        path.write_bytes(path.read_bytes() + b"extra")
    with pytest.raises(EphysReaderError):
        EdfBdfReader(path, "bad")


def test_unknown_record_count_is_named_and_file_change_is_detected(tmp_path):
    path = write_edf_bdf(tmp_path / "unknown.edf", [{"label": "C3", "records": [[1, 2]]}], record_count=-1)
    reader = EdfBdfReader(path, "unknown")
    assert "record_count_from_complete_payload" in {w["code"] for w in reader.describe()["warnings"]}
    path.write_bytes(path.read_bytes()[:-2] + b"\0\0")
    with pytest.raises(EphysReaderError, match="changed"):
        window(reader)


def test_zero_duration_annotation_only_has_no_fake_frequency(tmp_path):
    path = write_edf_bdf(tmp_path / "events.edf", [], reserved="EDF+D", duration="0",
                         annotation_records=[tal(0) + tal(42, "event")])
    description = EdfBdfReader(path, "events").describe()
    assert description["streams"] == []
    assert description["events"][0]["onset_seconds"] == 42
    assert description["events"][0]["sample_mappings"] == []


@pytest.mark.parametrize("physical_unit", [b"\xc2\xb5V", b"\xb5V", b"\xce\xbcV"])
def test_explicit_microvolt_spellings_are_scaled_and_audited(tmp_path, physical_unit):
    path = write_edf_bdf(tmp_path / "unit.edf", [{"label": "C3", "records": [[10, -20]], "unit": physical_unit}])
    reader = EdfBdfReader(path, "unit")
    np.testing.assert_allclose(window(reader), [[10e-6, -20e-6]])
    assert "header_non_ascii" in {w["code"] for w in reader.describe()["warnings"]}


def test_fractional_and_gap_events_do_not_get_rounded_to_samples(tmp_path):
    path = write_edf_bdf(tmp_path / "events.edf", [{"label": "C3", "records": [[1, 2], [3, 4]]}],
                         reserved="EDF+D", annotation_records=[tal(0) + tal("0.1", "fraction") + tal(2, "gap"), tal(4)])
    events = EdfBdfReader(path, "fraction").describe()["events"]
    assert events[0]["sample_mappings"][0]["sample_position"] == .2
    assert events[0]["sample_mappings"][0]["sample_index"] is None
    assert events[1]["sample_mappings"] == []


@pytest.mark.parametrize("selection", [[0], [["not hashable"]], [], ["absent"]])
def test_invalid_channel_selection_has_contract_error(tmp_path, selection):
    path = write_edf_bdf(tmp_path / "short.edf", [{"label": "C3", "records": [[1]]}])
    reader = EdfBdfReader(path, "short")
    stream = reader.describe()["streams"][0]
    with pytest.raises(EphysReaderError):
        reader.read_window(stream["stream_id"], stream["segments"][0]["segment_id"], selection, 0, 1)


@pytest.mark.parametrize("calibration", ["NaN", "Infinity", "1e999"])
def test_nonfinite_calibration_cannot_become_voltage(tmp_path, calibration):
    path = write_edf_bdf(tmp_path / "bad.edf", [{"label": "C3", "records": [[1]], "physical_max": calibration}])
    with pytest.raises(EphysReaderError):
        EdfBdfReader(path, "bad")


@pytest.mark.parametrize("options", [
    {"reserved": "EDF+D"},
    {"reserved": "EDF+Z"},
    {"reserved": "BDF+D"},
    {"duration": "1e-309"},
    {"duration": "1e-999"},
])
def test_invalid_clock_cannot_silently_become_continuous(tmp_path, options):
    path = write_edf_bdf(tmp_path / "bad_clock.edf", [{"label": "C3", "records": [[1]]}], **options)
    with pytest.raises(EphysReaderError):
        EdfBdfReader(path, "bad_clock")


def test_multiple_annotation_channels_keep_same_sample_events_distinct(tmp_path):
    path = write_edf_bdf(tmp_path / "events.edf", [
        {"label": "C3", "records": [[1, 2]]},
        {"label": "EDF Annotations", "annotation": True, "samples": 64, "records": [tal(0) + tal("0.5", "event")]},
        {"label": "EDF Annotations", "annotation": True, "samples": 64, "records": [tal("0.5", "event")]}],
        reserved="EDF+C")
    reader = EdfBdfReader(path, "multi-ann")
    events = reader.describe()["events"]
    assert len(events) == 2
    assert events[0]["event_id"] != events[1]["event_id"]
    assert [event["annotation_channel_index"] for event in events] == [1, 2]
    assert [event["sample_mappings"][0]["sample_index"] for event in events] == [1, 1]
