"""Known-byte header oracles, corruption, anonymity and mixed sampling rates."""
import hashlib

import pytest

from brainelecqc.eeg.header_qc import inspect_input_header, score_header
from tests.edf_fixtures import write_edf_bdf, tal


@pytest.mark.parametrize("bdf", [False, True])
def test_complete_anonymous_header_and_mixed_rates_are_valid(tmp_path, bdf):
    path = write_edf_bdf(tmp_path/("record.bdf" if bdf else "record.edf"), [
        {"label": "EEG C3", "records": [[1, 2, 3, 4]]},
        {"label": "ECG", "records": [[1, 2]]}], bdf=bdf, duration="0.5")
    data = bytearray(path.read_bytes())
    data[8:168] = b" "*160
    path.write_bytes(data)
    result = inspect_input_header(path)
    assert result["auto_points"] == 10
    assert result["sampling_rates_hz"] == [4, 8]
    assert result["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["identity_fields"] == {"patient_field_present": False, "recording_field_present": False}
    assert [c["raw_name"] for c in result["channels"]] == ["EEG C3", "ECG"]
    assert "patient" not in result and "recording" not in result


@pytest.mark.parametrize("offset,value,key,deduction", [
    (184, b"bad     ", "layout", 8), (256, b" "*16, "channels", 1),
    (244, b"0       ", "sampling", 4), (352, b" "*8, "calibration", 2),
    (360, b"32767   ", "calibration", 2), (236, b"2       ", "payload", 2),
    (168, b"xx.xx.xx", "clock", 1), (0, b"bogus   ", "signature", 1)])
def test_header_failures_are_reported_before_signal_decode(tmp_path, offset, value, key, deduction):
    path = write_edf_bdf(tmp_path/"record.edf", [{"label": "EEG C3", "records": [[1, 2, 3]]}])
    data = bytearray(path.read_bytes())
    data[offset:offset+len(value)] = value
    path.write_bytes(data)
    result = inspect_input_header(path)
    assert next(c for c in result["checks"] if c["id"] == key)["status"] == "needs_review"
    assert result["auto_points"] == 10-deduction
    assert score_header(result)["score"] == result["auto_points"]


def test_payload_truncation_and_wrong_source_are_detected(tmp_path):
    path = write_edf_bdf(tmp_path/"record.edf", [{"label": "C3", "records": [[1, 2, 3]]}])
    original = inspect_input_header(path)["source_sha256"]
    path.write_bytes(path.read_bytes()[:-1])
    result = inspect_input_header(path)
    assert result["auto_points"] == 8
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_input_header(path, expected_sha256=original)


@pytest.mark.parametrize("bdf", [False, True])
def test_plus_annotation_units_and_discontinuous_times(tmp_path, bdf):
    path = write_edf_bdf(tmp_path/("record.bdf" if bdf else "record.edf"),
        [{"label": "C3", "records": [[1, 2], [3, 4]]}], bdf=bdf,
        reserved="BDF+D" if bdf else "EDF+D", annotation_records=[tal(0), tal(5)])
    assert inspect_input_header(path)["auto_points"] == 10


def test_unknown_record_count_is_inferred_from_complete_payload(tmp_path):
    path = write_edf_bdf(tmp_path/"record.edf", [{"label": "C3", "records": [[1, 2, 3]]}], record_count=-1)
    result = inspect_input_header(path)
    assert result["auto_points"] == 10 and result["record_count"] == 1


@pytest.mark.parametrize("extension", [".e", ".nkt", ".EEG"])
def test_vendor_extensions_require_an_identified_format(tmp_path, extension):
    path = tmp_path/("sample"+extension)
    path.write_bytes(b"vendor signature to identify")
    result = inspect_input_header(path)
    assert result["status"] == "format_confirmation_required"
    assert result["auto_points"] is None


def test_truncated_fixed_header_returns_checks(tmp_path):
    path = tmp_path/"record.edf"
    path.write_bytes(b"0       ")
    result = inspect_input_header(path)
    assert result["auto_points"] == 1


def test_biosemi_status_is_dimensionless_but_eeg_requires_a_unit(tmp_path):
    path = write_edf_bdf(tmp_path/"record.bdf", [
        {"label": "C3", "records": [[1, 2, 3]]},
        {"label": "Status", "unit": "", "records": [[0, 0, 0]]}], bdf=True)
    assert inspect_input_header(path)["auto_points"] == 10
    raw = bytearray(path.read_bytes())
    raw[256+96*2:256+96*2+8] = b" "*8
    path.write_bytes(raw)
    assert inspect_input_header(path)["auto_points"] == 8
