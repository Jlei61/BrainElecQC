"""Optional formats use temporary known numerical inputs, never private files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brainelecqc.readers import EphysReaderError, discover_sources, identify_format, open_reader


def brainvision(tmp_path, segments=False):
    values = np.array([[-2, -1, 0, 1, 2, 3], [6, 5, 4, 3, 2, 1]], dtype="<i2")
    (tmp_path / "signal.eeg").write_bytes(values.T.tobytes())
    (tmp_path / "signal.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nCodepage=UTF-8\nDataFile=signal.eeg\nMarkerFile=signal.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=4000\n"
        "[Binary Infos]\nBinaryFormat=INT_16\n"
        "[Channel Infos]\nCh1=duplicate,,0.5,uV\nCh2=duplicate,,2,mV\n", encoding="utf-8")
    markers = "Mk1=New Segment,,1,1,0,20240101120000000000\nMk2=Stimulus,S  1,2,1,0\n"
    if segments:
        markers += "Mk3=New Segment,,4,1,0,20240101120001000000\nMk4=Stimulus,S  2,5,1,0\n"
    (tmp_path / "signal.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\nCodepage=UTF-8\nDataFile=signal.eeg\n[Marker Infos]\n" + markers,
        encoding="utf-8")
    return tmp_path / "signal.vhdr", values.astype(float) * np.array([.5e-6, 2e-3])[:, None]


def read(reader, start=None, stop=None, rows=None, segment=0):
    stream = reader.describe()["streams"][0]
    span = stream["segments"][segment]
    return reader.read_window(stream["stream_id"], span["segment_id"],
                              [stream["channels"][i]["channel_id"] for i in (range(len(stream["channels"])) if rows is None else rows)],
                              span["start_sample"] if start is None else start,
                              span["stop_sample"] if stop is None else stop)


@pytest.fixture
def optional_formats():
    return pytest.importorskip("mne")


def test_core_import_does_not_import_optional_mne():
    result = subprocess.run([sys.executable, "-c", "import sys; import brainelecqc.readers; assert 'mne' not in sys.modules"],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("extension", [".cnt", ".eeg", ".xyz"])
def test_ambiguous_or_unknown_suffix_never_selects_a_decoder(tmp_path, extension):
    path = tmp_path / ("signal" + extension)
    path.write_bytes(b"not a decoded file")
    with pytest.raises(EphysReaderError, match="Ambiguous|Unsupported"):
        open_reader(path, "source")


def test_nicolet_e_is_not_confused_with_data_or_nihon(tmp_path):
    path = tmp_path / "record.e"
    path.write_bytes(b"Nervus")
    assert identify_format(path) == "nicolet_e"
    with pytest.raises(EphysReaderError, match="separately validated decoder"):
        open_reader(path, "source")
    assert discover_sources(tmp_path)[0]["status"] == "unsupported_pending_validation"


def test_brainvision_calibration_duplicate_labels_order_and_no_cleaning(tmp_path):
    path, expected = brainvision(tmp_path)
    reader = open_reader(path, "bv")
    stream = reader.describe()["streams"][0]
    assert isinstance(reader._data, np.memmap) and reader._data.mode == "r"
    assert [c["raw_name"] for c in stream["channels"]] == ["duplicate", "duplicate"]
    assert [c["channel_type"] for c in stream["channels"]] == ["unknown", "unknown"]
    assert [c["unit"] for c in stream["channels"]] == ["V", "V"]
    assert stream["sampling_frequency_hz"] == 250
    assert stream["reference"] == "as_stored" and stream["processing_applied"] == []
    np.testing.assert_allclose(read(reader, 1, 6, [1, 0]), expected[[1, 0], 1:6], rtol=1e-15, atol=1e-18)
    events = reader.describe()["events"]
    assert events[1]["sample_mappings"][0]["sample_index"] == 1
    assert [s["relative_path"] for s in discover_sources(tmp_path)] == ["signal.vhdr"]
    np.testing.assert_array_equal(reader.read_window(stream["stream_id"], stream["segments"][0]["segment_id"], ["bv:ch:0"], 0, 1, "digital"), [[-2]])


def test_brainvision_explicit_segments_keep_elapsed_gap(tmp_path):
    path, expected = brainvision(tmp_path, segments=True)
    reader = open_reader(path, "bv")
    stream = reader.describe()["streams"][0]
    assert [(s["start_sample"], s["stop_sample"], s["recording_start_seconds"]) for s in stream["segments"]] == [(0, 3, 0), (3, 6, 1)]
    np.testing.assert_allclose(read(reader, segment=1), expected[:, 3:])
    with pytest.raises(EphysReaderError):
        read(reader, 2, 4)
    assert reader.describe()["events"][-1]["onset_seconds"] == 1.004


@pytest.mark.parametrize("mutation", ["missing_clock", "outside_sidecar", "partial_sample", "missing_sidecar", "changed_marker"])
def test_brainvision_bundle_errors_are_explicit(tmp_path, mutation):
    path, _ = brainvision(tmp_path, segments=True)
    if mutation == "changed_marker":
        reader = open_reader(path, "bv")
        marker = tmp_path / "signal.vmrk"
        marker.write_text(marker.read_text() + ";mutation\n")
        with pytest.raises(EphysReaderError, match="changed"):
            read(reader)
        return
    if mutation == "missing_clock":
        marker = tmp_path / "signal.vmrk"
        marker.write_text(marker.read_text().replace("20240101120001000000", ""))
    elif mutation == "outside_sidecar":
        path.write_text(path.read_text().replace("DataFile=signal.eeg", "DataFile=../signal.eeg"))
    elif mutation == "partial_sample":
        data = tmp_path / "signal.eeg"
        data.write_bytes(data.read_bytes() + b"x")
    else:
        (tmp_path / "signal.eeg").unlink()
    with pytest.raises((EphysReaderError, FileNotFoundError)):
        open_reader(path, "bv")


def test_discovery_does_not_hide_unrelated_eeg_beside_brainvision(tmp_path):
    brainvision(tmp_path)
    (tmp_path / "another.EEG").write_bytes(b"unknown vendor")
    entries = discover_sources(tmp_path)
    assert {entry["relative_path"] for entry in entries} == {"signal.vhdr", "another.EEG"}
    assert next(e for e in entries if e["relative_path"] == "another.EEG")["status"] == "requires_format_hint"


@pytest.mark.parametrize("format_name,dtype", [("IEEE_FLOAT_32", "<f4"), ("UINT_16", "<u2")])
def test_brainvision_vectorized_storage_and_float_adc_boundary(tmp_path, format_name, dtype):
    path, _ = brainvision(tmp_path)
    values = np.array([[1, 2, 65500], [5, 4, 3]], dtype=dtype)
    (tmp_path / "signal.eeg").write_bytes(values.tobytes())
    path.write_text(path.read_text().replace("MULTIPLEXED", "VECTORIZED").replace("INT_16", format_name))
    reader = open_reader(path, "vector")
    np.testing.assert_allclose(read(reader), values * np.array([.5e-6, .002])[:, None], rtol=1e-15)
    stream = reader.describe()["streams"][0]
    if format_name == "IEEE_FLOAT_32":
        with pytest.raises(EphysReaderError, match="ADC"):
            reader.read_window(stream["stream_id"], stream["segments"][0]["segment_id"], ["vector:ch:0"], 0, 1, "digital")


def test_fif_nonzero_source_origin_aux_channel_and_finite_window(tmp_path, optional_formats):
    mne = optional_formats
    expected = np.array([[1, -2, 3, 0, 5], [-7, 8, 9, 10, -11], [0, 1, 0, 2, 0]], float)
    expected[:2] *= 1e-6
    raw = mne.io.RawArray(expected, mne.create_info(["A1", "ECG", "TRIG"], 500, ["seeg", "ecg", "stim"]), first_samp=1000, verbose="ERROR")
    raw.set_annotations(mne.Annotations([.004], [0], ["pulse"]))
    path = tmp_path / "source_raw.fif"
    raw.save(path, fmt="double", overwrite=True, verbose="ERROR")
    reader = open_reader(path, "fif")
    info = reader.describe()
    stream = info["streams"][0]
    assert not reader._raw.preload
    assert stream["sample_index_origin"] == 1000
    assert stream["segments"][0]["recording_start_seconds"] == 2
    assert [c["channel_type"] for c in stream["channels"]] == ["seeg", "ecg", "stim"]
    assert stream["channels"][2]["quantity"] == "unknown"
    np.testing.assert_array_equal(read(reader, 1001, 1005, [2, 0, 1]), expected[[2, 0, 1], 1:5])
    assert info["events"][0]["sample_mappings"][0]["sample_index"] == 1002
    with pytest.raises(EphysReaderError):
        read(reader, 1004, 1006)


def test_fif_gap_annotation_is_not_returned_as_continuous_zero_data(tmp_path, optional_formats):
    mne = optional_formats
    raw = mne.io.RawArray(np.ones((1, 100)), mne.create_info(["A1"], 100, "eeg"), verbose="ERROR")
    raw.set_annotations(mne.Annotations([.2], [.2], ["BAD_ACQ_SKIP"]))
    path = tmp_path / "gap_raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    with pytest.raises(EphysReaderError, match="Discontinuity"):
        open_reader(path, "gap")


def test_nicolet_data_export_uses_header_gain_not_e_decoder(tmp_path, optional_formats):
    expected_codes = np.array([[1, -2, 3], [4, 5, -6]], dtype="<i2")
    path = tmp_path / "source.data"
    path.write_bytes(expected_codes.T.tobytes())
    (tmp_path / "source.head").write_text("elec_names=[A1,A1]\nconversion_factor=0.25\nnum_channels=2\nnum_samples=3\nsample_freq=2000\nstart_ts=2024-01-01 12:00:00.000000\n")
    reader = open_reader(path, "nicolet")
    np.testing.assert_array_equal(read(reader), expected_codes * .25e-6)
    assert reader.describe()["format"] == "nicolet_data"
    assert [c["raw_name"] for c in reader.describe()["streams"][0]["channels"]] == ["A1", "A1"]


def eeglab(tmp_path, compressed=False, embedded=False, trials=1):
    from scipy.io import savemat
    expected_uv = np.array([[1, -2, 3, 4], [5, 6, -7, 8]], dtype="<f4")
    data = tmp_path / "signal.fdt"
    data.write_bytes(expected_uv.T.tobytes())
    eeg = {"data": expected_uv if embedded else "signal.fdt", "trials": trials, "nbchan": 2,
           "pnts": 4, "srate": 100., "xmin": 0., "xmax": .03,
           "chanlocs": np.array([("A1",), ("A1",)], dtype=[("labels", "O")]),
           "event": np.array([("pulse", 2., 0.)], dtype=[("type", "O"), ("latency", "O"), ("duration", "O")])}
    path = tmp_path / "signal.set"
    savemat(path, {"EEG": eeg}, do_compression=compressed)
    return path, expected_uv.astype(float) * 1e-6


def test_eeglab_external_fdt_calibration_and_event(tmp_path, optional_formats):
    path, expected = eeglab(tmp_path)
    reader = open_reader(path, "set")
    assert not reader._raw.preload
    np.testing.assert_array_equal(read(reader, 1, 4, [1, 0]), expected[[1, 0], 1:4])
    assert reader.describe()["events"][0]["sample_mappings"][0]["sample_index"] == 1
    assert [c["raw_name"] for c in reader.describe()["streams"][0]["channels"]] == ["A1", "A1"]


@pytest.mark.parametrize("options", [{"compressed": True}, {"embedded": True}, {"trials": 2}])
def test_eeglab_unsupported_storage_does_not_preload(tmp_path, options):
    path, _ = eeglab(tmp_path, **options)
    with pytest.raises(EphysReaderError, match="Compressed|continuous EEGLAB"):
        open_reader(path, "set")
