"""Derived voltage windows retain identity, SI units and nonzero stored origins."""
import numpy as np
import pytest

from brainelecqc.readers.edf import EphysReaderError
from brainelecqc.readers.npy import NpyVoltageReader


def stream():
    return {"stream_id": "derived:stream", "source_id": "derived:source", "clock_id": "parent:clock",
            "sampling_frequency_hz": 2, "sample_index_origin": 100, "sample_count": 6,
            "channels": [{"channel_id": "c0", "raw_name": "same", "unit": "V", "source_channel_ids": ["parent:c0"]},
                         {"channel_id": "c1", "raw_name": "same", "unit": "V", "source_channel_ids": ["parent:c1"]}],
            "segments": [{"segment_id": "s0", "start_sample": 100, "stop_sample": 104,
                          "recording_start_seconds": .25, "parent_stream_id": "parent:stream", "parent_segment_id": "parent:s0",
                          "parent_start_sample": 10, "parent_stop_sample": 14, "clock_mapping": "identity"},
                         {"segment_id": "s1", "start_sample": 104, "stop_sample": 106,
                          "recording_start_seconds": 10.25, "parent_stream_id": "parent:stream", "parent_segment_id": "parent:s1",
                          "parent_start_sample": 14, "parent_stop_sample": 16, "clock_mapping": "identity"}]}


def fixture(tmp_path, data=None, metadata=None):
    data = np.array([[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]], dtype="<f8") * 1e-6 if data is None else data
    path = tmp_path / "voltage.npy"
    np.save(path, data, allow_pickle=False)
    return path, stream() if metadata is None else metadata, data


def test_nonzero_origin_duplicate_names_order_and_independent_copy(tmp_path):
    path, metadata, data = fixture(tmp_path)
    reader = NpyVoltageReader(path, metadata)
    observed = reader.read_window("derived:stream", "s0", ["c1", "c0"], 101, 104)
    np.testing.assert_array_equal(observed, data[[1, 0], 1:4])
    assert observed.flags.c_contiguous and observed.dtype == np.dtype("<f8")
    observed[:] = 0
    np.testing.assert_array_equal(reader.read_window("derived:stream", "s1", ["c0"], 104, 106), data[[0], 4:6])
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), data)
    metadata["channels"][0]["unit"] = "unknown"
    assert reader.describe()["streams"][0]["channels"][0]["unit"] == "V"


@pytest.mark.parametrize("first,last,segment", [(99, 101, "s0"), (103, 105, "s0"), (104, 107, "s1"), (104, 104, "s1")])
def test_no_tail_padding_or_gap_crossing(tmp_path, first, last, segment):
    path, metadata, _ = fixture(tmp_path)
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(path, metadata).read_window("derived:stream", segment, ["c0"], first, last)


def test_no_adc_codes_for_derived_voltage(tmp_path):
    path, metadata, _ = fixture(tmp_path)
    with pytest.raises(EphysReaderError, match="physical"):
        NpyVoltageReader(path, metadata).read_window("derived:stream", "s0", ["c0"], 100, 101, representation="digital")


@pytest.mark.parametrize("data", [np.zeros((2, 6), dtype="<f4"), np.zeros((2, 6), dtype=">f8"),
                                   np.zeros((6, 2), dtype="<f8"), np.zeros((2, 6, 1), dtype="<f8"),
                                   np.asfortranarray(np.zeros((2, 6), dtype="<f8"))])
def test_wrong_shape_dtype_or_order_are_rejected(tmp_path, data):
    path, metadata, _ = fixture(tmp_path, data)
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(path, metadata)


@pytest.mark.parametrize("mutation", ["unknown_unit", "duplicate_id", "missing_parent", "wrong_parent_length", "sample_hole", "overlap_clock"])
def test_descriptor_contract_cannot_be_guessed(tmp_path, mutation):
    metadata = stream()
    if mutation == "unknown_unit":
        metadata["channels"][0]["unit"] = "uV"
    elif mutation == "duplicate_id":
        metadata["channels"][1]["channel_id"] = "c0"
    elif mutation == "missing_parent":
        metadata["channels"][0]["source_channel_ids"] = []
    elif mutation == "wrong_parent_length":
        metadata["segments"][0]["parent_stop_sample"] = 15
    elif mutation == "sample_hole":
        metadata["segments"][1]["start_sample"] = 105
    else:
        metadata["segments"][1]["recording_start_seconds"] = .5
    path, _, _ = fixture(tmp_path)
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(path, metadata)


def test_npy_mmap_does_not_scan_all_values_and_rejects_nonfinite_window(tmp_path):
    path, metadata, data = fixture(tmp_path)
    mapped = np.lib.format.open_memmap(path, mode="r+")
    mapped[1, 5] = np.nan
    mapped.flush()
    del mapped
    reader = NpyVoltageReader(path, metadata)
    assert isinstance(reader._data, np.memmap)
    assert reader._data.mode == "r"
    np.testing.assert_array_equal(reader.read_window("derived:stream", "s0", ["c0", "c1"], 100, 104), data[:, :4])
    with pytest.raises(EphysReaderError, match="finite"):
        reader.read_window("derived:stream", "s1", ["c1"], 104, 106)


@pytest.mark.parametrize("change", ["truncated", "extra", "modified_after_open"])
def test_exact_payload_and_file_identity_are_required(tmp_path, change):
    path, metadata, _ = fixture(tmp_path)
    if change == "modified_after_open":
        reader = NpyVoltageReader(path, metadata)
        mapped = np.lib.format.open_memmap(path, mode="r+")
        mapped[0, 0] = 10
        mapped.flush()
        del mapped
        with pytest.raises(EphysReaderError, match="changed"):
            reader.read_window("derived:stream", "s0", ["c0"], 100, 101)
    else:
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] if change == "truncated" else raw + b"extra")
        with pytest.raises(EphysReaderError):
            NpyVoltageReader(path, metadata)


def test_object_arrays_and_archives_never_load_as_voltage(tmp_path):
    path = tmp_path / "object.npy"
    np.save(path, np.array([["not voltage"]], dtype=object), allow_pickle=True)
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(path, stream())
    archive = tmp_path / "archive.npz"
    np.savez(archive, voltage=np.zeros((2, 6)))
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(archive, stream())


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), 10 ** 1000, 1e-309, True])
def test_invalid_sampling_clock_is_rejected(tmp_path, frequency):
    path, metadata, _ = fixture(tmp_path)
    metadata["sampling_frequency_hz"] = frequency
    with pytest.raises(EphysReaderError):
        NpyVoltageReader(path, metadata)
