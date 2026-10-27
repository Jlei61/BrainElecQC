"""Identity-bound Python windows using byte-authored generated signals."""
import io
import os

import numpy as np
import pytest
from brainelecqc.config import settings
from brainelecqc import signals as ephys_signals
from brainelecqc.recordings import import_recording
from brainelecqc.signals import describe_artifact, list_events
from brainelecqc.storage import ImportConflict
from brainelecqc.windows import samples, overview
from tests.edf_fixtures import tal, write_edf_bdf
from tests.helpers import NamedInput


@pytest.fixture
def example(isolated_storage, tmp_path):
    source = write_edf_bdf(tmp_path / "fixture.edf", [
        {"label": "same", "records": [[-2, -1, 0, 1], [2, 3, 4, 5], [6, 7, 8, 9]]},
        {"label": "same", "records": [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], "unit": "mV"},
        {"label": "TRIG", "records": [[0, 1], [0, 2], [0, 3]], "unit": ""}],
        reserved="EDF+D", annotation_records=[tal(.25) + tal(.5, "pulse"), tal(1.25), tal(5.25)])
    record, _ = import_recording({"participant_id": "example", "modality": "eeg", "label": "fixture"},
                                [NamedInput(stream=io.BytesIO(source.read_bytes()), filename="fixture.edf")])
    aid = record["input_artifact_id"]
    catalog = describe_artifact(aid)
    stream = catalog["streams"][0]
    params = {"artifact_id": aid, "stream_id": stream["stream_id"],
              "segment_id": stream["segments"][0]["segment_id"],
              "channel_ids": [stream["channels"][1]["channel_id"], stream["channels"][0]["channel_id"]],
              "start_sample": 3, "stop_sample": 6, "representation": "physical"}
    return record, catalog, params


def test_catalog_and_samples_preserve_order_calibration_and_source_time(example):
    record, catalog, params = example
    assert len(catalog["streams"]) == 2
    assert catalog["scientific_validity"] == "pending"
    assert catalog["processing"] == []
    assert catalog["streams"][0]["channels"][0]["raw_name"] == "same"
    meta, data = samples(**params)
    assert meta["artifact_id"] == record["input_artifact_id"]
    assert meta["recording_id"] == record["recording_id"]
    assert meta["content_sha256"] == record["input_artifact"]["content_sha256"]
    assert meta["channel_ids"] == params["channel_ids"]
    assert meta["shape"] == [2, 3] and meta["units"] == ["V", "V"]
    assert meta["dtype"] == "<f8" and meta["order"] == "C"
    assert data.dtype == np.dtype("<f8") and data.flags.c_contiguous
    assert meta["data_kind"] == "raw_samples" and meta["processing"] == []
    assert meta["recording_start_seconds"] == 1.0
    assert meta["clock_alignment"] == "source_local_unaligned"
    np.testing.assert_allclose(data, [[.009, .008, .007], [1e-6, 2e-6, 3e-6]], atol=1e-15)
    dmeta, codes = samples(**{**params, "representation": "digital"})
    np.testing.assert_array_equal(codes, [[9, 8, 7], [1, 2, 3]])
    assert dmeta["units"] == ["count", "count"]


def test_overview_is_distinct_minmax_with_short_final_bin(example):
    _, _, params = example
    query = {**params, "start_sample": 0, "stop_sample": 7, "representation": "digital", "bin_samples": 3}
    meta, values = overview(**query)
    assert meta["data_kind"] == "minmax_overview" and meta["shape"] == [2, 3, 2]
    assert meta["bin_start_samples"] == [0, 3, 6] and meta["bin_stop_samples"] == [3, 6, 7]
    np.testing.assert_array_equal(values, [[[10, 12], [7, 9], [6, 6]], [[-2, 0], [1, 3], [4, 4]]])


@pytest.mark.parametrize("field,value", [("stop_sample", 9), ("start_sample", -1), ("start_sample", 3.5),
                                        ("representation", "filtered"), ("channel_ids", ["unknown"]),
                                        ("stop_sample", 1000000000), ("segment_id", "unknown")])
def test_invalid_or_gap_crossing_windows_raise_before_returning_data(example, field, value):
    _, _, params = example
    with pytest.raises((ValueError, LookupError)):
        samples(**{**params, field: value})


def test_second_segment_keeps_gap_and_events_keep_source_clock(example):
    record, catalog, params = example
    span = catalog["streams"][0]["segments"][1]
    query = {**params, "segment_id": span["segment_id"], "start_sample": 8, "stop_sample": 9}
    meta, data = samples(**query)
    assert meta["recording_start_seconds"] == 5.25 and data.shape == (2, 1)
    events = list_events(record["input_artifact_id"], offset=0, limit=1)
    assert events["total"] == 1 and len(events["events"]) == 1
    event = events["events"][0]
    assert event["text"] == "pulse" and event["clock_alignment"] == "source_local_unaligned"
    assert event["sample_mappings"][0]["sample_index"] == 1


def test_warm_reader_cache_rechecks_fingerprint_and_rejects_same_size_tampering(example, monkeypatch):
    record, _, params = example
    checks = []
    original = ephys_signals.inspect_snapshot

    def observed(*args):
        checks.append(args)
        return original(*args)

    monkeypatch.setattr(ephys_signals, "inspect_snapshot", observed)
    samples(**params)
    assert checks == []  # Already SHA256-verified catalog, exact stat guard still matches.
    path = settings.storage_root / record["input_artifact"]["storage_path"] / "files" / "fixture.edf"
    before = path.stat()
    path.chmod(0o644)
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ImportConflict, match="fingerprint mismatch"):
        samples(**params)
    assert checks


def test_mutation_during_window_is_discarded(example, monkeypatch):
    record, catalog, params = example
    entry = ephys_signals.verified_catalog(record["input_artifact_id"])
    reader = entry["readers"][catalog["streams"][0]["stream_id"]]
    original = reader.read_window

    def mutated(*args, **kwargs):
        data = original(*args, **kwargs)
        (entry["root"] / "files" / "unexpected.txt").write_bytes(b"changed input set")
        return data

    monkeypatch.setattr(reader, "read_window", mutated)
    with pytest.raises(ImportConflict, match="changed during window read"):
        samples(**params)


def test_unreadable_source_is_explicit_without_hiding_readable_voltage(example):
    record, _ = import_recording({"participant_id": "example", "modality": "eeg", "label": "unsupported"},
                                [NamedInput(stream=io.BytesIO(b"not EDF"), filename="bad.edf")])
    catalog = describe_artifact(record["input_artifact_id"])
    assert catalog["status"] == "unreadable" and catalog["errors"]
    assert catalog["streams"] == []


@pytest.mark.parametrize("channels,start,stop", [(None, 0, 1), (["x"], "0", 1),
                                                (["x"], 0, None), (["x"], False, 1),
                                                (["x", "x"], 0, 1)])
def test_python_window_contract_rejects_bad_types_before_arithmetic(channels, start, stop):
    with pytest.raises(ValueError):
        samples("absent", "stream", "segment", channels, start, stop, "physical")
    with pytest.raises(ValueError):
        overview("absent", "stream", "segment", channels, start, stop, "physical", 1)
