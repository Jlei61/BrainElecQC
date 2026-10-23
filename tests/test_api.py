"""Identity-bound binary windows using byte-authored generated signals."""
import io
import json
import os
import struct

import numpy as np
import pytest
from flask import Flask
from werkzeug.datastructures import FileStorage

from brainelecqc.api import ephys_bp
from brainelecqc.api import ephys_samples_bp
from brainelecqc.config import settings
from brainelecqc import signals as ephys_signals
from brainelecqc.recordings import import_recording
from brainelecqc.windows import samples, overview
from tests.edf_fixtures import tal, write_edf_bdf


@pytest.fixture
def example(isolated_storage, tmp_path):
    source = write_edf_bdf(tmp_path / "fixture.edf", [
        {"label": "same", "records": [[-2, -1, 0, 1], [2, 3, 4, 5], [6, 7, 8, 9]]},
        {"label": "same", "records": [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], "unit": "mV"},
        {"label": "TRIG", "records": [[0, 1], [0, 2], [0, 3]], "unit": ""}],
        reserved="EDF+D", annotation_records=[tal(.25) + tal(.5, "pulse"), tal(1.25), tal(5.25)])
    record, _ = import_recording({"participant_id": "example", "modality": "eeg", "label": "fixture"},
                                [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="fixture.edf")])
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(ephys_bp)
    app.register_blueprint(ephys_samples_bp)
    client = app.test_client()
    aid = record["input_artifact_id"]
    catalog = client.get(f"/api/v1/ephys/artifacts/{aid}/signals").get_json()["data"]
    stream = catalog["streams"][0]
    params = [("stream_id", stream["stream_id"]), ("segment_id", stream["segments"][0]["segment_id"]),
              ("channel_id", stream["channels"][1]["channel_id"]),
              ("channel_id", stream["channels"][0]["channel_id"]),
              ("start_sample", 3), ("stop_sample", 6), ("representation", "physical")]
    return client, record, catalog, params


def decode(response):
    assert response.status_code == 200, response.get_json()
    assert response.mimetype == "application/octet-stream"
    raw = response.data
    assert raw[:4] == b"EPHY"
    length = struct.unpack("<I", raw[4:8])[0]
    meta = json.loads(raw[8:8 + length])
    offset = (8 + length + 7) // 8 * 8
    assert not any(raw[8 + length:offset])
    array = np.frombuffer(raw[offset:], dtype="<f8").reshape(meta["shape"])
    return meta, array


def endpoint(record, name="samples"):
    return f"/api/v1/ephys/artifacts/{record['input_artifact_id']}/{name}"


def test_catalog_and_binary_preserve_order_calibration_and_source_time(example):
    client, record, catalog, params = example
    assert len(catalog["streams"]) == 2
    assert catalog["scientific_validity"] == "pending"
    assert catalog["processing"] == []
    assert catalog["streams"][0]["channels"][0]["raw_name"] == "same"
    meta, data = decode(client.get(endpoint(record), query_string=params))
    assert meta["artifact_id"] == record["input_artifact_id"]
    assert meta["recording_id"] == record["recording_id"]
    assert meta["content_sha256"] == record["input_artifact"]["content_sha256"]
    assert meta["channel_ids"] == [v for k, v in params if k == "channel_id"]
    assert meta["shape"] == [2, 3] and meta["units"] == ["V", "V"]
    assert meta["dtype"] == "<f8" and meta["order"] == "C"
    assert meta["data_kind"] == "raw_samples" and meta["processing"] == []
    assert meta["recording_start_seconds"] == 1.0
    assert meta["clock_alignment"] == "source_local_unaligned"
    np.testing.assert_allclose(data, [[.009, .008, .007], [1e-6, 2e-6, 3e-6]], atol=1e-15)
    digital = [(k, "digital" if k == "representation" else v) for k, v in params]
    dmeta, codes = decode(client.get(endpoint(record), query_string=digital))
    np.testing.assert_array_equal(codes, [[9, 8, 7], [1, 2, 3]])
    assert dmeta["units"] == ["count", "count"]


def test_overview_is_distinct_minmax_with_short_final_bin(example):
    client, record, _, params = example
    query = [(k, 0 if k == "start_sample" else 7 if k == "stop_sample" else
              "digital" if k == "representation" else v) for k, v in params] + [("bin_samples", 3)]
    meta, values = decode(client.get(endpoint(record, "overview"), query_string=query))
    assert meta["data_kind"] == "minmax_overview" and meta["shape"] == [2, 3, 2]
    assert meta["bin_start_samples"] == [0, 3, 6] and meta["bin_stop_samples"] == [3, 6, 7]
    np.testing.assert_array_equal(values, [[[10, 12], [7, 9], [6, 6]], [[-2, 0], [1, 3], [4, 4]]])


@pytest.mark.parametrize("field,value", [("stop_sample", 9), ("start_sample", -1), ("start_sample", 3.5),
                                        ("representation", "filtered"), ("channel_id", "unknown"),
                                        ("stop_sample", 1000000000), ("segment_id", "unknown")])
def test_invalid_or_gap_crossing_windows_never_yield_binary(example, field, value):
    client, record, _, params = example
    query = [(k, value if k == field else v) for k, v in params]
    response = client.get(endpoint(record), query_string=query)
    assert response.status_code in (400, 404)
    assert "error" in response.get_json()


def test_second_segment_keeps_gap_and_events_keep_source_clock(example):
    client, record, catalog, params = example
    span = catalog["streams"][0]["segments"][1]
    query = [(k, span["segment_id"] if k == "segment_id" else 8 if k == "start_sample" else
              9 if k == "stop_sample" else v) for k, v in params]
    meta, data = decode(client.get(endpoint(record), query_string=query))
    assert meta["recording_start_seconds"] == 5.25 and data.shape == (2, 1)
    events = client.get(endpoint(record, "events"), query_string={"offset": 0, "limit": 1}).get_json()["data"]
    assert events["total"] == 1 and len(events["events"]) == 1
    event = events["events"][0]
    assert event["text"] == "pulse" and event["clock_alignment"] == "source_local_unaligned"
    assert event["sample_mappings"][0]["sample_index"] == 1


def test_warm_reader_cache_rechecks_fingerprint_and_rejects_same_size_tampering(example, monkeypatch):
    client, record, _, params = example
    checks = []
    original = ephys_signals.inspect_snapshot

    def observed(*args):
        checks.append(args)
        return original(*args)

    monkeypatch.setattr(ephys_signals, "inspect_snapshot", observed)
    assert client.get(endpoint(record), query_string=params).status_code == 200
    assert checks == []  # Already SHA256-verified catalog, exact stat guard still matches.
    path = settings.storage_root / record["input_artifact"]["storage_path"] / "files" / "fixture.edf"
    before = path.stat()
    path.chmod(0o644)
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    response = client.get(endpoint(record), query_string=params)
    assert response.status_code == 409 and checks
    assert response.get_json()["error"]["code"] == "EPHYS_INPUT_CHANGED"


def test_mutation_during_window_is_discarded(example, monkeypatch):
    client, record, catalog, params = example
    entry = ephys_signals.verified_catalog(record["input_artifact_id"])
    reader = entry["readers"][catalog["streams"][0]["stream_id"]]
    original = reader.read_window

    def mutated(*args, **kwargs):
        data = original(*args, **kwargs)
        (entry["root"] / "files" / "unexpected.txt").write_bytes(b"changed input set")
        return data

    monkeypatch.setattr(reader, "read_window", mutated)
    assert client.get(endpoint(record), query_string=params).status_code == 409


def test_unreadable_source_is_explicit_without_hiding_readable_voltage(example):
    client, _, _, _ = example
    record, _ = import_recording({"participant_id": "example", "modality": "eeg", "label": "unsupported"},
                                [FileStorage(stream=io.BytesIO(b"not EDF"), filename="bad.edf")])
    catalog = client.get(endpoint(record, "signals")).get_json()["data"]
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
