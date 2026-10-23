"""Processed windows retain parent/sample identity without changing raw input."""
from copy import deepcopy
import io

import numpy as np
import pytest
from werkzeug.datastructures import FileStorage

from brainelecqc.db import get_db
from brainelecqc.config import settings
from brainelecqc.recordings import import_recording, get_recording, verify_recording
from brainelecqc.signals import describe_artifact, list_events
from brainelecqc.windows import samples, overview
from brainelecqc.storage import ImportConflict
from brainelecqc.derived import publish_voltage
from tests.edf_fixtures import tal, write_edf_bdf


@pytest.fixture
def processed(isolated_storage, tmp_path):
    source = write_edf_bdf(tmp_path / "raw.edf", [
        {"label": "same", "records": [[-2, -1, 0, 1], [2, 3, 4, 5]]},
        {"label": "same", "records": [[3, 2, 1, 0], [-1, -2, -3, -4]]}],
        reserved="EDF+D", annotation_records=[tal(2.5), tal(8.5)])
    rec, _ = import_recording({"participant_id": "example", "modality": "seeg", "label": "raw"},
                             [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="raw.edf")])
    parent = describe_artifact(rec["input_artifact_id"])
    raw_stream = parent["streams"][0]
    data = np.array([[-5, -3, -1, 1, 3, 5, 7, 9]], dtype="<f8") * 1e-6
    path = tmp_path / "bipolar.npy"
    np.save(path, data, allow_pickle=False)
    desc = {"stream_key": "bipolar", "parent_stream_id": raw_stream["stream_id"],
            "sampling_frequency_hz": 4., "sample_index_origin": 100, "sample_count": 8,
            "reference": "explicit_linear_montage", "channels": [{"raw_name": "A-B", "unit": "V",
                "source_channel_ids": [c["channel_id"] for c in raw_stream["channels"]],
                "weights": [1., -1.]}],
            "segments": [{"start_sample": 100 + s["start_sample"], "stop_sample": 100 + s["stop_sample"],
                "recording_start_seconds": s["recording_start_seconds"], "parent_segment_id": s["segment_id"],
                "parent_start_sample": s["start_sample"], "parent_stop_sample": s["stop_sample"],
                "clock_mapping": "identity"} for s in raw_stream["segments"]]}
    return rec, desc, path


def publish(example, descriptor=None):
    rec, desc, path = example
    return publish_voltage(rec["input_artifact_id"], [{"path": path, "stream": descriptor or desc}],
                           [{"operation": "linear_montage", "parameters": {"weights": [1., -1.]}}],
                           {"fixture_generator": "1"})


def test_processed_roundtrip_nonzero_origin_gaps_and_raw_are_independent(processed):
    rec, _, _ = processed
    result = publish(processed)
    catalog = describe_artifact(result["artifact_id"])
    stream = catalog["streams"][0]
    assert catalog["data_kind"] == "derived_samples" and catalog["scientific_validity"] == "pending"
    assert catalog["parent_artifact_id"] == rec["input_artifact_id"]
    assert catalog["parent_content_sha256"] == rec["input_artifact"]["content_sha256"]
    events = list_events(result["artifact_id"])
    assert events["event_scope"] == "parent_input_original_samples"
    assert events["parent_artifact_id"] == rec["input_artifact_id"]
    assert catalog["processing"][0]["operation"] == "linear_montage"
    assert stream["segments"][1]["recording_start_seconds"] == 8.5
    meta, values = samples(result["artifact_id"], stream["stream_id"], stream["segments"][1]["segment_id"],
                           stream["channel_order"], 105, 108, "physical")
    assert meta["recording_start_seconds"] == 8.75 and meta["sample_index_origin"] == 100
    np.testing.assert_array_equal(values, np.array([[5, 7, 9]]) * 1e-6)
    assert meta["data_kind"] == "derived_samples" and meta["software_versions"] == {"fixture_generator": "1"}
    over, data = overview(result["artifact_id"], stream["stream_id"], stream["segments"][1]["segment_id"],
                          stream["channel_order"], 104, 108, "physical", 3)
    assert over["source_data_kind"] == "derived_samples" and data.shape == (1, 2, 2)
    assert get_recording(rec["recording_id"])["input_artifact_id"] == rec["input_artifact_id"]
    assert verify_recording(rec["recording_id"])["valid"]
    raw = describe_artifact(rec["input_artifact_id"])
    assert raw["data_kind"] == "raw_samples" and raw["processing"] == []


def test_republish_never_overwrites_and_rejects_digital_or_gap(processed):
    first, second = publish(processed), publish(processed)
    assert first["artifact_id"] != second["artifact_id"] and first["storage_path"] != second["storage_path"]
    stream = describe_artifact(first["artifact_id"])["streams"][0]
    for stop, representation in [(108, "physical"), (104, "digital")]:
        with pytest.raises(ValueError):
            samples(first["artifact_id"], stream["stream_id"], stream["segments"][0]["segment_id"],
                    stream["channel_order"], 100, stop, representation)


@pytest.mark.parametrize("change", ["time", "unit", "parent_channel", "shape", "rate", "mapping", "nan"])
def test_bad_derived_contract_is_not_published(processed, change):
    rec, desc, path = processed
    desc = deepcopy(desc)
    if change == "time": desc["segments"][0]["recording_start_seconds"] = 0
    if change == "unit": desc["channels"][0]["unit"] = "unknown"
    if change == "parent_channel": desc["channels"][0]["source_channel_ids"][0] = "other-artifact:ch:0"
    if change == "shape": np.save(path, np.zeros((2, 8)), allow_pickle=False)
    if change == "rate": desc["sampling_frequency_hz"] = 1000
    if change == "mapping": desc["segments"][1]["parent_start_sample"] = 0
    if change == "nan": np.save(path, np.full((1, 8), np.nan), allow_pickle=False)
    with pytest.raises(ValueError): publish(processed, desc)
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM ephys_artifacts WHERE recording_id=?",
                            (rec["recording_id"],)).fetchone()["n"] == 1


def test_changed_parent_or_derived_input_is_never_accepted(processed):
    rec, _, _ = processed
    result = publish(processed)
    describe_artifact(result["artifact_id"])
    path = settings.storage_root / rec["input_artifact"]["storage_path"] / "files" / "raw.edf"
    path.chmod(0o644)
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(ImportConflict): describe_artifact(result["artifact_id"])


def test_same_filename_in_different_artifacts_does_not_share_signal_identity(processed):
    rec, _, path = processed
    source = path.parent / "raw.edf"
    again, _ = import_recording({"participant_id": "example", "modality": "seeg", "label": "raw"},
                               [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="raw.edf")])
    a, b = [describe_artifact(r["input_artifact_id"])["streams"][0] for r in (rec, again)]
    for key in ("stream_id", "source_id", "clock_id"):
        assert a[key] != b[key]
    assert set(a["channel_order"]).isdisjoint(b["channel_order"])


def test_identical_parent_restoration_refreshes_derived_guard(processed):
    rec, _, _ = processed
    result = publish(processed)
    stream = describe_artifact(result["artifact_id"])["streams"][0]
    path = settings.storage_root / rec["input_artifact"]["storage_path"] / "files" / "raw.edf"
    original = path.read_bytes()
    path.chmod(0o644)
    path.write_bytes(original)
    path.chmod(0o444)
    meta, data = samples(result["artifact_id"], stream["stream_id"], stream["segments"][0]["segment_id"],
                         stream["channel_order"], 100, 104, "physical")
    assert meta["data_kind"] == "derived_samples"
    np.testing.assert_array_equal(data, np.array([[-5, -3, -1, 1]]) * 1e-6)
