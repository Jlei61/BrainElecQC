"""Known bytes test identity/integrity; they are not biological validation."""
import io
import json
import sqlite3
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from werkzeug.datastructures import FileStorage

from brainelecqc import create_app, recordings
from brainelecqc.config import settings
from brainelecqc.db import get_db
from brainelecqc.storage import ImportConflict


def upload(data=b"recording bytes", name="same.edf"):
    return FileStorage(stream=io.BytesIO(data), filename=name)


def payload(**overrides):
    return {"participant_id": "example", "label": "same", "modality": "eeg", **overrides}


def imported(data=b"recording bytes", **overrides):
    return recordings.import_recording(payload(**overrides), uploads=[upload(data)])[0]


def root(record):
    return settings.storage_root / record["input_artifact"]["storage_path"]


def zip_upload(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, contents in members:
            archive.writestr(name, contents)
    stream.seek(0)
    return FileStorage(stream=stream, filename="recording.zip")


def assert_no_partial():
    assert recordings.list_recordings() == []
    assert not settings.inputs_root.exists() or not list(settings.inputs_root.iterdir())


def test_same_name_is_not_identity_and_raw_is_unreviewed():
    first, second = imported(), imported()
    assert first["recording_id"] != second["recording_id"]
    assert first["input_artifact_id"] != second["input_artifact_id"]
    assert first["input_artifact"]["content_sha256"] == second["input_artifact"]["content_sha256"]
    assert first["scientific_validity"] == "pending"
    assert first["status"] == "stored_unreviewed"
    assert len(recordings.list_recordings("example")) == 2
    assert recordings.list_recordings("different") == []


def test_idempotency_is_bound_to_bytes_and_metadata():
    first = imported(import_key="request-1")
    second, reused = recordings.import_recording(payload(import_key="request-1"), [upload()])
    assert reused and first["recording_id"] == second["recording_id"]
    for changes in ({"label": "changed"}, {"modality": "seeg"}, {"participant_id": "other"},
                    {"metadata": {"reference": "changed"}}):
        with pytest.raises(ImportConflict):
            imported(import_key="request-1", **changes)
    with pytest.raises(ImportConflict):
        imported(b"different bytes", import_key="request-1")
    assert len(recordings.list_recordings()) == 1


def test_concurrent_retry_publishes_one_snapshot():
    def submit(_):
        return recordings.import_recording(payload(import_key="concurrent"), [upload()])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert sorted(reused for _, reused in results) == [False, True]
    assert len({r["recording_id"] for r, _ in results}) == 1
    assert len(list(settings.inputs_root.iterdir())) == 1


def test_local_copy_survives_original_replacement_and_deletion(tmp_path):
    source = tmp_path / "source.edf"
    source.write_bytes(b"original")
    record, _ = recordings.import_recording(payload(), paths=[source])
    copied = root(record) / "files" / source.name
    assert source.stat().st_ino != copied.stat().st_ino
    source.write_bytes(b"replaced")
    source.unlink()
    assert copied.read_bytes() == b"original"
    assert recordings.verify_recording(record["recording_id"])["valid"]
    assert str(tmp_path) not in json.dumps(record["manifest"])


@pytest.mark.parametrize("change", ["bytes", "add", "delete"])
def test_all_sources_rechecked_after_final_copy(tmp_path, monkeypatch, change):
    first = tmp_path / "first"
    first.mkdir()
    member = first / "one.edf"
    member.write_bytes(b"first")
    second = tmp_path / "two.edf"
    second.write_bytes(b"second")
    original = recordings.SnapshotWriter.add_path
    def mutate(writer, source, selected_root):
        if source == second:
            if change == "bytes": member.write_bytes(b"FIRST")
            elif change == "add": (first / "event.tsv").write_bytes(b"new")
            else: member.unlink()
        return original(writer, source, selected_root)
    monkeypatch.setattr(recordings.SnapshotWriter, "add_path", mutate)
    with pytest.raises(ImportConflict):
        recordings.import_recording(payload(), paths=[first, second])
    assert_no_partial()


@pytest.mark.parametrize("change", ["bytes", "add", "delete", "manifest"])
def test_verification_detects_external_tampering(change):
    record = imported(import_key="immutable")
    member = root(record) / "files" / "same.edf"
    if change == "bytes":
        member.chmod(0o644)
        member.write_bytes(b"Recording bytes")
    elif change == "add": (member.parent / "extra").write_bytes(b"extra")
    elif change == "delete": member.unlink()
    else:
        manifest = root(record) / "manifest.json"
        manifest.chmod(0o644)
        manifest.write_text("{}")
    assert not recordings.verify_recording(record["recording_id"])["valid"]
    with pytest.raises(ImportConflict): imported(import_key="immutable")


def test_each_zip_member_is_fingerprinted():
    a, _ = recordings.import_recording(payload(), [zip_upload([("s/a.edf", b"one"), ("s/b.tsv", b"two")])])
    b, _ = recordings.import_recording(payload(), [zip_upload([("s/a.edf", b"one"), ("s/b.tsv", b"TWO")])])
    assert a["input_artifact"]["content_sha256"] != b["input_artifact"]["content_sha256"]
    assert a["input_artifact"]["file_count"] == 2
    assert all(len(m["sha256"]) == 64 for m in a["manifest"]["files"])


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../../escape", "..\\escape"])
def test_paths_cannot_escape_upload_snapshot(name):
    for source in (upload(name=name), zip_upload([(name, b"escape")])):
        with pytest.raises(ValueError): recordings.import_recording(payload(), [source])
        assert_no_partial()


def test_links_are_not_followed(tmp_path):
    target = tmp_path / "target.edf"
    target.write_bytes(b"outside")
    link = tmp_path / "link.edf"
    link.symlink_to(target)
    with pytest.raises(ValueError): recordings.import_recording(payload(), paths=[link])
    info = zipfile.ZipInfo("link.edf")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError): recordings.import_recording(payload(), [zip_upload([(info, b"../outside")])])
    assert_no_partial()


@pytest.mark.parametrize("names", [("same.edf", "same.edf"), ("a.edf", "A.edf"), ("a", "a/b")])
def test_ambiguous_member_names_fail(names):
    with pytest.raises((ValueError, OSError)):
        recordings.import_recording(payload(), [zip_upload([(n, b"bytes") for n in names])])
    assert_no_partial()


def test_stream_failure_and_database_failure_roll_back():
    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell(): raise OSError("generated failure")
            return super().read(size)
    with pytest.raises(OSError):
        recordings.import_recording(payload(), [FileStorage(stream=Broken(b"partial"), filename="broken.edf")])
    assert_no_partial()
    with get_db() as connection:
        connection.execute("CREATE TRIGGER reject_artifact BEFORE INSERT ON ephys_artifacts "
                           "BEGIN SELECT RAISE(ABORT, 'generated failure'); END")
    with pytest.raises(sqlite3.IntegrityError): imported()
    assert_no_partial()


def test_duplicate_labels_preserved_but_duplicate_ids_rejected():
    metadata = {"channels": [{"channel_id": "one", "raw_name": "A1"}, {"channel_id": "two", "raw_name": "A1"}]}
    assert imported(metadata=metadata)["metadata"] == metadata
    metadata["channels"][1]["channel_id"] = "one"
    with pytest.raises(ValueError): imported(metadata=metadata)


@pytest.mark.parametrize("changes", [{"modality": "unknown"}, {"metadata": {"value": float("nan")}},
                                      {"metadata": {"format_hints": ["edf"]}}, {"server_path": "input.edf"}])
def test_unknown_or_ambiguous_declarations_are_not_silently_guessed(changes):
    with pytest.raises(ValueError): imported(**changes)
    assert_no_partial()


def test_http_only_accepts_uploads_and_roundtrips_identity():
    client = create_app({"TESTING": True}).test_client()
    def post(**overrides):
        return client.post("/api/v1/ephys/recordings", data={
            "metadata": json.dumps(payload(import_key="http", **overrides)),
            "files": (io.BytesIO(b"raw"), "record.edf")})
    first = post()
    assert first.status_code == 201
    rid = first.get_json()["data"]["recording_id"]
    assert post().get_json()["meta"]["reused"]
    assert post(label="changed").status_code == 409
    assert client.get(f"/api/v1/ephys/recordings/{rid}").status_code == 200
    assert client.post(f"/api/v1/ephys/recordings/{rid}/verify").get_json()["data"]["valid"]
    assert len(client.get("/api/v1/ephys/recordings").get_json()["data"]) == 1
    assert client.post("/api/v1/ephys/recordings", json={"paths": ["/private/input.edf"]}).status_code == 400
    assert client.get("/api/v1/ephys/recordings/missing").status_code == 404
