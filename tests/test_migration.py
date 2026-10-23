"""Independent stores remain portable without importing an external database."""
import io
import shutil
import sqlite3

import pytest
from werkzeug.datastructures import FileStorage

from brainelecqc import create_app
from brainelecqc.config import settings
from brainelecqc.db import init_db, get_db
from brainelecqc.recordings import import_recording, get_recording, verify_recording


def test_repeatable_initialization_preserves_identity():
    record, _ = import_recording({"label": "generated", "modality": "eeg"},
                                 [FileStorage(stream=io.BytesIO(b"example"), filename="example.edf")])
    with get_db() as connection:
        before = connection.execute("SELECT * FROM ephys_recordings").fetchall()
    init_db()
    init_db()
    with get_db() as connection:
        assert connection.execute("SELECT * FROM ephys_recordings").fetchall() == before
        assert connection.execute("PRAGMA integrity_check").fetchone()["integrity_check"] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert verify_recording(record["recording_id"])["valid"]


def test_store_can_move_without_original_paths(tmp_path, monkeypatch):
    record, _ = import_recording({"label": "portable", "modality": "ecog"},
                                 [FileStorage(stream=io.BytesIO(b"example"), filename="example.edf")])
    old = settings.storage_root
    moved = tmp_path / "moved-store"
    shutil.copytree(old, moved)
    shutil.rmtree(old)
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(moved))
    init_db()
    assert get_recording(record["recording_id"])["input_artifact_id"] == record["input_artifact_id"]
    assert verify_recording(record["recording_id"])["valid"]


def test_two_applications_use_separate_databases(tmp_path):
    first = create_app({"STORAGE_ROOT": tmp_path / "one", "TESTING": True})
    second = create_app({"STORAGE_ROOT": tmp_path / "two", "TESTING": True})
    with first.app_context():
        record, _ = import_recording({"label": "local", "modality": "seeg"},
                                    [FileStorage(stream=io.BytesIO(b"example"), filename="one.edf")])
    assert len(first.test_client().get("/api/v1/ephys/recordings").get_json()["data"]) == 1
    assert second.test_client().get("/api/v1/ephys/recordings").get_json()["data"] == []
    assert second.test_client().get("/api/v1/ephys/recordings/" + record["recording_id"]).status_code == 404


def test_foreign_database_is_rejected_without_altering_tables(tmp_path, monkeypatch):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    database = foreign / "brainelecqc.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES(1, 'keep')")
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(foreign))
    with pytest.raises(ValueError, match="different application"): init_db()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM unrelated").fetchall() == [(1, "keep")]
        assert connection.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone() == (1,)


def test_unknown_future_database_version_is_not_downgraded():
    with get_db() as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="Unsupported database version"): init_db()
    with get_db() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()["user_version"] == 99
