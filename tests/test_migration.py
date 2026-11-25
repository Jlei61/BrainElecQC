"""Independent stores remain portable without importing an external database."""
import asyncio
import io
import shutil
import sqlite3

import pytest
from brainelecqc.config import settings, storage_context
from brainelecqc.db import init_db, get_db
from brainelecqc.recordings import import_recording, get_recording, list_recordings, verify_recording
from tests.helpers import NamedInput


def test_repeatable_initialization_preserves_identity():
    record, _ = import_recording({"label": "generated", "modality": "eeg"},
                                 [NamedInput(stream=io.BytesIO(b"example"), filename="example.edf")])
    with get_db() as connection:
        before = connection.execute("SELECT * FROM ephys_recordings").fetchall()
    init_db()
    init_db()
    with get_db() as connection:
        assert connection.execute("SELECT * FROM ephys_recordings").fetchall() == before
        assert connection.execute("PRAGMA integrity_check").fetchone()["integrity_check"] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert verify_recording(record["recording_id"])["valid"]


def test_version_one_recordings_survive_seeg_qc_schema_upgrade():
    record, _ = import_recording({"label": "generated", "modality": "seeg"},
                                 [NamedInput(io.BytesIO(b"example"), "example.edf")])
    with get_db() as connection:
        connection.execute("DROP TABLE seeg_qc_audit")
        connection.execute("DROP TABLE seeg_qc_runs")
        connection.execute("PRAGMA user_version=1")
    init_db()
    init_db()
    assert get_recording(record["recording_id"])["input_artifact_id"] == record["input_artifact_id"]
    assert verify_recording(record["recording_id"])["valid"]
    with get_db() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()["user_version"] == 2
        assert connection.execute("SELECT count(*) AS count FROM seeg_qc_runs").fetchone()["count"] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_store_can_move_without_original_paths(tmp_path, monkeypatch):
    record, _ = import_recording({"label": "portable", "modality": "ecog"},
                                 [NamedInput(stream=io.BytesIO(b"example"), filename="example.edf")])
    old = settings.storage_root
    moved = tmp_path / "moved-store"
    shutil.copytree(old, moved)
    shutil.rmtree(old)
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(moved))
    init_db()
    assert get_recording(record["recording_id"])["input_artifact_id"] == record["input_artifact_id"]
    assert verify_recording(record["recording_id"])["valid"]


def test_nested_storage_contexts_keep_separate_databases_and_restore_after_error(tmp_path):
    original = settings.storage_root
    first, second = tmp_path / "one", tmp_path / "two"
    with storage_context(first):
        init_db()
        record, _ = import_recording({"label": "local", "modality": "seeg"},
                                    [NamedInput(stream=io.BytesIO(b"example"), filename="one.edf")])
        with pytest.raises(RuntimeError, match="generated interruption"):
            with storage_context(second):
                init_db()
                assert settings.storage_root == second.resolve()
                assert list_recordings() == []
                with pytest.raises(LookupError):
                    get_recording(record["recording_id"])
                raise RuntimeError("generated interruption")
        assert settings.storage_root == first.resolve()
        assert [r["recording_id"] for r in list_recordings()] == [record["recording_id"]]
        assert verify_recording(record["recording_id"])["valid"]
    assert settings.storage_root == original
    assert list_recordings() == []


def test_async_plugin_tasks_preserve_each_tasks_storage_across_await(tmp_path):
    original = settings.storage_root

    async def run_tasks():
        ready = [asyncio.Event(), asyncio.Event()]
        records = {}

        async def task(index):
            own_root = tmp_path / f"task-{index}"
            with storage_context(own_root):
                init_db()
                record, _ = import_recording({"label": f"task-{index}", "modality": "eeg"},
                                            [NamedInput(io.BytesIO(bytes([index])), "example.edf")])
                records[index] = record
                ready[index].set()
                await ready[1 - index].wait()
                await asyncio.sleep(0)
                assert settings.storage_root == own_root.resolve()
                assert [r["recording_id"] for r in list_recordings()] == [record["recording_id"]]
                assert verify_recording(record["recording_id"])["valid"]
                with pytest.raises(LookupError):
                    get_recording(records[1 - index]["recording_id"])
            assert settings.storage_root == original

        await asyncio.wait_for(asyncio.gather(task(0), task(1)), timeout=5)
        assert records[0]["recording_id"] != records[1]["recording_id"]

    asyncio.run(run_tasks())
    assert settings.storage_root == original
    assert list_recordings() == []


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
