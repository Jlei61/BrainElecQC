"""Small independent SQLite store with transactional connections."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .config import settings


def current_ts():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dict_row(cursor, row):
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


@contextmanager
def get_db():
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.db_path, timeout=30)
    connection.row_factory = _dict_row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db():
    with get_db() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()["user_version"]
        if version not in (0, 1):
            raise ValueError("Unsupported database version; use the matching application version")
        # Reject databases from an unrelated application rather than altering them.
        tables = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if (version == 0 and tables) or tables - {"ephys_recordings", "ephys_artifacts"}:
            raise ValueError("Storage database belongs to a different application")
        connection.executescript(Path(__file__).with_name("schema.sql").read_text())
