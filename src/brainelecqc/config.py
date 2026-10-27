"""Task-local storage selection for Python callers and host plugins."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

_storage_root = ContextVar("brainelecqc_storage_root", default=None)


@contextmanager
def storage_context(root):
    """Select a store for this task, restoring the previous selection on exit.

Nested contexts and asyncio tasks keep their own selection. Callers starting
a worker thread or process should enter its storage context in that worker.
"""
    if root is None or isinstance(root, str) and not root.strip():
        raise ValueError("Provide a storage directory")
    resolved = Path(root).expanduser().resolve()
    token = _storage_root.set(resolved)
    try:
        yield resolved
    finally:
        _storage_root.reset(token)


class Settings:
    @property
    def storage_root(self):
        return Path(_storage_root.get() or os.environ.get("BRAINELECQC_STORAGE_ROOT") or
                    Path.home() / ".local" / "share" / "brainelecqc").expanduser().resolve()

    @property
    def db_path(self):
        return self.storage_root / "brainelecqc.sqlite3"

    @property
    def inputs_root(self):
        return self.storage_root / "inputs"

    @property
    def derivatives_root(self):
        return self.storage_root / "derivatives"


settings = Settings()
