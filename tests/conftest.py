"""Every test uses an independent temporary store before importing application code."""
import pytest


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    root = tmp_path / "private-store"
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(root))
    from brainelecqc.db import init_db
    init_db()
    yield root
