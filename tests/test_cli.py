"""Real subprocess tests of CLI store selection and format declaration."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from brainelecqc.__main__ import _format_hints
from tests.edf_fixtures import write_edf_bdf


def run_cli(*arguments, env=None):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    environment.update(env or {})
    return subprocess.run([sys.executable, "-m", "brainelecqc", *map(str, arguments)],
                          env=environment, capture_output=True, text=True)


def signal(tmp_path, name="record.edf"):
    return write_edf_bdf(tmp_path / name, [{"label": "A1", "records": [[1, 2, 3, 4]]}])


def test_import_and_list_use_explicit_store_instead_of_environment(tmp_path):
    source = signal(tmp_path)
    root = tmp_path / "selected"
    other = tmp_path / "unselected"
    before = source.read_bytes()
    imported = run_cli("--storage-root", root, "import", source, "--modality", "eeg",
                       env={"BRAINELECQC_STORAGE_ROOT": str(other)})
    assert imported.returncode == 0, imported.stderr
    record = json.loads(imported.stdout)["recording"]
    assert record["label"] == source.name
    listed = run_cli("--storage-root", root, "list")
    assert listed.returncode == 0, listed.stderr
    assert [r["recording_id"] for r in json.loads(listed.stdout)] == [record["recording_id"]]
    assert source.read_bytes() == before
    assert not other.exists()


def test_missing_list_store_is_not_created(tmp_path):
    absent = tmp_path / "typo"
    result = run_cli("--storage-root", absent, "list")
    assert result.returncode == 2
    assert "No database" in result.stderr
    assert not absent.exists()


def test_single_file_hint_reaches_signal_factory(tmp_path, monkeypatch):
    source = signal(tmp_path)
    root = tmp_path / "hint-store"
    result = run_cli("--storage-root", root, "import", source, "--modality", "seeg", "--format-hint", "edf")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)["recording"]
    assert record["metadata"] == {"format_hints": {source.name: "edf"}}
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(root))
    from brainelecqc import signals
    captured, original = [], signals.open_reader
    def observed(path, source_id, format_hint=None):
        captured.append((path.name, format_hint))
        return original(path, source_id, format_hint)
    monkeypatch.setattr(signals, "open_reader", observed)
    catalog = signals.describe_artifact(record["input_artifact_id"])
    assert captured == [(source.name, "edf")]
    assert catalog["status"] == "readable"


def test_explicit_bundle_hint_maps_snapshot_relative_path(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "session").mkdir(parents=True)
    source = signal(bundle / "session")
    sidecar = bundle / "note.txt"
    sidecar.write_text("generated acquisition declaration")
    hints = _format_hints(["session/record.edf=edf"], [bundle])
    assert hints == {"session/record.edf": "edf"}
    result = run_cli("--storage-root", tmp_path / "bundle-store", "import", bundle,
                     "--modality", "ecog", "--format-hint", "session/record.edf=edf")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)["recording"]
    assert record["metadata"]["format_hints"] == hints
    assert {m["relative_path"] for m in record["manifest"]["files"]} == {"session/record.edf", "note.txt"}


@pytest.mark.parametrize("hints", [["missing.edf=edf"], ["../record.edf=edf"],
                                   ["record.edf=not-a-format"], ["record.edf=edf", "record.edf=bdf"]])
def test_invalid_hints_fail_before_creating_database(tmp_path, hints):
    source = signal(tmp_path)
    root = tmp_path / "invalid-store"
    arguments = ["--storage-root", root, "import", source, "--modality", "eeg"]
    for hint in hints:
        arguments.extend(["--format-hint", hint])
    result = run_cli(*arguments)
    assert result.returncode == 2
    assert not root.exists()


def test_bare_hint_is_refused_for_a_multi_file_bundle(tmp_path):
    one, two = signal(tmp_path, "one.edf"), signal(tmp_path, "two.edf")
    with pytest.raises(ValueError, match="RELATIVE_PATH"):
        _format_hints(["edf"], [one, two])


def test_ambiguous_filename_hint_is_explicit_vendor_mapping(tmp_path):
    # Only mapping behavior is checked here; this is not a vendor decoder fixture.
    source = tmp_path / "record.eeg"
    source.write_bytes(b"generated placeholder")
    assert _format_hints(["nihon"], [source]) == {"record.eeg": "nihon"}
