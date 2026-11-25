"""A generated EDF exercises standalone scan, review, freeze and export."""

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from brainelecqc.config import settings
from brainelecqc.db import get_db
from brainelecqc.recordings import import_recording
from brainelecqc.seeg import export_run, freeze_run, get_audit, get_run, run_seeg_qc, save_review
from brainelecqc.storage import ImportConflict
from brainelecqc.windows import samples
from tests.edf_fixtures import tal, write_edf_bdf
from tests.test_cli import run_cli


@pytest.fixture
def recording(tmp_path):
    pace = 500
    duration = 14
    drift = np.random.default_rng(73).normal(0, 8, duration * pace)
    drift[2400:4100] += 1000
    line = 120 * np.sin(2 * np.pi * 50 * np.arange(duration * pace) / pace)
    line[-pace:] = 0
    return write_edf_bdf(
        tmp_path / "generated-seeg.edf",
        [
            {"label": "A1", "records": np.rint(drift).astype(int).reshape(duration, pace)},
            {"label": "A2", "records": np.rint(line).astype(int).reshape(duration, pace)},
            {"label": "ECG", "records": np.zeros((duration, 250), dtype=int)},
        ],
        reserved="EDF+D",
        annotation_records=[tal(onset) for onset in [*range(12), 19, 20]],
    )


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _cli(*args):
    result = run_cli("--storage-root", settings.storage_root, *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cli_seeg_scan_review_freeze_and_export(recording, tmp_path):
    original = recording.read_bytes()
    scan_dir = tmp_path / "scan"
    created = _cli("qc-seeg", recording, "--output", scan_dir)
    run_id = created["run_id"]
    run = json.loads((scan_dir / "qc_result.json").read_text())
    report = run["report"]
    assert run == _cli("show-seeg", run_id)
    assert run["provenance"]["content_sha256"] == report["catalog"]["content_sha256"]
    assert report["integrity"]["sources"][0]["sha256"] == _sha(original)
    by_rate = {stream["sampling_frequency_hz"]: stream for stream in report["catalog"]["streams"]}
    assert set(by_rate) == {250, 500}
    stream = by_rate[500]
    first, second = stream["segments"]
    assert (first["start_sample"], first["stop_sample"], first["recording_start_seconds"]) == (0, 6000, 0)
    assert (second["start_sample"], second["stop_sample"], second["recording_start_seconds"]) == (6000, 7000, 19)
    assert report["coverage"]["scanned_fraction"] == 1
    assert run["review"]["summary"]["kept_seconds"] == 0
    assert run["review"]["summary"]["unknown_seconds"] == 42

    a1, a2 = [channel["channel_id"] for channel in stream["channels"]]
    event, = report["waveform_events"]
    assert event["channel_id"] == a1 and event["start_sample"] == 2400
    assert event["recovery_status"] == "confirmed"
    assert {stage["kind"] for stage in event["stages"]} >= {"jump", "plateau", "recovery"}
    event_windows = [window["window_id"] for window in report["windows"] if event["event_id"] in window["waveform_event_ids"]]
    assert len(event_windows) == 2
    line_flags = [window["flags"] for window in report["windows"] if window["channel_id"] == a2]
    assert any("line_noise" in flags for flags in line_flags)

    metadata, values = samples(run["artifact_id"], stream["stream_id"], first["segment_id"], [a1], 2398, 2402, "physical")
    assert metadata["units"] == ["V"] and metadata["shape"] == [1, 4]
    assert np.max(np.abs(values[0, :2])) < 0.0001
    assert np.min(values[0, 2:]) > 0.0009

    review_path = scan_dir / "review.json"
    review = json.loads(review_path.read_text())
    review.update(reviewer="Synthetic reviewer", decisions=[
        {"channel_id": a1, "scope": "channel", "action": "keep", "reason": "Generated background"},
        {"channel_id": a1, "scope": "interval", "action": "exclude", "reason": "Generated step and recovery",
         "segment_id": first["segment_id"], "start_sample": event["start_sample"], "stop_sample": event["stop_sample"],
         "evidence_window_ids": event_windows},
    ])
    review_path.write_text(json.dumps(review))
    saved = _cli("review-seeg", run_id, "--review", review_path)
    assert saved["revision"] == 1
    stale = run_cli("--storage-root", settings.storage_root, "review-seeg", run_id, "--review", review_path)
    assert stale.returncode == 2 and "revision conflict" in stale.stderr
    confirmed = _cli("show-seeg", run_id)["review"]["summary"]
    excluded = (event["stop_sample"] - event["start_sample"]) / stream["sampling_frequency_hz"]
    assert confirmed["excluded_seconds"] == pytest.approx(excluded)
    assert confirmed["kept_seconds"] == pytest.approx(14 - excluded)
    assert confirmed["unknown_seconds"] == 28

    frozen = _cli("freeze-seeg", run_id, "--base-revision", 1, "--reviewer", "Synthetic reviewer")
    assert frozen["frozen"] and frozen["assessment"]["availability"] == "limited_usable"
    package, repeated = tmp_path / "frozen.zip", tmp_path / "frozen-copy.zip"
    _cli("export-seeg", run_id, "--output", package)
    _cli("export-seeg", run_id, "--output", repeated)
    assert package.read_bytes() == repeated.read_bytes()
    with zipfile.ZipFile(package) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["format"] == "brainelecqc_seeg_qc_zip_v1"
        assert set(bundle.namelist()) == set(manifest["files"]) | {"manifest.json"}
        assert all(_sha(bundle.read(path)) == checksum for path, checksum in manifest["files"].items())
        snapshot = json.loads(bundle.read("qc_result.json"))
        assert snapshot["frozen"] and snapshot["review"]["revision"] == 1
        stages = list(csv.DictReader(io.StringIO(bundle.read("waveform_events.csv").decode("utf-8-sig"))))
        assert json.loads(stages[0]["stages"]) == event["stages"]
    assert recording.read_bytes() == original


def test_python_review_audit_freeze_and_input_fingerprint(recording):
    imported, _ = import_recording({"modality": "seeg", "label": "generated"}, paths=[recording])
    run = run_seeg_qc(imported["input_artifact_id"])
    run_id = run["run_id"]
    channel = run["report"]["channel_map"][0]["channel_id"]
    with pytest.raises(ImportConflict, match="Freeze"):
        export_run(run_id)
    decision = {"channel_id": channel, "scope": "channel", "action": "keep", "reason": "Fixture background"}
    first = save_review(run_id, {"base_revision": 0, "reviewer": "Reviewer A", "decisions": [decision]})
    updated = save_review(run_id, {"base_revision": 1, "reviewer": "Reviewer B",
                                   "decisions": first["review"]["decisions"]})
    assert updated["review"]["decisions"] == first["review"]["decisions"]
    frozen = freeze_run(run_id, {"base_revision": 2, "reviewer": "Reviewer B"})
    assert frozen["frozen"] and get_run(run_id)["review"]["revision"] == 2
    with pytest.raises(ImportConflict, match="frozen"):
        save_review(run_id, {"base_revision": 2, "reviewer": "Reviewer B", "decisions": []})
    audit = get_audit(run_id)
    assert [(row["action"], row["revision"]) for row in audit] == [
        ("created", 0), ("scan_completed", 0), ("review_saved", 1), ("review_saved", 2), ("frozen", 2),
    ]
    assert audit[2]["snapshot"]["decisions"] == first["review"]["decisions"]
    payload = export_run(run_id)
    assert audit[-1]["snapshot"]["export_sha256"] == _sha(payload)

    with get_db() as connection:
        stored = connection.execute("SELECT storage_path FROM ephys_artifacts WHERE artifact_id=?",
                                    (run["artifact_id"],)).fetchone()
    copied = settings.storage_root / stored["storage_path"] / "files" / recording.name
    copied.chmod(0o600)
    copied.write_bytes(recording.read_bytes() + b"changed snapshot")
    with pytest.raises(ImportConflict, match="fingerprint mismatch"):
        export_run(run_id)
