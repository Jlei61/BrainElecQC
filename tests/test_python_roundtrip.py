"""Real CLI import followed by Python catalog and calibrated sample access."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from brainelecqc.config import storage_context
from brainelecqc.recordings import get_recording, verify_recording
from brainelecqc.signals import describe_artifact
from brainelecqc.windows import samples
from tests.edf_fixtures import write_edf_bdf


def test_cli_import_to_python_samples_preserves_identity_order_and_units(tmp_path):
    source = write_edf_bdf(tmp_path / "input.edf", [
        {"label": "same", "records": [[-2, -1, 0, 1]]},
        {"label": "same", "records": [[12, 11, 10, 9]], "unit": "mV"}])
    store, unselected = tmp_path / "selected-store", tmp_path / "unselected-store"
    before = source.read_bytes()
    environment = dict(os.environ, BRAINELECQC_STORAGE_ROOT=str(unselected),
                       PYTHONPATH=str(Path(__file__).parents[1] / "src"))

    def cli(*arguments):
        result = subprocess.run([sys.executable, "-m", "brainelecqc", "--storage-root", str(store),
                                 *map(str, arguments)], env=environment, cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    imported = cli("import", source, "--modality", "eeg", "--label", "roundtrip")
    record = imported["recording"]
    assert not imported["reused"]
    listed = cli("list")
    assert [r["recording_id"] for r in listed] == [record["recording_id"]]
    assert listed[0]["input_artifact_id"] == record["input_artifact_id"]
    assert source.read_bytes() == before
    assert not unselected.exists()
    source.unlink()

    with storage_context(store):
        assert get_recording(record["recording_id"])["input_artifact_id"] == record["input_artifact_id"]
        assert verify_recording(record["recording_id"])["valid"]
        stream = describe_artifact(record["input_artifact_id"])["streams"][0]
        assert [c["raw_name"] for c in stream["channels"]] == ["same", "same"]
        ordered = [c["channel_id"] for c in reversed(stream["channels"])]
        metadata, values = samples(record["input_artifact_id"], stream["stream_id"],
                                   stream["segments"][0]["segment_id"], ordered, 0, 4, "physical")

    np.testing.assert_allclose(values, [[.012, .011, .010, .009], [-2e-6, -1e-6, 0, 1e-6]],
                               rtol=0, atol=1e-15)
    assert metadata["recording_id"] == record["recording_id"]
    assert metadata["artifact_id"] == record["input_artifact_id"]
    assert metadata["content_sha256"] == record["input_artifact"]["content_sha256"]
    assert metadata["channel_ids"] == ordered
    assert metadata["units"] == ["V", "V"] and metadata["shape"] == [2, 4]
    assert metadata["dtype"] == "<f8" and values.dtype == np.dtype("<f8")
    assert metadata["interval_convention"] == "half_open"
    assert metadata["start_sample"] == 0 and metadata["stop_sample"] == 4
    assert metadata["sampling_frequency_hz"] == 4
    assert metadata["recording_start_seconds"] == 0
