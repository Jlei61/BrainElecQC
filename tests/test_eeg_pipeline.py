"""EEG CLI, immutable reader boundary, output and original-clock integration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brainelecqc.eeg.export import export_eeg_qc
from brainelecqc.eeg.input import read_eeg_input
from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.recordings import import_recording, verify_recording
from brainelecqc.signals import describe_artifact
from tests.edf_fixtures import tal, write_edf_bdf


def _source(tmp_path, *, units=("uV", "uV"), clipped=True, discontinuous=False, mixed_rate=False):
    fs, seconds = 256, 8
    t = np.arange(fs * seconds) / fs
    first = np.rint(1000 * np.sin(2 * np.pi * 10 * t)).astype(int)
    second = np.rint(700 * np.sin(2 * np.pi * 12 * t + .2)).astype(int)
    if clipped:
        first[600:650] = 32767
    auxiliary = np.rint(200 * np.sin(2 * np.pi * 3 * t)).astype(int)
    def ch(label, values, unit="uV"):
        lo, hi = (-.8, .8) if unit == "mV" else (-800, 800)
        return {"label": label, "records": values.reshape(seconds, -1).tolist(), "unit": unit,
                "physical_min": lo, "physical_max": hi}
    channels = [ch("EEG Oz", first, units[0]), ch("ECG", auxiliary),
                ch("EEG Cz", second, units[1]), ch("EOG", -auxiliary),
                ch("EMG chin", auxiliary), ch("POL SpO2", np.zeros_like(first))]
    if mixed_rate:
        channels.append(ch("EEG slow", second[::2]))
    stamps = [0, 1, 2, 3, 10, 11, 12, 13]
    options = {"reserved": "EDF+D", "annotation_records": [tal(s) for s in stamps]} if discontinuous else {}
    source = write_edf_bdf(tmp_path / "recording.edf", channels, **options)
    return source, np.stack([first, second]), fs


def _import(source):
    record, _ = import_recording({"modality": "eeg", "label": "QC integration"}, paths=[source])
    return record, record["input_artifact_id"]


def _cli(tmp_path, source, output, *extra):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    return subprocess.run([sys.executable, "-m", "brainelecqc", "--storage-root", str(tmp_path / "cli-store"),
                           "qc-eeg", str(source), "--output", str(output), "--no-figures", *map(str, extra)],
                          cwd=tmp_path, env=environment, capture_output=True, text=True)


def _manifest_hashes_match(output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {p.name for p in output.iterdir()} - {"manifest.json"}
    for name, fingerprint in manifest["files"].items():
        content = (output / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == fingerprint["sha256"]
        assert len(content) == fingerprint["size_bytes"]
    return manifest


def test_cli_full_numerical_path_preserves_source_and_exports_bound_mask(tmp_path):
    source, digital, fs = _source(tmp_path)
    source_before = source.read_bytes()
    output = tmp_path / "qc-result"
    completed = _cli(tmp_path, source, output, "--duration", 8)
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    manifest = _manifest_hashes_match(output)
    assert response["report"] == str(output/"review_report.md")
    assert {"review_report.md", "review_items.md"} <= manifest["files"].keys()
    assert "![" not in (output/"review_report.md").read_text()
    assert response["run_id"] == manifest["run_id"]
    assert manifest["unit"] == "V" and manifest["axes"] == ["channel", "sample"]
    assert manifest["start_sample"] == 0 and manifest["stop_sample"] == fs * 8
    assert manifest["source_sha256"] == hashlib.sha256(source_before).hexdigest()
    assert manifest["selection"] == {"eeg_indices": [0, 2], "reference_indices": {"ecg": 1, "eog": 3, "emg": 4}}
    raw = np.load(output / "review_signals.npz")["raw_data_volts"]
    mask = np.load(output / "keep_mask.npy")
    masked = np.load(output / "masked_data_volts.npy")
    expected_mask = np.ones_like(raw, dtype=bool)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(masked[mask], raw[mask])
    assert np.isnan(masked[~mask]).all()
    assert source.read_bytes() == source_before
    decisions = json.loads((output / "decisions.json").read_text())
    assert decisions["automatic_exclusions"] == []
    assert decisions["decision_mode"] == manifest["decision_mode"] == "review_only"
    plateau = next(x for x in decisions["review_queue"] if x["kind"] == "calibration_endpoint_plateau")
    assert (plateau["start_sample"], plateau["stop_sample"]) == (600, 650)
    assert plateau["threshold_basis"] == "project_screening_parameter"
    assert all(json.loads((output / "verification.json").read_text())[k]
               for k in ("retained_values_exact", "excluded_values_nan", "original_clock_preserved"))


def test_selection_preserves_order_voltage_scale_and_explicit_auxiliaries(tmp_path):
    source, digital, fs = _source(tmp_path, units=("mV", "uV"))
    record, artifact = _import(source)
    result = run_eeg_qc(artifact, eeg_indices=[2, 0], reference_indices={"ecg": 1, "eog": 3, "emg": 4})
    assert [c["source_index"] for c in result["channels"]] == [2, 0]
    assert [c["physical_unit"] for c in result["channels"]] == ["uV", "mV"]
    expected = (digital[[1, 0]] * (1600 / 65535) + 800 / 65535) * 1e-6
    np.testing.assert_allclose(result["raw_data_volts"], expected, rtol=0, atol=2e-19)
    assert result["manifest"]["recording_id"] == record["recording_id"]
    assert result["manifest"]["artifact_id"] == artifact
    assert result["manifest"]["channel_ids"] == [c["channel_id"] for c in result["channels"]]
    assert result["features"]["reference_channels"]["ecg"]["source_index"] == 1
    assert result["keep_mask"].all()
    np.testing.assert_array_equal(result["masked_data_volts"], result["raw_data_volts"])
    assert verify_recording(record["recording_id"])["valid"]


def test_default_selection_keeps_flat_auxiliary_out_of_eeg_exclusions(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    assert result["manifest"]["selection"]["eeg_indices"] == [0, 2]
    assert result["decisions"]["automatic_exclusions"] == []
    assert result["keep_mask"].all()


def test_discontinuous_segment_preserves_source_clock_and_stored_sample_index(tmp_path):
    source, _, fs = _source(tmp_path, discontinuous=True, clipped=False)
    _, artifact = _import(source)
    stream = describe_artifact(artifact)["streams"][0]
    with pytest.raises(ValueError, match="segment_id"):
        run_eeg_qc(artifact)
    segment = stream["segments"][1]
    result = run_eeg_qc(artifact, segment_id=segment["segment_id"], start_seconds=10.5, duration_seconds=2)
    assert result["features"]["start_sample"] == round(4.5 * fs)
    assert result["features"]["stop_sample"] == round(6.5 * fs)
    assert result["features"]["start_seconds"] == 10.5
    assert result["features"]["stop_seconds"] == 12.5
    assert result["manifest"]["segment_id"] == segment["segment_id"]
    assert all(row["start_seconds"] == 10.5 and row["stop_seconds"] == 12.5
               for row in result["retained_intervals"])
    with pytest.raises(ValueError, match="inside the continuous segment"):
        run_eeg_qc(artifact, segment_id=stream["segments"][0]["segment_id"], duration_seconds=5)


def test_multirate_record_requires_explicit_stream_and_selects_native_rate(tmp_path):
    source, _, _ = _source(tmp_path, mixed_rate=True)
    _, artifact = _import(source)
    with pytest.raises(ValueError, match="stream_id"):
        run_eeg_qc(artifact)
    streams = describe_artifact(artifact)["streams"]
    slow = next(s for s in streams if s["sampling_frequency_hz"] == 128)
    result = run_eeg_qc(artifact, stream_id=slow["stream_id"])
    assert result["manifest"]["selection"]["eeg_indices"] == [6]
    assert result["features"]["sampling_rate_hz"] == 128
    assert result["raw_data_volts"].shape == (1, 1024)
    assert result["references"] == {}


def test_unknown_eeg_unit_requires_resolution(tmp_path):
    source, _, _ = _source(tmp_path, units=("Ohm", "uV"))
    _, artifact = _import(source)
    with pytest.raises(ValueError, match="units to volts"):
        run_eeg_qc(artifact)


def test_features_reproduce_and_run_identity_is_distinct(tmp_path):
    source, _, _ = _source(tmp_path)
    _, artifact = _import(source)
    first, second = run_eeg_qc(artifact), run_eeg_qc(artifact)
    assert first["features"] == second["features"]
    assert first["decisions"] == second["decisions"]
    assert first["manifest"]["run_id"] != second["manifest"]["run_id"]
    assert first["manifest"]["artifact_id"] == second["manifest"]["artifact_id"]
    np.testing.assert_array_equal(first["keep_mask"], second["keep_mask"])


def test_export_preserves_existing_result_and_rejects_repository_output(tmp_path):
    source, _, _ = _source(tmp_path)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    output = tmp_path / "export"
    export_eeg_qc(result, output, figures=False)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="new output directory"):
        export_eeg_qc(result, output, figures=False)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    with pytest.raises(ValueError, match="outside the Git checkout"):
        export_eeg_qc(result, checkout / "export", figures=False)
    assert not (checkout / "export").exists()


@pytest.mark.parametrize("selection", [{"eeg_indices": []}, {"eeg_indices": [0, 0]},
    {"eeg_indices": [True]}, {"reference_indices": {"ecg": 0}}, {"reference_indices": {"unknown": 1}},
    {"start_seconds": float("nan")}, {"duration_seconds": 0}, {"start_seconds": 9}])
def test_invalid_selection_has_actionable_error(tmp_path, selection):
    source, _, _ = _source(tmp_path)
    _, artifact = _import(source)
    with pytest.raises(ValueError):
        read_eeg_input(artifact, **selection)


@pytest.mark.parametrize("config", [
    {"selection": None}, {"selection": []}, {"selection": {"unknown": 1}},
    {"feature_config": {"unknown": 1}}, {"policy_config": {"unknown": 1}},
    {"feature_config": {"welch_seconds": 0}}, {"policy_config": {"padding_seconds": -1}},
    {"feature_config": []}, {"policy_config": []},
])
def test_cli_invalid_config_exits_cleanly_without_result(tmp_path, config):
    source, _, _ = _source(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "invalid-result"
    completed = _cli(tmp_path, source, output, "--config", config_path)
    assert completed.returncode == 2
    assert "brainelecqc:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()
    assert not (tmp_path / "cli-store").exists()
