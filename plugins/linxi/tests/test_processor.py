"""Real Linxi host integration using generated EDF bytes in temporary directories.

Install the plugin and Linxi test environment before running this module. Host
classes, registration, YAML loading and task execution are exercised directly.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest
import yaml

# Linxi establishes its numerical runtime before importing NumPy.
from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner
from linxi.processor import DefaultProcessor, PROCESS_STAGES
from linxi.processor.registry import get_registered_processors

import numpy as np
from linxi_brainelecqc import BrainElecQCProcessor
from brainelecqc.config import settings, storage_context
from brainelecqc.recordings import get_recording, verify_recording
from tests.test_eeg_pipeline import _source


@pytest.fixture(autouse=True)
def isolated_caller_store(tmp_path, monkeypatch):
    """The plugin must use its task directory, not the caller's default store."""
    outside = tmp_path / "unrelated-caller-store"
    monkeypatch.setenv("BRAINELECQC_STORAGE_ROOT", str(outside))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    yield outside


def context(source, work_dir):
    value = LinxiContext(file_create_date=datetime.now(timezone.utc))
    value.run_state.input_path = str(source) if source is not None else None
    value.run_state.work_dir = work_dir
    return value


def make_source(tmp_path, **options):
    directory = tmp_path / "input"
    directory.mkdir()
    return _source(directory, **options)


def guards(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino,
            hashlib.sha256(path.read_bytes()).hexdigest())


def contents(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def verify_bundle(summary, work_dir):
    required = {"schema_version", "recording_id", "input_artifact_id", "run_id", "data_sha256",
                "bundle_dir", "manifest_path", "report_path", "score_path", "auto_total",
                "assessed_max", "pending_max", "review_status", "review_id", "figures", "source_reused"}
    assert required <= summary.keys()
    assert summary["schema_version"] == 1
    assert summary["review_status"] == "pending" and summary["review_id"] is None
    output = Path(summary["bundle_dir"])
    assert output == work_dir / "brainelecqc" / "runs" / summary["run_id"]
    assert Path(summary["manifest_path"]) == output / "manifest.json"
    assert Path(summary["report_path"]) == output / "review_report.md"
    assert Path(summary["score_path"]) == output / "quality_score.json"
    manifest = json.loads(Path(summary["manifest_path"]).read_text())
    assert manifest["recording_id"] == summary["recording_id"]
    assert manifest["artifact_id"] == summary["input_artifact_id"]
    assert manifest["run_id"] == summary["run_id"]
    assert manifest["data_sha256"] == summary["data_sha256"]
    assert manifest["review_status"] == "pending"
    assert set(manifest["files"]) == {p.name for p in output.iterdir()} - {"manifest.json"}
    for name, fingerprint in manifest["files"].items():
        payload = (output / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == fingerprint["sha256"]
        assert len(payload) == fingerprint["size_bytes"]
    score = json.loads(Path(summary["score_path"]).read_text())
    assert summary["score_100"] == score["score_100"]
    assert summary["assessment_coverage"] == score["assessment_coverage"]
    assert Path(summary["score_report_path"]).is_file()
    assert "EEG 自动 QC 报告" in Path(summary["report_path"]).read_text()
    for key in ("auto_total", "assessed_max", "pending_max"):
        assert type(summary[key]) is int
        assert summary[key] == score[key]
    assert 0 <= summary["auto_total"] <= summary["assessed_max"]
    assert summary["assessed_max"] + summary["pending_max"] == 100
    assert summary["auto_total"] == sum(r["auto_points"] for r in score["rows"] if r["auto_points"] is not None)
    return output, manifest


def test_registration_uses_real_quality_stage_and_validation_contract(tmp_path):
    assert issubclass(BrainElecQCProcessor, DefaultProcessor)
    assert BrainElecQCProcessor in get_registered_processors(PROCESS_STAGES.QUALITY)
    assert BrainElecQCProcessor.VALIDATION_SKIP is True
    processor = BrainElecQCProcessor(figures=False, name="quality-test")
    assert processor.name == "quality-test"
    assert list(tmp_path.iterdir()) == []


def test_yaml_task_runner_preserves_voltage_order_clock_and_manifest(tmp_path, isolated_caller_store):
    source, digital, fs = make_source(tmp_path, units=("mV", "uV"))
    before = guards(source)
    work_dir = tmp_path / "host-work"
    params = {"figures": False, "title": "Temporary EEG QC",
              "qc_config": {"selection": {"eeg_indices": [2, 0], "start_seconds": 1.25,
                                            "duration_seconds": 2.5},
                            "method_config": {"comparison_groups": ["common", "common"]}},
              "recording_info": {"label": "generated EEG", "participant_id": "temporary",
                                 "session_id": "generated-session", "metadata": {"reference": "as_recorded"}}}
    definition_path = tmp_path / "pipeline.yml"
    definition_path.write_text(yaml.safe_dump({"name": "temporary-qc", "linxi_plugin": ["linxi_brainelecqc"],
        "steps": [{"stage": "quality", "processor_name": "BrainElecQCProcessor", "params": params}]}))
    definition = PipelineDefinition.from_file(definition_path)
    state = context(source, work_dir)
    caller_root = settings.storage_root
    results = TaskRunner(definition).run(state, skip_dry_run=False, allow_errors=False)
    assert results == [state] and not state.run_state.failed
    assert len(state.history) == 1
    summary = state.get_metric("brainelecqc")
    output, manifest = verify_bundle(summary, work_dir)
    assert summary["source_reused"] is False and summary["figures"] == {}
    assert state.get_metric("brainelecqc_status")["status"] == "completed"
    assert state.get_metric("brainelecqc_status")["phase"] == "complete"
    assert manifest["unit"] == "V" and manifest["axes"] == ["channel", "sample"]
    assert manifest["start_sample"] == 320 and manifest["stop_sample"] == 960
    assert manifest["start_seconds"] == 1.25 and manifest["sampling_rate_hz"] == fs
    assert manifest["clock_alignment"] == "source_local_unaligned"
    channels = json.loads((output / "channels.json").read_text())
    assert [c["source_index"] for c in channels] == [2, 0]
    assert [c["raw_name"] for c in channels] == ["EEG Cz", "EEG Oz"]
    assert [c["physical_unit"] for c in channels] == ["uV", "mV"]
    with np.load(output / "review_signals.npz", allow_pickle=False) as archive:
        raw = archive["raw_data_volts"]
    expected = (digital[[1, 0], 320:960] * (1600 / 65535) + 800 / 65535) * 1e-6
    np.testing.assert_allclose(raw, expected, rtol=0, atol=2e-19)
    np.testing.assert_array_equal(np.load(output / "masked_data_volts.npy", allow_pickle=False), raw)
    assert np.load(output / "keep_mask.npy", allow_pickle=False).all()
    assert guards(source) == before
    assert settings.storage_root == caller_root and not isolated_caller_store.exists()
    with storage_context(work_dir / "brainelecqc" / "store"):
        record = get_recording(summary["recording_id"])
        assert record["metadata"] == params["recording_info"]["metadata"]
        assert record["modality"] == "eeg"
        assert verify_recording(record["recording_id"])["valid"]


def test_validation_context_never_creates_qc_or_updates_metrics(tmp_path):
    state = context(None, tmp_path / "dry-run")
    state.run_state.validating = True
    state.put_metric("unrelated", {"value": "keep"})
    result = BrainElecQCProcessor(figures=False)(state)
    assert result is state
    assert state.get_metric("brainelecqc") is None
    assert state.get_metric("brainelecqc_status") is None
    assert state.get_metric("unrelated") == {"value": "keep"}
    assert not (state.run_state.work_dir / "brainelecqc").exists()


def test_repeat_reuses_source_identity_and_preserves_previous_run(tmp_path):
    source, _, _ = make_source(tmp_path)
    state = context(source, tmp_path / "repeat")
    processor = BrainElecQCProcessor(figures=False)
    processor(state)
    first = deepcopy(state.get_metric("brainelecqc"))
    before = contents(Path(first["bundle_dir"]))
    processor(state)
    second = state.get_metric("brainelecqc")
    assert first["recording_id"] == second["recording_id"]
    assert first["input_artifact_id"] == second["input_artifact_id"]
    assert first["data_sha256"] == second["data_sha256"]
    assert first["run_id"] != second["run_id"] and second["source_reused"] is True
    assert contents(Path(first["bundle_dir"])) == before
    verify_bundle(second, state.run_state.work_dir)


def test_tasks_are_isolated_and_restore_nested_storage_context(tmp_path):
    source, _, _ = make_source(tmp_path)
    first = context(source, tmp_path / "task-one")
    second = context(source, tmp_path / "task-two")
    external = tmp_path / "outer-context"
    with storage_context(external):
        for state in (first, second):
            BrainElecQCProcessor(figures=False)(state)
            assert settings.storage_root == external
    a, b = first.get_metric("brainelecqc"), second.get_metric("brainelecqc")
    assert a["recording_id"] != b["recording_id"]
    assert a["input_artifact_id"] != b["input_artifact_id"]
    assert not external.exists()
    assert Path(a["bundle_dir"]).is_relative_to(first.run_state.work_dir)
    assert Path(b["bundle_dir"]).is_relative_to(second.run_state.work_dir)


def test_recording_metadata_is_part_of_import_identity(tmp_path):
    source, _, _ = make_source(tmp_path)
    state = context(source, tmp_path / "metadata-identity")
    BrainElecQCProcessor(figures=False, recording_info={"label": "first"})(state)
    first = deepcopy(state.get_metric("brainelecqc"))
    BrainElecQCProcessor(figures=False, recording_info={"label": "second"})(state)
    second = state.get_metric("brainelecqc")
    assert second["recording_id"] != first["recording_id"]
    assert not second["source_reused"]
    assert Path(first["manifest_path"]).is_file()


def test_failure_clears_old_success_reference_and_preserves_earlier_bundle(tmp_path):
    source, _, _ = make_source(tmp_path)
    state = context(source, tmp_path / "failure")
    processor = BrainElecQCProcessor(figures=False)
    processor(state)
    first = deepcopy(state.get_metric("brainelecqc"))
    before = contents(Path(first["bundle_dir"]))
    state.put_metric("unrelated", {"value": "keep"})
    source.write_bytes(b"malformed changed input")
    caller_root = settings.storage_root
    definition = PipelineDefinition(name="failing-qc", linxi_plugin=["linxi_brainelecqc"],
        steps=[{"stage": "quality", "processor_name": "BrainElecQCProcessor", "params": {"figures": False}}])
    with pytest.raises((ValueError, OSError)):
        TaskRunner(definition).run(state, skip_dry_run=False, allow_errors=False)
    assert state.run_state.failed and state.run_state.error
    assert state.get_metric("brainelecqc") is None
    failure = state.get_metric("brainelecqc_status")
    assert failure["status"] == "failed" and failure["error"]
    assert failure["phase"] in {"import", "analyze", "export"}
    assert state.get_metric("unrelated") == {"value": "keep"}
    assert settings.storage_root == caller_root
    assert contents(Path(first["bundle_dir"])) == before
    assert len(list((state.run_state.work_dir / "brainelecqc" / "runs").iterdir())) == 1


@pytest.mark.parametrize("kwargs", [
    {"qc_config": {"unknown": 1}}, {"qc_config": [1]},
    {"qc_config": {"selection": {"start_seconds": float("nan")}}},
    {"qc_config": {"selection": {"duration_seconds": float("inf")}}},
    {"qc_config": {"selection": {"duration_seconds": 0}}},
    {"qc_config": {"selection": {"start_seconds": True}}},
    {"qc_config": {"selection": {"eeg_indices": [0, 0]}}},
    {"qc_config": {"feature_config": {"unknown": 1}}},
    {"qc_config": {"method_config": {"unknown": 1}}},
    {"qc_config": {"selection": {"eeg_indices": {0, 1}}}},
    {"display_start_seconds": float("nan")}, {"display_start_seconds": -1},
    {"display_duration_seconds": 0}, {"display_duration_seconds": True},
    {"figures": "yes"}, {"title": 3}, {"recording_info": {"modality": "seeg"}},
    {"recording_info": {"metadata": {"bad": object()}}},
])
def test_invalid_configuration_fails_at_construction_without_files(tmp_path, kwargs):
    before = set(tmp_path.iterdir())
    with pytest.raises((TypeError, ValueError)):
        BrainElecQCProcessor(**kwargs)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("git_kind", ["directory", "worktree_file"])
def test_git_work_directory_is_rejected_before_data_copy(tmp_path, git_kind):
    source, _, _ = make_source(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    marker = checkout / ".git"
    marker.mkdir() if git_kind == "directory" else marker.write_text("gitdir: ignored-test-location")
    state = context(source, checkout / "work")
    before = guards(source)
    with pytest.raises(ValueError, match="Git|git|repository"):
        BrainElecQCProcessor(figures=False)(state)
    assert not (state.run_state.work_dir / "brainelecqc").exists()
    assert state.get_metric("brainelecqc") is None
    assert guards(source) == before


def test_input_directory_cannot_contain_work_directory(tmp_path):
    source, _, _ = make_source(tmp_path)
    state = context(source.parent, source.parent / "work")
    with pytest.raises(ValueError, match="input|work|inside|contain"):
        BrainElecQCProcessor(figures=False)(state)
    assert not (state.run_state.work_dir / "brainelecqc").exists()


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_input_is_rejected(tmp_path, directory):
    source, _, _ = make_source(tmp_path)
    link = tmp_path / ("linked-directory" if directory else "linked.edf")
    link.symlink_to(source.parent if directory else source, target_is_directory=directory)
    state = context(link, tmp_path / "link-work")
    before = guards(source)
    with pytest.raises(ValueError, match="[Ss]ymlink|symbolic|link"):
        BrainElecQCProcessor(figures=False)(state)
    assert not (state.run_state.work_dir / "brainelecqc").exists()
    assert guards(source) == before


def test_directory_input_and_figure_paths_belong_to_current_run(tmp_path):
    source, _, _ = make_source(tmp_path, clipped=False)
    state = context(source.parent, tmp_path / "figure-work")
    BrainElecQCProcessor(figures=True, display_start_seconds=0, display_duration_seconds=3)(state)
    summary = state.get_metric("brainelecqc")
    output, manifest = verify_bundle(summary, state.run_state.work_dir)
    assert summary["figures"]
    for name, path in summary["figures"].items():
        png = Path(path)
        assert png.suffix == ".png" and png.parent == output and png.name in manifest["files"]
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not any(p.suffix.lower() in {".svg", ".pdf"} for p in output.iterdir())
    assert "![" in Path(summary["report_path"]).read_text() or ".png)" in Path(summary["report_path"]).read_text()


def test_failed_host_retry_requires_fresh_context_and_resumes_verified_input(tmp_path):
    """A previous failure belongs to its context; a fresh retry clears the host lock."""
    from brainelecqc.recordings import list_recordings

    source, _, _ = make_source(tmp_path)
    before = guards(source)
    work_dir = tmp_path / "retry-work"
    original = context(source, work_dir)
    invalid = PipelineDefinition(name="invalid-range", linxi_plugin=["linxi_brainelecqc"],
        steps=[{"stage": "quality", "processor_name": "BrainElecQCProcessor", "params": {
            "figures": False, "qc_config": {"selection": {"start_seconds": 999}}}}])
    with pytest.raises(ValueError):
        TaskRunner(invalid).run(original, skip_dry_run=False, allow_errors=False)
    lock = work_dir / "LinxiProcessNotCompleted.lock"
    assert original.run_state.failed and original.run_state.error
    assert lock.is_file()
    assert original.get_metric("brainelecqc") is None
    with storage_context(work_dir / "brainelecqc" / "store"):
        previous_record = list_recordings()[0]
        assert verify_recording(previous_record["recording_id"])["valid"]
    valid = PipelineDefinition(name="retry-valid-range", linxi_plugin=["linxi_brainelecqc"],
        steps=[{"stage": "quality", "processor_name": "BrainElecQCProcessor", "params": {"figures": False}}])
    with pytest.raises(ValueError, match=r"(?:fresh|new) LinxiContext"):
        TaskRunner(valid).run(original, skip_dry_run=False, allow_errors=False,
                              incomplete_workdir_strategy="resume")
    runs = work_dir / "brainelecqc" / "runs"
    assert not runs.exists() or not list(runs.iterdir())
    assert original.run_state.failed and lock.is_file()
    retry = context(source, work_dir)
    results = TaskRunner(valid).run(retry, skip_dry_run=False, allow_errors=False,
                                    incomplete_workdir_strategy="resume")
    assert results == [retry] and not retry.run_state.failed and retry.run_state.error is None
    assert not lock.exists()
    summary = retry.get_metric("brainelecqc")
    assert summary["source_reused"] is True
    assert summary["recording_id"] == previous_record["recording_id"]
    assert summary["input_artifact_id"] == previous_record["input_artifact_id"]
    verify_bundle(summary, work_dir)
    assert guards(source) == before
