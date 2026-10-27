"""Evidence remains bound to source channels, discontinuous clocks and arrays."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from brainelecqc.eeg.binding import validate_result_identity
from brainelecqc.eeg.export import export_eeg_qc
from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.eeg.features import compute_eeg_features
from brainelecqc.eeg.policy import propose_eeg_decisions
from brainelecqc.eeg.paper import compute_paper_methods, validate_method_config, attach_paper_candidates
from brainelecqc.signals import describe_artifact
from tests.test_eeg_pipeline import _import, _source
from tests.test_eeg_events import signals


@pytest.fixture
def paper_result(tmp_path):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    return run_eeg_qc(artifact, eeg_indices=[2, 0], method_config={"comparison_groups": ["M2", "M1"]})


def test_continuous_subselection_preserves_channel_order_and_reference_binding(paper_result):
    result = paper_result
    assert validate_result_identity(result)
    assert result["features"]["channel_names"] == ["EEG Cz", "EEG Oz"]
    assert result["method_results"]["bad_channels"]["channel_ids"] == result["features"]["channel_ids"]
    assert [row["group_id"] for row in result["method_results"]["bad_channels"]["channels"]] == ["M2", "M1"]
    assert result["keep_mask"].all()
    assert result["decisions"]["automatic_exclusions"] == []


def test_discontinuous_segment_clock_is_shared_by_all_paper_methods(tmp_path):
    source, _, fs = _source(tmp_path, discontinuous=True, clipped=False)
    _, artifact = _import(source)
    segment = describe_artifact(artifact)["streams"][0]["segments"][1]
    result = run_eeg_qc(artifact, segment_id=segment["segment_id"], start_seconds=10.5,
                        duration_seconds=2, eeg_indices=[2, 0])
    assert validate_result_identity(result)
    origin, stop = int(4.5 * fs), int(6.5 * fs)
    methods = result["method_results"]
    for row in (methods["bad_channels"]["input_range"], methods["spectral"]["input"], methods["events"]):
        assert (row["start_sample"], row["stop_sample"]) == (origin, stop)
        assert (row["start_seconds"], row["stop_seconds"]) == (10.5, 12.5)
    windows = methods["bad_channels"]["windows"]
    assert windows[0]["start_sample"] == origin
    assert windows[0]["start_seconds"] == 10.5
    assert windows[-1]["stop_sample"] == stop


def test_numeric_export_hashes_all_method_evidence(paper_result, tmp_path):
    directory = tmp_path / "report"
    export_eeg_qc(paper_result, directory, figures=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {"paper_methods.json", "bad_channels.tsv", "line_windows.tsv", "spectral_fits.tsv", "events_ecg.tsv", "events_eog.tsv"} <= set(manifest["files"])
    for name, row in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == row["sha256"]
    methods = json.loads((directory / "paper_methods.json").read_text())
    assert methods["bad_channels"]["data_sha256"] == manifest["data_sha256"]
    assert methods["spectral"]["input"]["data_sha256"] == manifest["data_sha256"]
    assert methods["events"]["data_sha256"] == manifest["data_sha256"]


@pytest.mark.parametrize("block", ["bad_channels", "spectral", "events"])
def test_stale_method_hash_is_rejected_before_export(paper_result, tmp_path, block):
    result = deepcopy(paper_result)
    target = result["method_results"][block]
    if block == "spectral":
        target = target["input"]
    target["data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint|identity"):
        validate_result_identity(result)
    with pytest.raises(ValueError, match="fingerprint|identity"):
        export_eeg_qc(result, tmp_path / "stale", figures=False)
    assert not (tmp_path / "stale").exists()


@pytest.mark.parametrize("block", ["bad_channels", "spectral", "events"])
def test_same_shape_reordered_method_channels_are_rejected(paper_result, block):
    result = deepcopy(paper_result)
    target = result["method_results"][block]
    if block == "spectral":
        target = target["input"]
    target["channel_ids"] = target["channel_ids"][::-1]
    with pytest.raises(ValueError, match="channel_ids"):
        validate_result_identity(result)


def test_same_samples_different_time_origin_is_rejected(paper_result):
    result = deepcopy(paper_result)
    result["method_results"]["bad_channels"]["input_range"]["start_seconds"] += 10
    with pytest.raises(ValueError, match="clock"):
        validate_result_identity(result)


def test_reference_swap_is_rejected_even_when_eeg_hash_matches(paper_result):
    result = deepcopy(paper_result)
    result["references"]["eog"] = result["references"]["ecg"].copy()
    with pytest.raises(ValueError, match="reference_fingerprint"):
        validate_result_identity(result)


def test_keep_mask_cannot_disagree_with_review_only_decisions(paper_result):
    result = deepcopy(paper_result)
    result["keep_mask"][0, 50] = False
    with pytest.raises(ValueError, match="Keep mask"):
        validate_result_identity(result)


def test_preserved_values_cannot_silently_change_after_measurement(paper_result):
    result = deepcopy(paper_result)
    result["masked_data_volts"][0, 50] += 1e-6
    with pytest.raises(ValueError, match="Retained sample"):
        validate_result_identity(result)


def test_regenerated_bundle_can_validate_without_duplicate_masked_data(paper_result):
    result = deepcopy(paper_result)
    del result["masked_data_volts"]
    del result["channels"]
    assert validate_result_identity(result)


def _event_result():
    data, refs, fs = signals()
    ids, names = ["first", "second"], ["same", "same"]
    channels = [{"channel_id": cid, "raw_name": name, "unit": "V"} for cid, name in zip(ids, names)]
    ref_meta = {kind: {"channel_id": f"aux:{kind}", "raw_name": kind} for kind in refs}
    source = {"data_volts": data, "channels": channels, "references": refs, "reference_channels": ref_meta,
              "metadata": {"sampling_frequency_hz": fs, "start_sample": 4096, "recording_start_seconds": 80}}
    features = compute_eeg_features(data, fs, ids, names, references=refs, start_sample=4096, start_seconds=80)
    features["reference_channels"] = ref_meta
    methods = compute_paper_methods(source, validate_method_config())
    decisions = attach_paper_candidates(propose_eeg_decisions(data, features), methods)
    manifest = {key: features[key] for key in ("channel_ids", "data_sha256", "sampling_rate_hz", "start_sample", "stop_sample", "start_seconds")}
    manifest["unit"] = "V"
    return {"features": features, "manifest": manifest, "decisions": decisions, "method_results": methods,
            "raw_data_volts": data, "keep_mask": np.ones_like(data, dtype=bool), "references": refs, "channels": channels}


def test_boundary_events_keep_clipped_context_without_entering_the_mean():
    result = _event_result()
    block = result["method_results"]["events"]["by_kind"]["eog"]
    boundary = [row for row in block["event_table"] if not row["complete_epoch"]]
    assert boundary
    assert all(not row["included_in_average"] for row in boundary)
    assert validate_result_identity(result)


def test_boundary_event_cannot_be_marked_as_a_complete_average_member():
    result = _event_result()
    block = result["method_results"]["events"]["by_kind"]["eog"]
    boundary = next(row for row in block["event_table"] if not row["complete_epoch"])
    boundary["included_in_average"] = True
    with pytest.raises(ValueError, match="boundary epoch"):
        validate_result_identity(result)
