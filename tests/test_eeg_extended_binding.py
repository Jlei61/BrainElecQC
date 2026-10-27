"""Continuity and muscle evidence retain source channels, clocks and auxiliaries."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from brainelecqc.eeg.binding import validate_result_identity
from brainelecqc.eeg.continuity import compute_continuity_qc
from brainelecqc.eeg.features import compute_eeg_features
from brainelecqc.eeg.muscle import compute_muscle_qc
from brainelecqc.eeg.paper import attach_paper_candidates, compute_paper_methods, validate_method_config
from brainelecqc.eeg.policy import propose_eeg_decisions
from tests.test_eeg_muscle import signals


@pytest.fixture(scope="module")
def extended_result():
    data, emg, fs = signals()
    data[1, 5 * fs:5 * fs + 60] = 0
    ids, names = ["first", "second"], ["EEG A", "EEG B"]
    channels = [{"channel_id": cid, "raw_name": name, "unit": "V"} for cid, name in zip(ids, names)]
    refs = {"emg": emg}
    ref_meta = {"emg": {"channel_id": "aux-emg", "raw_name": "EMG chin", "source_index": 2}}
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


def _muscle(result, **overrides):
    f = result["features"]
    args = dict(references=result["references"], reference_channels=f["reference_channels"],
                start_sample=f["start_sample"], start_seconds=f["start_seconds"])
    args.update(overrides)
    return compute_muscle_qc(result["raw_data_volts"], f["sampling_rate_hz"], f["channel_ids"], f["channel_names"], **args)


def test_new_methods_have_complete_binding_and_legacy_blocks_remain_valid(extended_result):
    assert extended_result["method_results"]["continuity"]["candidates"]
    assert extended_result["method_results"]["muscle"]["candidates"]
    assert validate_result_identity(extended_result)
    legacy = deepcopy(extended_result)
    del legacy["method_results"]["continuity"]
    del legacy["method_results"]["muscle"]
    assert validate_result_identity(legacy)


@pytest.mark.parametrize("method", ["continuity", "muscle"])
@pytest.mark.parametrize("field", ["data_sha256", "channel_ids", "channel_names", "rate", "clock", "measurements"])
def test_new_methods_reject_same_shape_wrong_identity(extended_result, method, field):
    result = deepcopy(extended_result)
    block = result["method_results"][method]
    if field == "data_sha256":
        block[field] = "0" * 64
    elif field in {"channel_ids", "channel_names"}:
        block[field] = block[field][::-1]
    elif field == "rate":
        block["sampling_rate_hz" if method == "continuity" else "sample_rate_hz"] += 1
    elif field == "clock":
        block["input_range"]["start_seconds"] += 1
    else:
        key = "measurements" if method == "continuity" else "channels"
        block[key] = block[key][::-1]
    with pytest.raises(ValueError, match="identity|clock"):
        validate_result_identity(result)


@pytest.mark.parametrize("method", ["continuity", "muscle"])
@pytest.mark.parametrize("field", ["channel_id", "channel_name", "channel_index", "start_sample", "stop_seconds"])
def test_candidate_points_to_its_own_channel_and_original_clock(extended_result, method, field):
    result = deepcopy(extended_result)
    row = result["method_results"][method]["candidates"][0]
    row[field] = "wrong" if field in {"channel_id", "channel_name"} else -1
    with pytest.raises(ValueError, match="identity|clock"):
        validate_result_identity(result)


@pytest.mark.parametrize("key", ["envelope_uv", "z_score", "excluded"])
def test_muscle_sample_axis_rejects_truncated_channel_array(extended_result, key):
    result = deepcopy(extended_result)
    result["method_results"]["muscle"]["sample_metrics"][key][0] = result["method_results"]["muscle"]["sample_metrics"][key][0][1:]
    with pytest.raises(ValueError, match="sample axis"):
        validate_result_identity(result)


def test_exclusion_mask_requires_boolean_samples(extended_result):
    result = deepcopy(extended_result)
    block = result["method_results"]["muscle"]["sample_metrics"]
    block["excluded"][0] = np.asarray(block["excluded"][0], dtype=float).tolist()
    with pytest.raises(ValueError, match="booleans"):
        validate_result_identity(result)


@pytest.mark.parametrize("method", ["continuity", "muscle"])
def test_present_method_block_requires_complete_evidence(extended_result, method):
    result = deepcopy(extended_result)
    result["method_results"][method] = None
    with pytest.raises(ValueError, match="identity"):
        validate_result_identity(result)


def test_boolean_channel_index_cannot_alias_an_integer(extended_result):
    result = deepcopy(extended_result)
    result["method_results"]["continuity"]["candidates"][0]["channel_index"] = True
    with pytest.raises(ValueError, match="integer"):
        validate_result_identity(result)


@pytest.mark.parametrize("change", ["signal", "clock", "source", "reference_array", "missing_reference"])
def test_emg_reference_swap_and_sample_shape_are_rejected(extended_result, change):
    result = deepcopy(extended_result)
    if change == "signal":
        result["references"]["emg"] = np.roll(result["references"]["emg"], 40)
    elif change == "clock":
        result["references"]["emg"] = result["references"]["emg"][1:]
    elif change == "source":
        result["method_results"]["muscle"]["reference"]["source_reference"]["channel_id"] = "another-emg"
    elif change == "missing_reference":
        del result["references"]["emg"]
    else:
        values = result["method_results"]["muscle"]["sample_metrics"]["reference_envelope_uv"]
        values.pop()
    with pytest.raises(ValueError, match="identity|clock|fingerprint|availability|sample axis"):
        validate_result_identity(result)


@pytest.mark.parametrize("reference_kind", ["constant", "nonfinite", "missing"])
def test_unavailable_emg_preserves_explicit_status_and_raw_binding(extended_result, reference_kind):
    result = deepcopy(extended_result)
    if reference_kind == "missing":
        del result["references"]["emg"]
        del result["features"]["reference_channels"]["emg"]
    else:
        result["references"]["emg"] = np.zeros_like(result["references"]["emg"])
        if reference_kind == "nonfinite":
            result["references"]["emg"][100] = np.nan
    result["method_results"]["muscle"] = _muscle(result)
    assert validate_result_identity(result)


def test_early_unavailable_muscle_output_still_binds_input_and_emg(extended_result):
    result = deepcopy(extended_result)
    result["method_results"]["muscle"] = _muscle(result, config={"min_duration_seconds": 60})
    assert result["method_results"]["muscle"]["status"] == "insufficient_duration"
    assert validate_result_identity(result)
    result["references"]["emg"] = np.roll(result["references"]["emg"], 1)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_result_identity(result)


def test_real_bundle_extended_methods_bind_and_tampering_is_rejected():
    location = os.environ.get("BRAINELECQC_REAL_EEG_BUNDLE")
    if not location:
        pytest.skip("Set BRAINELECQC_REAL_EEG_BUNDLE for external real EEG validation")
    source = Path(location)
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("channels.json", "features.json", "paper_methods.json", "decisions.json", "review_signals.npz", "keep_mask.npy"):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest["files"][name]["sha256"]
    f = json.loads((source / "features.json").read_text())
    channels = json.loads((source / "channels.json").read_text())
    with np.load(source / "review_signals.npz", allow_pickle=False) as values:
        data = values["raw_data_volts"]
        refs = {key: values[key] for key in values.files if key != "raw_data_volts"}
    result = {"features": f, "manifest": manifest, "channels": channels,
              "raw_data_volts": data, "references": refs, "keep_mask": np.load(source / "keep_mask.npy"),
              "decisions": json.loads((source / "decisions.json").read_text()),
              "method_results": json.loads((source / "paper_methods.json").read_text())}
    assert validate_result_identity(result)
    result["method_results"]["continuity"] = compute_continuity_qc(data, f["sampling_rate_hz"],
        f["channel_ids"], f["channel_names"], channels=channels, start_sample=f["start_sample"], start_seconds=f["start_seconds"])
    result["method_results"]["muscle"] = _muscle(result)
    assert validate_result_identity(result)
    original = refs["emg"].copy()
    refs["emg"][10] += 1e-6
    with pytest.raises(ValueError, match="fingerprint"):
        validate_result_identity(result)
    refs["emg"] = original
    result["method_results"]["continuity"]["candidates"][0]["start_seconds"] += .01
    with pytest.raises(ValueError, match="clock"):
        validate_result_identity(result)
