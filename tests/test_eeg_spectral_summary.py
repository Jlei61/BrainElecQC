"""Spectral plots choose channels reproducibly and retain source identity."""
import copy

import pytest

from brainelecqc.eeg.binding import validate_result_identity
from brainelecqc.eeg.pipeline import run_eeg_qc
from brainelecqc.eeg.spectral_summary import occipital_name, select_spectral_channels
from tests.test_eeg_pipeline import _source, _import


@pytest.mark.parametrize("name, expected", [("O1..", "o1"), ("EEG Oz", "oz"), ("EEG O2-M1", "o2"),
                                           ("O10", None), ("FO1", None), ("ECG", None)])
def test_occipital_labels_do_not_guess_unrelated_channels(name, expected):
    assert occipital_name(name) == expected


def test_display_selection_uses_named_leads_and_flagged_candidates_not_alpha_height():
    names = ["Fp1", "Oz..", "O1..", "O2..", "T7", "T8", "Cz"]
    result = {"features": {"channel_names": names, "channel_ids": [str(i) for i in range(7)]},
              "method_results": {"bad_channels": {"channels": [
                  {"channel_index": 4, "flags": ["high_frequency_noise"], "hf_z": 9},
                  {"channel_index": 5, "flags": [], "hf_z": 1000},
                  {"channel_index": 6, "flags": ["amplitude_deviation"], "amplitude_z": -8}]},
                  "spectral": {"line_noise": {"observations": []}}}}
    assert [r["channel_index"] for r in select_spectral_channels(result)] == [2, 1, 3, 4, 6, 0]


@pytest.mark.parametrize("tamper", ["name", "window_time", "window_index", "full_coverage", "band", "missing_window"])
def test_alpha_results_reject_misaligned_evidence(tmp_path, tamper):
    source, _, _ = _source(tmp_path, clipped=False)
    _, artifact = _import(source)
    result = run_eeg_qc(artifact)
    assert validate_result_identity(result)
    result = copy.deepcopy(result)
    alpha = result["method_results"]["spectral"]["alpha_power"]
    if tamper == "name":
        alpha["full"][0]["channel_name"] = "wrong channel"
    elif tamper == "window_time":
        alpha["windows"][0]["start_seconds"] += 1
    elif tamper == "window_index":
        alpha["windows"][0]["window_index"] += 1
    elif tamper == "full_coverage":
        alpha["full"][0]["stop_sample"] -= 1
        alpha["full"][0]["stop_seconds"] -= 1/alpha["sfreq"]
    elif tamper == "band":
        alpha["alpha_band_hz"][0] += 1
    else:
        alpha["windows"].pop()
    with pytest.raises(ValueError):
        validate_result_identity(result)
