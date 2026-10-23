import numpy as np
import pytest
from scripts.figure_acceptance import candidate_metrics, synthetic_case, require_external


@pytest.mark.parametrize("modality", ["EEG", "SEEG", "ECoG"])
def test_known_defects_and_clean_counterexamples(modality):
    data, fs, _, truth = synthetic_case(modality)
    metrics = candidate_metrics(data, fs, list(range(8)), 8)
    assert "flat" in metrics[truth["flat"]]["flags"]
    assert "line_noise" in metrics[truth["line"]]["flags"]
    assert "high_frequency_candidate" in metrics[truth["noisy"]]["flags"]
    assert "ECG_coupling_candidate" in metrics[truth["ecg"]]["flags"]
    assert "high_frequency_candidate" in metrics[truth["emg_like"]]["flags"]
    assert all(not metrics[i]["flags"] for i in truth["clean_controls"])
    assert all(m["status"] == "pending" for m in metrics)


def test_no_reference_is_not_a_negative_ecg_finding():
    data, fs, _, _ = synthetic_case("EEG")
    metrics = candidate_metrics(data, fs, [3])
    assert metrics[0]["ecg_correlation"] is None
    assert "ECG_coupling_candidate" not in metrics[0]["flags"]
    with pytest.raises(ValueError):
        require_external("reports")


def test_high_frequency_neural_activity_remains_pending_not_deleted():
    t = np.arange(1000)/500
    data = (15e-6*np.sin(2*np.pi*90*t))[None, :]
    original = data.copy()
    result = candidate_metrics(data, 500, [0])
    assert result[0]["flags"] == ["high_frequency_candidate"]
    assert result[0]["status"] == "pending"
    np.testing.assert_array_equal(data, original)
