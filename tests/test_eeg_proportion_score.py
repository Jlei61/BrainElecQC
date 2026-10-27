"""Scoring scope is independent of duplicated windows and auxiliary metadata."""
import pytest

from brainelecqc.eeg.impact import proportion_points
from brainelecqc.eeg.proportion_score import score_line_burden, score_spectrum_burden
from tests.test_eeg_extended_score import _inputs


@pytest.mark.parametrize("fraction,points", [(0, 20), (.013, 19), (.02, 18), (.03125, 18),
    (.05, 16), (.10, 12), (.25, 6), (.50, 0), (1, 0)])
def test_twenty_point_bands(fraction, points):
    assert proportion_points(fraction, 20) == points


def test_line_local_channel_extent_and_harmonics_count_once():
    result, _, _ = _inputs(64)
    f = result["features"]
    hit = {"dimension_id": "line", "channel_id": "eeg:0",
           "start_sample": f["start_sample"], "stop_sample": f["start_sample"]+1000}
    outcome = score_line_burden(result, {"evidence": [hit, hit]}, {"score": 0, "reason": "", "metrics": {}})
    assert outcome["score"] == 19
    assert outcome["metrics"]["channel_time_burden"]["fraction"] == .1/64
    assert outcome["metrics"]["time_burden"]["fraction"] == .1


def test_missing_header_is_unscored_and_one_local_spectral_issue_is_small():
    from brainelecqc.eeg.header_qc import score_header
    assert score_header(None)["score"] is None
    outcome = {"score": 5, "metrics": {"channels": [
        {"band_fraction": .5 if i == 0 else 1, "valid_psd": True} for i in range(64)]}}
    assert score_spectrum_burden({}, outcome)["score"] == 4
