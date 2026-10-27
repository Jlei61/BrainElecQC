"""Paper-method contracts and independently implemented numerical checks."""

import hashlib
import json
import warnings

import numpy as np
import pytest
from scipy.signal import detrend, welch
from scipy.stats import f as f_distribution

from brainelecqc.eeg.spectral import (
    SpectralQCConfig, _fit_one, _fooof_class, _line_tapers, _thomson_spectrum,
    compare_spectral_reference, compute_spectral_qc,
)


def _noise(seconds=30, sfreq=256, seed=26):
    return np.random.default_rng(seed).normal(0, 8e-6, round(seconds * sfreq))


def _sinusoid(frequency, seconds=30, sfreq=256, amplitude=15e-6):
    return amplitude * np.cos(2 * np.pi * frequency * np.arange(round(seconds * sfreq)) / sfreq + .3)


def test_thomson_statistic_and_amplitude_agree_with_mne_reference():
    """MNE supplies a separate implementation, fed the identical explicit DPSS."""
    from mne.filter import _mt_spectrum_remove

    sample = detrend(_noise(4) + _sinusoid(50, 4) + .5 * _sinusoid(100, 4), type="linear")
    tapers, _ = _line_tapers(len(sample), 256, 2)
    frequencies, statistics, amplitudes = _thomson_spectrum(sample[None], 256, tapers)
    threshold = f_distribution.ppf(1 - .01 / len(sample), 2, 12)
    reference_cleaned, reference_frequencies = _mt_spectrum_remove(
        sample, 256, None, None, tapers, threshold, lambda n: (tapers, threshold))
    selected = np.flatnonzero(statistics[0] > threshold)
    np.testing.assert_array_equal(frequencies[selected], reference_frequencies)
    reconstructed = sum(2 * abs(amplitudes[0, index]) * np.cos(
        2 * np.pi * frequencies[index] * np.arange(len(sample)) / 256 + np.angle(amplitudes[0, index]))
        for index in selected)
    np.testing.assert_allclose(sample - reference_cleaned, reconstructed, rtol=1e-12, atol=2e-17)
    # Locate the actual F value by the reference implementation's decision
    # boundary, rather than repeating our formula in the test.
    target = int(np.flatnonzero(frequencies == 50)[0])
    for multiplier, should_find in ((1 - 1e-8, True), (1 + 1e-8, False)):
        boundary = float(statistics[0, target] * multiplier)
        _, found = _mt_spectrum_remove(sample, 256, None, None, tapers, boundary,
                                       lambda n: (tapers, boundary))
        assert (50 in found) is should_find


def test_line_detection_locations_harmonics_and_multiplicity():
    data = np.stack([_noise(12) + _sinusoid(50.25, 12) + _sinusoid(100, 12), _noise(12, seed=8)])
    saved = data.copy()
    result = compute_spectral_qc(data, 256, ["a", "b"], ["EEG", "EEG"], start_sample=4096, start_seconds=200)
    line = result["line_noise"]
    assert line["settings"]["n_tapers"] == 7
    assert line["settings"]["time_half_bandwidth_product"] == 4
    assert line["multiplicity"]["hypothesis_count"] == 2 * 9 * 4 * 17
    assert line["windows"][0]["start_sample"] == 4096
    assert line["windows"][-1]["stop_seconds"] == 212
    target = next(x for x in line["observations"] if x["channel_id"] == "a" and x["nominal_hz"] == 50)
    assert target["peak_frequency_hz"] == 50.25
    assert target["flagged"]
    assert target["amplitude_uv"] == pytest.approx(15, rel=.12)
    assert target["p_corrected"] == target["p_value"] * line["multiplicity"]["hypothesis_count"]
    harmonics = [x for x in line["observations"] if x["channel_id"] == "a" and x["nominal_hz"] == 100]
    assert any(x["flagged"] for x in harmonics)
    assert harmonics[0]["families"] == [{"fundamental_hz": 50.0, "harmonic": 2}]
    assert all(not x["flagged"] for x in line["observations"] if x["channel_id"] == "b")
    np.testing.assert_array_equal(data, saved)
    assert result["input"]["data_sha256"] == hashlib.sha256(np.asarray(data, dtype="<f8").tobytes()).hexdigest()
    json.dumps(result, allow_nan=False)


def test_shared_harmonic_counts_scanned_bins_once():
    # At 300 Hz both fundamental families refer to one tested frequency band.
    data = _noise(4, sfreq=1024)[None]
    result = compute_spectral_qc(data, 1024, ["a"])
    line = result["line_noise"]
    shared = next(x for x in line["targets"] if x["nominal_hz"] == 300)
    assert shared["families"] == [{"fundamental_hz": 50.0, "harmonic": 6},
                                  {"fundamental_hz": 60.0, "harmonic": 5}]
    assert line["multiplicity"]["hypothesis_count"] == len(line["targets"]) * 17


def test_nonstationary_transient_is_not_a_confirmed_line_by_amplitude():
    data = _noise(12)
    data[1300:1350] += 700e-6
    result = compute_spectral_qc(data[None], 256, ["x"])
    assert not any(x["flagged"] for x in result["line_noise"]["observations"])


def test_linear_drift_roundoff_is_not_a_line():
    data = np.linspace(-1e-3, 2e-3, 12 * 256)[None]
    result = compute_spectral_qc(data, 256, ["ramp"])
    assert all(row["status"] == "no_residual_power" for row in result["line_noise"]["observations"])
    assert not any(row["flagged"] for row in result["line_noise"]["observations"])
    reference = compare_spectral_reference(data, result)
    assert reference["passed"]
    assert reference["line"]["channel_windows_compared"] == 0
    assert reference["line"]["unassessed_channel_windows"] == 9


def test_reference_helper_checks_positive_and_negative_realizations():
    data = np.stack([_noise(12) + _sinusoid(50, 12), _noise(12, seed=12)])
    result = compute_spectral_qc(data, 256, ["a", "b"])
    reference = compare_spectral_reference(data, result)
    assert reference["passed"]
    assert reference["line"]["channel_windows_compared"] == 18
    assert reference["line"]["observed_peak_amplitude_checks"] == 72
    assert reference["line"]["f_boundary_checks"] == 48
    assert reference["line"]["f_boundary_failures"] == 0
    assert reference["parameterization"]["compared_full_channel_fits"] == 2
    with pytest.raises(ValueError, match="exact spectral-analysis input"):
        compare_spectral_reference(data * 2, result)
    json.dumps(reference, allow_nan=False)


def test_short_constant_and_low_sampling_rate_are_explicit():
    constant = compute_spectral_qc(np.full((1, 30 * 256), 2e-6), 256, ["x"])
    assert all(x["status"] == "no_ac_power" and not x["flagged"] for x in constant["line_noise"]["observations"])
    assert constant["parameterization"]["full"][0]["status"] == "no_power"
    short = compute_spectral_qc(_noise(3)[None], 256, ["x"], start_sample=100)
    assert short["line_noise"]["status"] == "insufficient_duration"
    assert short["line_noise"]["tail_samples"] == 768
    assert short["parameterization"]["full"][0]["status"] == "insufficient_duration"
    low = compute_spectral_qc(_noise(30, sfreq=40)[None], 40, ["x"])
    assert low["line_noise"]["status"] == "insufficient_bandwidth"
    assert low["parameterization"]["full"][0]["status"] == "insufficient_bandwidth"
    for value in (constant, short, low):
        json.dumps(value, allow_nan=False)


def test_fooof_matches_author_model_and_welch_units():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fooof import FOOOF
    data = (_noise(65) + _sinusoid(10.5, 65, amplitude=10e-6))[None]
    data.setflags(write=False)
    result = compute_spectral_qc(data, 256, ["o"], ["Oz"], start_sample=300, start_seconds=14.5)
    f, p = welch(data, 256, window="hann", nperseg=1024, noverlap=512,
                 nfft=1024, detrend="constant", scaling="density", average="mean")
    np.testing.assert_array_equal(result["psd"]["frequencies_hz"], f)
    np.testing.assert_array_equal(result["psd"]["power_v2_hz"], p)
    direct = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0,
                   peak_threshold=2, aperiodic_mode="fixed", verbose=False)
    direct.fit(f, p[0], [2, 40])
    actual = result["parameterization"]["full"][0]
    np.testing.assert_array_equal(actual["model_log10_v2_hz"], direct.fooofed_spectrum_)
    np.testing.assert_array_equal(actual["aperiodic_log10_v2_hz"], direct.get_model("aperiodic", "log"))
    assert actual["r_squared"] == direct.r_squared_
    assert actual["mae_log10_power"] == direct.error_
    assert actual["alpha_status"] == "peak_present"
    assert actual["alpha_peaks"][0]["center_frequency_hz"] == pytest.approx(10.5, abs=.15)
    assert actual["alpha_peaks"][0]["height_db"] == 10 * actual["alpha_peaks"][0]["height_log10_power"]
    windows = result["parameterization"]["windows"]
    assert [(x["start_sample"], x["stop_sample"]) for x in windows] == [(300, 7980), (7980, 15660), (15660, 16940)]
    assert windows[-1]["status"] == "insufficient_duration"
    assert result["parameterization"]["curve_unit"] == "log10(V^2/Hz)"
    json.dumps(result, allow_nan=False)


def test_smooth_aperiodic_spectrum_has_no_alpha_peak():
    frequencies = np.arange(0, 128.25, .25)
    power = 1e-10 / np.maximum(frequencies, .25) ** 1.6
    result = _fit_one(frequencies, power, SpectralQCConfig(), 60,
                      {"periodogram_count": 29, "frequency_resolution_hz": .25}, _fooof_class())
    assert result["alpha_status"] == "no_fitted_peak"
    assert result["aperiodic_parameters"]["exponent"] == pytest.approx(1.6)
    assert result["rmse_log10_power"] < 1e-10


@pytest.mark.parametrize("kwargs", [
    {"line_window_seconds": 0}, {"line_step_seconds": 5}, {"line_family_alpha": 1},
    {"welch_overlap_fraction": 1}, {"parameter_min_seconds": 31},
    {"fit_range_hz": (8, 13)}, {"peak_width_limits_hz": (8, 1)},
    {"max_n_peaks": 2.5}, {"line_fundamentals_hz": (50, 50)}, {"peak_threshold_sd": True},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SpectralQCConfig(**kwargs)


@pytest.mark.parametrize("data,rate,ids,kwargs", [
    (np.zeros((2, 20)), 256, ["same", "same"], {}),
    (np.zeros((1, 20)), 0, ["x"], {}),
    (np.zeros((1, 20)), 256, ["x"], {"start_sample": 1.5}),
    (np.full((1, 20), np.nan), 256, ["x"], {}),
    (np.zeros((1, 20), dtype=complex), 256, ["x"], {}),
])
def test_invalid_data_is_rejected(data, rate, ids, kwargs):
    with pytest.raises(ValueError):
        compute_spectral_qc(data, rate, ids, **kwargs)
