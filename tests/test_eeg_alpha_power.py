"""Known-power signals, independent PSD calls, and alpha clock contracts."""

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.signal import welch

from brainelecqc.eeg.alpha_power import _integrate_band, compute_alpha_power
from brainelecqc.eeg.spectral import compute_spectral_qc


def _sine(amplitude_uv=20, frequency=10, sfreq=256, seconds=12):
    return amplitude_uv * 1e-6 * np.sin(2 * np.pi * frequency * np.arange(round(sfreq * seconds)) / sfreq)


def _compute(data, sfreq=256, ids=None, names=None, **kwargs):
    ids = ids or [f"channel-{i}" for i in range(len(data))]
    return compute_spectral_qc(data, sfreq, ids, names, **kwargs)


def test_sinusoid_absolute_power_and_amplitude_scaling():
    x = _sine()[None]
    base = _compute(x)["alpha_power"]
    doubled = _compute(2 * x)["alpha_power"]
    assert base["full"][0]["alpha_power_uv2"] == pytest.approx(20**2 / 2, rel=1e-12)
    assert base["full"][0]["reference_power_uv2"] == pytest.approx(200, rel=1e-12)
    assert base["full"][0]["relative_alpha_percent"] == pytest.approx(100)
    assert doubled["full"][0]["alpha_power_uv2"] == pytest.approx(4 * base["full"][0]["alpha_power_uv2"])
    assert len(base["windows"]) == 5
    assert all(row["alpha_power_uv2"] == pytest.approx(200, rel=1e-12) for row in base["windows"])
    assert base["config"]["integration"].startswith("linear interpolation")
    json.dumps(base, allow_nan=False)


def test_extra_beta_changes_relative_but_preserves_absolute_alpha():
    alpha = _sine()[None]
    beta = _sine(frequency=20)[None]
    first = _compute(alpha)["alpha_power"]["full"][0]
    second = _compute(alpha + beta)["alpha_power"]["full"][0]
    assert second["alpha_power_uv2"] == pytest.approx(first["alpha_power_uv2"], rel=1e-12)
    assert second["reference_power_uv2"] == pytest.approx(400, rel=1e-12)
    assert second["relative_alpha_percent"] == pytest.approx(50, rel=1e-12)


def test_off_grid_edges_integrate_a_linear_density_exactly():
    frequencies = np.array([0, 4, 8, 12, 16, 20, 40], dtype=float)
    power = frequencies * 1e-12
    value, status = _integrate_band(frequencies, power, (8.3, 12.7))
    assert status == "computed"
    assert value == pytest.approx((12.7**2 - 8.3**2) / 2, abs=1e-12)
    assert _integrate_band(frequencies, power, (8, 41)) == (None, "insufficient_bandwidth")
    assert _integrate_band(frequencies, power, (8.3, 9.7)) == (None, "insufficient_frequency_resolution")


def _reference_area(frequencies, power, limits):
    """Independent quadrature over the piecewise-linear density in µV²/Hz."""
    lower, upper = limits
    knots = frequencies[(frequencies > lower) & (frequencies < upper)]
    value, _ = quad(lambda hz: np.interp(hz, frequencies, power * 1e12), lower, upper,
                    points=knots, epsabs=1e-10, epsrel=1e-12, limit=1000)
    return value


def test_full_and_each_window_agree_with_scipy_mne_and_independent_quadrature():
    from mne.time_frequency import psd_array_welch

    data = np.random.default_rng(11).normal(0, 8e-6, (2, 256 * 12))
    data[0] += _sine(frequency=10.4)
    alpha, reference = (8.3, 12.7), (2.1, 39.3)
    spectral = _compute(data, config={"alpha_band_hz": alpha, "fit_range_hz": reference})
    result = spectral["alpha_power"]
    f, scipy_power = welch(data, fs=256, window="hann", nperseg=1024, noverlap=512,
                           detrend="constant", scaling="density", average="mean")
    mne_power, mne_f = psd_array_welch(data, 256, n_fft=1024, n_per_seg=1024,
        n_overlap=512, window="hann", average="mean", remove_dc=True, verbose=False)
    np.testing.assert_array_equal(f, mne_f)
    np.testing.assert_allclose(scipy_power, mne_power, rtol=1e-12, atol=1e-30)
    for row in result["full"]:
        index = row["channel_index"]
        assert row["alpha_power_uv2"] == pytest.approx(_reference_area(mne_f, mne_power[index], alpha), rel=1e-12)
        assert row["reference_power_uv2"] == pytest.approx(_reference_area(f, scipy_power[index], reference), rel=1e-12)
    for row in result["windows"]:
        window = data[row["channel_index"], row["start_offset"]:row["stop_offset"]]
        p, f = psd_array_welch(window, 256, n_fft=1024, n_per_seg=1024,
            n_overlap=0, window="hann", average="mean", remove_dc=True, verbose=False)
        assert row["alpha_power_uv2"] == pytest.approx(_reference_area(f, p, alpha), rel=1e-12)
        assert row["reference_power_uv2"] == pytest.approx(_reference_area(f, p, reference), rel=1e-12)


def test_low_sampling_rate_retains_available_absolute_power():
    alpha_only = _compute(_sine(sfreq=40)[None], sfreq=40)["alpha_power"]
    row = alpha_only["full"][0]
    assert row["alpha_power_uv2"] == pytest.approx(200)
    assert row["reference_power_uv2"] is None and row["relative_alpha_percent"] is None
    assert row["status"] == "alpha_only"
    assert row["reference_power_status"] == "insufficient_bandwidth"
    unavailable = _compute(_sine(frequency=5, sfreq=20)[None], sfreq=20)["alpha_power"]
    assert unavailable["full"][0]["alpha_power_status"] == "insufficient_bandwidth"
    assert unavailable["full"][0]["alpha_power_uv2"] is None
    assert unavailable["full"][0]["relative_alpha_percent"] is None
    json.dumps([alpha_only, unavailable], allow_nan=False)


@pytest.mark.parametrize("value", [0.0, 2e-6])
def test_zero_ac_power_has_explicit_status_and_null_values(value):
    result = _compute(np.full((1, 3072), value))["alpha_power"]
    for row in result["full"] + result["windows"]:
        assert row["status"] == "no_power"
        assert row["alpha_power_uv2"] is None
        assert row["reference_power_uv2"] is None
        assert row["relative_alpha_percent"] is None


@pytest.mark.parametrize("samples,status", [(1, "insufficient_frequency_grid"),
                                            (2, "insufficient_frequency_resolution"),
                                            (8, "insufficient_frequency_resolution")])
def test_extremely_short_psd_has_explicit_frequency_support(samples, status):
    data = np.arange(samples, dtype=float)[None] * 1e-6
    result = _compute(data)["alpha_power"]
    assert result["full"][0]["alpha_power_status"] == status
    assert result["full"][0]["alpha_power_uv2"] is None
    json.dumps(result, allow_nan=False)


def test_original_clock_actual_full_coverage_and_tail_are_preserved():
    data = np.random.default_rng(13).normal(0, 1e-6, (1, 13 * 256 + 1))
    spectral = _compute(data, start_sample=6400, start_seconds=50.25)
    result = spectral["alpha_power"]
    full = result["full"][0]
    assert (full["start_sample"], full["stop_sample"]) == (6400, 9472)
    assert (full["start_seconds"], full["stop_seconds"]) == (50.25, 62.25)
    assert result["tail_samples"] == 257
    assert (result["tail"]["start_sample"], result["tail"]["stop_sample"]) == (9472, 9729)
    assert result["tail"]["stop_seconds"] == 50.25 + data.shape[1] / 256
    assert [(row["start_sample"], row["stop_sample"]) for row in result["windows"]] == [
        (6400, 7424), (6912, 7936), (7424, 8448), (7936, 8960), (8448, 9472)]
    assert [row["window_index"] for row in result["windows"]] == list(range(5))
    assert spectral["schema_version"] == "eeg-paper-spectral-1.2"


def test_channel_identity_order_and_input_arrays_are_preserved():
    data = np.stack([_sine(amplitude_uv=40), _sine(amplitude_uv=20)])
    before = data.copy()
    data.setflags(write=False)
    spectral = _compute(data, ids=["z-id", "a-id"], names=["EEG", "EEG"])
    saved_spectral = deepcopy(spectral)
    result = compute_alpha_power(data, spectral)
    assert result["channel_ids"] == ["z-id", "a-id"]
    assert result["channel_names"] == ["EEG", "EEG"]
    assert [row["alpha_power_uv2"] for row in result["full"]] == pytest.approx([800, 200])
    assert [row["channel_id"] for row in result["windows"][:4]] == ["z-id", "a-id", "z-id", "a-id"]
    np.testing.assert_array_equal(data, before)
    assert spectral == saved_spectral


@pytest.mark.parametrize("alteration", ["values", "shape", "order", "fingerprint", "channels", "coverage", "units", "frequency_grid"])
def test_inconsistent_input_or_spectral_metadata_is_rejected(alteration):
    data = np.stack([_sine(amplitude_uv=40), _sine()])
    spectral = _compute(data)
    if alteration == "values":
        data = data + 1e-6
    elif alteration == "shape":
        data = data[:, :-1]
    elif alteration == "order":
        data = data[::-1]
    elif alteration == "fingerprint":
        spectral["input"]["data_sha256"] = "wrong"
    elif alteration == "channels":
        spectral["psd"]["channel_ids"] = spectral["psd"]["channel_ids"][::-1]
    elif alteration == "coverage":
        spectral["psd"]["estimator"]["covered_samples"] -= 1
    elif alteration == "units":
        spectral["input"]["unit"] = "mV"
    elif alteration == "frequency_grid":
        spectral["psd"]["frequencies_hz"][3] += .01
    with pytest.raises(ValueError):
        compute_alpha_power(data, spectral)
