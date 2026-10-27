"""Absolute and relative alpha power on explicitly bounded Welch spectra.

Power is the area under the observed density spectrum. The two band edges are
linearly interpolated inside the sampled frequency range, then joined with all
interior frequency points for trapezoidal integration. Time rows refer to the
samples actually used by Welch, with the uncovered recording tail kept apart.
"""

import hashlib
from numbers import Integral, Real

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


ALPHA_POWER_VERSION = "eeg-alpha-power-1.0"


def _band(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) for x in value)
            or not 0 <= value[0] < value[1]):
        raise ValueError(f"{name} must contain two increasing nonnegative frequencies")
    return tuple(float(x) for x in value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate(data_volts, spectral):
    data = np.asarray(data_volts)
    if (data.ndim != 2 or 0 in data.shape or data.dtype.kind not in "iuf"
            or not np.all(np.isfinite(data))):
        raise ValueError("data_volts must contain finite real channel-by-sample values")
    data = np.asarray(data, dtype=np.float64)
    try:
        origin, psd, estimator = spectral["input"], spectral["psd"], spectral["psd"]["estimator"]
        ids, names = list(origin["channel_ids"]), list(origin["channel_names"])
        digest = hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes()).hexdigest()
        if (data.shape != (len(ids), origin["n_samples"]) or digest != origin["data_sha256"]):
            raise ValueError("alpha power requires the exact spectral-analysis input and fingerprint")
        if (any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids) or len(names) != len(ids)
                or any(not isinstance(value, str) for value in names) or psd["channel_ids"] != ids):
            raise ValueError("spectral channel identities and order must match the input")
        sfreq = origin["sfreq"]
        if (isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0):
            raise ValueError("spectral sampling rate must be finite and positive")
        start_sample = _integer(origin["start_sample"], "start_sample")
        start_seconds = origin["start_seconds"]
        if (isinstance(start_seconds, bool) or not isinstance(start_seconds, Real)
                or not np.isfinite(start_seconds + data.shape[1] / sfreq)):
            raise ValueError("spectral time origin must be finite")
        if origin["unit"] != "V" or estimator["density_unit"] != "V^2/Hz":
            raise ValueError("alpha power requires volts and a V^2/Hz density spectrum")
        if any(estimator[key] != value for key, value in {
                "window": "hann", "detrend": "constant", "average": "mean"}.items()):
            raise ValueError("alpha power requires the recorded Hann, demeaned, mean-Welch estimator")
        length = _integer(estimator["nperseg"], "nperseg", 1)
        overlap = _integer(estimator["noverlap"], "noverlap")
        nfft = _integer(estimator["nfft"], "nfft", 1)
        if not overlap < length <= data.shape[1] or nfft < length:
            raise ValueError("Welch segment, overlap and FFT lengths must match the input")
        starts = list(range(0, data.shape[1] - length + 1, length - overlap))
        covered = starts[-1] + length
        if (estimator["covered_samples"] != covered or estimator["tail_samples"] != data.shape[1] - covered
                or estimator["periodogram_count"] != len(starts)
                or estimator["periodogram_start_offsets"] != starts):
            raise ValueError("spectral Welch coverage must match its segment and overlap settings")
        frequencies = np.asarray(psd["frequencies_hz"], dtype=float)
        power = np.asarray(psd["power_v2_hz"], dtype=float)
        if (frequencies.ndim != 1 or not len(frequencies) or power.shape != (len(ids), len(frequencies))
                or not np.all(np.isfinite(power)) or np.any(power < 0)
                or not np.array_equal(frequencies, np.fft.rfftfreq(nfft, 1 / sfreq))):
            raise ValueError("spectral PSD must have the recorded frequency grid and finite nonnegative powers")
    except (KeyError, TypeError) as error:
        raise ValueError("spectral must contain complete input, PSD and Welch metadata") from error
    cfg = spectral.get("config", {})
    alpha = _band(cfg.get("alpha_band_hz", (8, 13)), "alpha_band_hz")
    reference = _band(cfg.get("fit_range_hz", (2, 40)), "reference_band_hz")
    if not reference[0] <= alpha[0] < alpha[1] <= reference[1]:
        raise ValueError("alpha_band_hz must lie inside reference_band_hz")
    return data, ids, names, digest, float(sfreq), start_sample, float(start_seconds), frequencies, power, estimator, starts, covered, alpha, reference


def _integrate_band(frequencies, power, band):
    """Integrate an observed PSD with interpolated edges; return power in µV²."""
    frequencies, power = np.asarray(frequencies), np.asarray(power)
    lower, upper = band
    if len(frequencies) < 2:
        return None, "insufficient_frequency_grid"
    if lower < frequencies[0] or upper > frequencies[-1]:
        return None, "insufficient_bandwidth"
    if not np.any((frequencies >= lower) & (frequencies <= upper)):
        return None, "insufficient_frequency_resolution"
    inside = (frequencies > lower) & (frequencies < upper)
    x = np.r_[lower, frequencies[inside], upper]
    y = np.r_[np.interp(lower, frequencies, power), power[inside], np.interp(upper, frequencies, power)]
    value = float(trapezoid(y, x=x) * 1e12)
    if not np.isfinite(value):
        raise ValueError("the PSD scale exceeds finite band-power range")
    if value <= 0:
        return None, "no_power"
    return value, "computed"


def _power_row(frequencies, power, alpha_band, reference_band):
    alpha, alpha_status = _integrate_band(frequencies, power, alpha_band)
    reference, reference_status = _integrate_band(frequencies, power, reference_band)
    relative = None
    if alpha_status == "computed" and reference_status == "computed":
        relative = float(100 * alpha / reference)
        relative_status, status = "computed", "computed"
    elif alpha_status == "computed":
        relative_status, status = f"reference_{reference_status}", "alpha_only"
    else:
        relative_status, status = f"alpha_{alpha_status}", alpha_status
    return {"alpha_power_uv2": alpha, "reference_power_uv2": reference,
            "relative_alpha_percent": relative, "status": status,
            "alpha_power_status": alpha_status, "reference_power_status": reference_status,
            "relative_alpha_status": relative_status}


def _location(left, right, sfreq, start_sample, start_seconds):
    return {"start_offset": left, "stop_offset": right,
            "start_sample": start_sample + left, "stop_sample": start_sample + right,
            "start_seconds": start_seconds + left / sfreq, "stop_seconds": start_seconds + right / sfreq}


def compute_alpha_power(data_volts, spectral):
    """Describe alpha power using the input spectrum and its Welch time windows.

    The observed PSD includes periodic and aperiodic activity. Relative alpha is
    100 times alpha-band power divided by power in the configured reference band.
    Measurement statuses and the source clock are preserved independently of QC
    scoring or decisions.
    """
    (data, ids, names, digest, sfreq, start_sample, start_seconds, frequencies, power,
     estimator, starts, covered, alpha_band, reference_band) = _validate(data_volts, spectral)
    length, overlap, nfft = estimator["nperseg"], estimator["noverlap"], estimator["nfft"]
    config = {"alpha_band_hz": list(alpha_band), "reference_band_hz": list(reference_band),
              "integration": "linear interpolation at both band boundaries, then trapezoidal integration",
              "frequency_support": "band contained in grid; at least two grid points and one native in-band point",
              "source_psd_unit": "V^2/Hz", "power_unit": "uV^2",
              "relative_alpha_definition": "100 * alpha_power / reference_power",
              "window": "hann", "detrend": "constant", "scaling": "density",
              "nperseg": length, "noverlap": overlap, "nfft": nfft,
              "step_samples": length - overlap, "window_seconds": length / sfreq,
              "step_seconds": (length - overlap) / sfreq,
              "periodogram_count": len(starts), "frequency_resolution_hz": sfreq / nfft}
    result = {"schema_version": ALPHA_POWER_VERSION, "data_sha256": digest,
              "channel_ids": ids, "channel_names": names, "sfreq": sfreq,
              "alpha_band_hz": list(alpha_band), "reference_band_hz": list(reference_band), "config": config,
              "input_interval": _location(0, data.shape[1], sfreq, start_sample, start_seconds),
              "tail_samples": data.shape[1] - covered,
              "tail": {**_location(covered, data.shape[1], sfreq, start_sample, start_seconds),
                       "n_samples": data.shape[1] - covered,
                       "status": "uncovered_tail" if covered < data.shape[1] else "none"},
              "full": [], "windows": []}
    for index, (identity, name) in enumerate(zip(ids, names)):
        result["full"].append({"channel_index": index, "channel_id": identity, "channel_name": name,
            **_location(0, covered, sfreq, start_sample, start_seconds),
            **_power_row(frequencies, power[index], alpha_band, reference_band)})
    for window_index, left in enumerate(starts):
        segment = data[:, left:left + length]
        window_frequencies, window_power = welch(segment, fs=sfreq, window="hann", nperseg=length,
            noverlap=0, nfft=nfft, detrend="constant", return_onesided=True,
            scaling="density", average="mean", axis=-1)
        # A constant segment has exactly zero AC power; remove mean-roundoff.
        window_power[np.ptp(segment, axis=-1) == 0] = 0.0
        for index, (identity, name) in enumerate(zip(ids, names)):
            result["windows"].append({"channel_index": index, "channel_id": identity, "channel_name": name,
                "window_index": window_index, **_location(left, left + length, sfreq, start_sample, start_seconds),
                **_power_row(window_frequencies, window_power[index], alpha_band, reference_band)})
    return result
