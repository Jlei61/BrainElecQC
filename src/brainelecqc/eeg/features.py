"""EEG measurements on a continuous, calibrated channel-by-time array.

All sample intervals are half open. Input arrays are read only to this module;
the returned dictionary contains ordinary JSON values. Spectra use SciPy Welch
with equal periodogram weights, density scaling, a Hann window and no padding.
"""

from dataclasses import asdict
import hashlib
from numbers import Integral, Real

import numpy as np
from scipy.signal import find_peaks, hilbert, welch

from .config import EEGConfig
from .methods import measurement_methods


FEATURE_VERSION = "eeg-features-1.1"
FEATURE_KEYS = (
    "rms_uv", "ptp_uv", "flat_fraction", "large_amplitude_fraction", "relative_rms",
    "low_frequency_fraction", "high_frequency_fraction", "alpha_peak_hz",
    "alpha_prominence_db", "line_50_db", "line_60_db", "ecg_correlation",
    "eog_correlation", "emg_envelope_correlation",
)


def _numeric_array(value, ndim, name):
    arr = np.asarray(value)
    if arr.ndim != ndim or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}-dimensional array")
    arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite samples")
    if arr.size and np.max(np.abs(arr)) > 1e100:
        raise ValueError(f"{name} magnitude exceeds the numeric feature range")
    return arr


def _validate(data_volts, sfreq, channel_ids, channel_names, start_sample,
              start_seconds, references, config):
    data = _numeric_array(data_volts, 2, "data_volts")
    if 0 in data.shape:
        raise ValueError("data_volts must contain channels and samples")
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a positive finite sampling rate")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if (isinstance(start_seconds, bool) or not isinstance(start_seconds, Real)
            or not np.isfinite(start_seconds)):
        raise ValueError("start_seconds must be finite")
    if not np.isfinite(data.shape[1] / float(sfreq) + float(start_seconds)):
        raise ValueError("the sample rate and time origin must produce a finite stop time")
    if isinstance(channel_ids, (str, bytes)):
        raise ValueError("channel_ids must contain one stable ID per channel")
    ids = list(channel_ids)
    if (len(ids) != len(data) or any(not isinstance(x, str) or not x.strip() for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("channel_ids must be unique, nonempty strings matching the channels")
    names = ids.copy() if channel_names is None else list(channel_names)
    if (isinstance(channel_names, (str, bytes)) or len(names) != len(ids)
            or any(not isinstance(x, str) for x in names)):
        raise ValueError("channel_names must match channels; repeated names are supported")
    if references is None:
        references = {}
    if not isinstance(references, dict) or set(references) - {"ecg", "eog", "emg"}:
        raise ValueError("references accepts ecg, eog and emg arrays")
    refs = {}
    for key, value in references.items():
        refs[key] = _numeric_array(value, 1, f"references[{key}]")
        if len(refs[key]) != data.shape[1]:
            raise ValueError(f"references[{key}] must share the EEG sample count and time axis")
    if isinstance(config, dict):
        config = EEGConfig(**config)
    if config is None:
        config = EEGConfig()
    if not isinstance(config, EEGConfig):
        raise ValueError("config must be EEGConfig or a parameter dictionary")
    return data, float(sfreq), ids, names, refs, config


def _spectrum(data, sfreq, cfg):
    n = data.shape[-1]
    if n / sfreq < cfg.min_spectral_seconds or n < 4:
        return np.empty(0), np.empty((len(data), 0)), {
            "status": "insufficient_duration", "actual_duration_seconds": n / sfreq,
            "periodogram_count": 0, "covered_samples": 0, "tail_samples": n,
        }
    nperseg = min(n, max(4, round(cfg.welch_seconds * sfreq)))
    noverlap = min(nperseg - 1, int(nperseg * cfg.welch_overlap_fraction))
    hop = nperseg - noverlap
    starts = list(range(0, n - nperseg + 1, hop))
    frequencies, power = welch(data, fs=sfreq, window="hann", nperseg=nperseg,
                               noverlap=noverlap, nfft=nperseg, detrend="constant",
                               return_onesided=True, scaling="density", average="mean", axis=-1)
    if not np.all(np.isfinite(power)):
        raise ValueError("the sample scale and rate exceed the numeric PSD range")
    covered = starts[-1] + nperseg
    return frequencies, power, {
        "status": "computed", "method": "scipy.signal.welch", "window": "hann",
        "detrend": "constant", "scaling": "density", "average": "mean",
        "nperseg": nperseg, "nfft": nperseg, "noverlap": noverlap,
        "frequency_resolution_hz": sfreq / nperseg,
        "actual_welch_seconds": nperseg / sfreq,
        "periodogram_count": len(starts), "periodogram_start_offsets": starts,
        "periodogram_weights": [1.0 / len(starts)] * len(starts),
        "covered_samples": covered, "covered_seconds": covered / sfreq,
        "tail_samples": n - covered, "zero_padding_samples": 0,
        "weighting": "equal complete Welch periodograms; coverage counts overlapping samples once",
    }


def _harmonic_mask(frequencies, cfg):
    mask = np.zeros(len(frequencies), dtype=bool)
    for fundamental in (50.0, 60.0):
        for center in np.arange(fundamental, frequencies[-1] + 1, fundamental):
            mask |= np.abs(frequencies - center) <= cfg.line_peak_half_width_hz
    return mask


def _band_power(frequencies, power, low, high, exclude=None):
    mask = (frequencies >= low) & (frequencies < high)
    if exclude is not None:
        mask &= ~exclude
    if not np.any(mask):
        return None
    return float(np.sum(power[mask]) * (frequencies[1] - frequencies[0]))


def _background(frequencies, power, cfg):
    mask = ((frequencies >= cfg.background_fit_band[0])
            & (frequencies < cfg.background_fit_band[1]) & (power > 0))
    mask &= ~_harmonic_mask(frequencies, cfg)
    mask &= ~((frequencies >= cfg.alpha_band[0] - 1) & (frequencies <= cfg.alpha_band[1] + 1))
    if np.count_nonzero(mask) < 8 or not np.any(power > 0):
        return {"status": "insufficient_bandwidth", "slope": None, "intercept_log10": None,
                "rmse_db": None, "fit_range_hz": list(cfg.background_fit_band)}
    x = np.log10(frequencies[mask])
    y = np.log10(np.maximum(power[mask], np.finfo(float).tiny))
    keep = np.ones(len(x), dtype=bool)
    for _ in range(4):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        center = np.median(residual[keep])
        scale = max(1.4826 * np.median(np.abs(residual[keep] - center)), 0.05)
        next_keep = np.abs(residual - center) <= 2.5 * scale
        if np.count_nonzero(next_keep) < 8 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return {"status": "computed", "slope": float(slope), "intercept_log10": float(intercept),
            "rmse_db": float(10 * np.sqrt(np.mean((y[keep] - slope*x[keep] - intercept)**2))),
            "fit_range_hz": [float(10**x[keep].min()), float(10**x[keep].max())],
            "fitted_bin_count": int(np.count_nonzero(keep)),
            "method": "iterative clipped log10 PSD versus log10 frequency; alpha and mains excluded"}


def _alpha_peak(frequencies, power, cfg):
    low, high = cfg.alpha_band
    if frequencies[-1] < high + 1 or np.count_nonzero((frequencies >= low) & (frequencies < high)) < 3:
        return {"status": "insufficient_bandwidth", "peak_hz": None, "prominence_db": None}
    if not np.any(power > 0):
        return {"status": "no_power", "peak_hz": None, "prominence_db": None}
    floor = max(float(np.max(power)) * 1e-12, np.finfo(float).tiny)
    logpower = 10 * np.log10(np.maximum(power, floor))
    # A peak has descending flanks. A monotonic 1/f slope has no local peak.
    wlen = max(3, 2 * int(round(2.5 / (frequencies[1] - frequencies[0]))) + 1)
    peaks, properties = find_peaks(logpower, prominence=cfg.alpha_min_prominence_db, wlen=wlen)
    candidates = [i for i, peak in enumerate(peaks) if low <= frequencies[peak] < high]
    if not candidates:
        return {"status": "no_prominent_peak", "peak_hz": None, "prominence_db": None}
    i = max(candidates, key=lambda j: properties["prominences"][j])
    return {"status": "detected", "peak_hz": float(frequencies[peaks[i]]),
            "prominence_db": float(properties["prominences"][i])}


def _line_peak(frequencies, power, center, cfg):
    if frequencies[-1] < center + cfg.line_baseline_outer_hz:
        return {"center_hz": float(center), "status": "insufficient_bandwidth",
                "peak_hz": None, "prominence_db": None, "detected": False}
    peak_mask = np.abs(frequencies - center) <= cfg.line_peak_half_width_hz
    distance = np.abs(frequencies - center)
    baseline_mask = (distance >= cfg.line_baseline_inner_hz) & (distance <= cfg.line_baseline_outer_hz)
    if np.count_nonzero(peak_mask) < 1 or np.count_nonzero(baseline_mask) < 4:
        return {"center_hz": float(center), "status": "insufficient_resolution",
                "peak_hz": None, "prominence_db": None, "detected": False}
    if not np.any(power > 0):
        return {"center_hz": float(center), "status": "no_power",
                "peak_hz": None, "prominence_db": None, "detected": False}
    floor = max(float(np.max(power)) * 1e-12, np.finfo(float).tiny)
    peak_indices = np.flatnonzero(peak_mask)
    index = peak_indices[np.argmax(power[peak_mask])]
    baseline = max(float(np.median(power[baseline_mask])), floor)
    prominence = float(10 * np.log10(max(float(power[index]), floor) / baseline))
    return {"center_hz": float(center), "status": "computed",
            "peak_hz": float(frequencies[index]), "prominence_db": prominence,
            "detected": bool(prominence >= cfg.line_min_prominence_db),
            "baseline_psd_v2_hz": baseline,
            "baseline_offsets_hz": [cfg.line_baseline_inner_hz, cfg.line_baseline_outer_hz]}


def _spectral_features(frequencies, power, cfg):
    empty = {key: None for key in ("low_frequency_fraction", "high_frequency_fraction",
                                  "alpha_peak_hz", "alpha_prominence_db", "line_50_db", "line_60_db")}
    if not len(frequencies):
        return {**empty, "status": "insufficient_duration", "alpha_status": "insufficient_duration",
                "line_50_status": "insufficient_duration", "line_60_status": "insufficient_duration",
                "line_frequency": "insufficient", "line_harmonics": [], "band_powers_v2": {},
                "background": {"status": "insufficient_duration", "slope": None}}
    nyquist = float(frequencies[-1])
    total_hi = min(cfg.total_power_band[1], nyquist)
    total = _band_power(frequencies, power, cfg.total_power_band[0], total_hi)
    low = (_band_power(frequencies, power, *cfg.low_frequency_band)
           if nyquist >= cfg.low_frequency_band[1] else None)
    high_hi = min(cfg.high_frequency_band[1], nyquist)
    excluded = _harmonic_mask(frequencies, cfg)
    high = (_band_power(frequencies, power, cfg.high_frequency_band[0], high_hi, excluded)
            if high_hi > cfg.high_frequency_band[0] + 5 else None)
    alpha = _alpha_peak(frequencies, power, cfg)
    line50, line60 = (_line_peak(frequencies, power, center, cfg) for center in (50, 60))
    detected = [str(f) for f, result in ((50, line50), (60, line60)) if result["detected"]]
    mains = "both" if len(detected) == 2 else detected[0] if detected else (
        "none" if all(item["status"] == "computed" for item in (line50, line60)) else "insufficient")
    harmonics = []
    for fundamental in (50, 60):
        for center in range(fundamental * 2, int(nyquist) + 1, fundamental):
            item = _line_peak(frequencies, power, center, cfg)
            harmonics.append({"fundamental_hz": fundamental, "harmonic": center // fundamental, **item})
    band_powers = {name: _band_power(frequencies, power, lo, hi) if nyquist >= hi else None
                   for name, (lo, hi) in {"delta": (0.5, 4), "theta": (4, 8), "alpha": cfg.alpha_band,
                                         "beta": (13, 30)}.items()}
    band_powers.update({"total": total, "high_frequency_excluding_mains": high})
    return {"status": "computed" if total and total > 0 else "no_power",
            "low_frequency_fraction": low / total if low is not None and total and total > 0 else None,
            "high_frequency_fraction": high / total if high is not None and total and total > 0 else None,
            "alpha_peak_hz": alpha["peak_hz"], "alpha_prominence_db": alpha["prominence_db"],
            "alpha_status": alpha["status"], "line_50_db": line50["prominence_db"],
            "line_60_db": line60["prominence_db"], "line_50_status": line50["status"],
            "line_60_status": line60["status"], "line_50": line50, "line_60": line60,
            "line_frequency": mains, "line_harmonics": harmonics, "band_powers_v2": band_powers,
            "background": _background(frequencies, power, cfg),
            "power_bands_hz": {"total": [cfg.total_power_band[0], total_hi],
                               "low": list(cfg.low_frequency_band),
                               "high": [cfg.high_frequency_band[0], high_hi]},
            "low_frequency_status": ("insufficient_bandwidth" if low is None else
                                      "computed" if total and total > 0 else "no_power"),
            "high_frequency_status": ("insufficient_bandwidth" if high is None else
                                       "computed" if total and total > 0 else "no_power")}


def _correlation(x, y):
    xc, yc = x - np.mean(x), y - np.mean(y)
    sx, sy = float(np.linalg.norm(xc)), float(np.linalg.norm(yc))
    floor_x = np.finfo(float).eps * max(float(np.linalg.norm(x)), np.finfo(float).tiny) * 32
    floor_y = np.finfo(float).eps * max(float(np.linalg.norm(y)), np.finfo(float).tiny) * 32
    if sx <= floor_x or sy <= floor_y:
        return None
    return float(np.clip(np.dot(xc / sx, yc / sy), -1, 1))


def _emg_envelope(data, sfreq, cfg):
    if len(data) / sfreq < cfg.min_spectral_seconds:
        return None
    high = min(cfg.high_frequency_band[1], sfreq * 0.45)
    low = cfg.high_frequency_band[0]
    if high <= low + 5:
        return None
    f = np.fft.rfftfreq(len(data), 1 / sfreq)
    mask = (f >= low) & (f < high) & ~_harmonic_mask(f, cfg)
    if not np.any(mask):
        return None
    filtered = np.fft.irfft(np.fft.rfft(data) * mask, n=len(data))
    return np.abs(hilbert(filtered))


def compute_eeg_features(data_volts, sfreq, channel_ids, channel_names=None, *,
                         start_sample=0, start_seconds=0.0, references=None, config=None):
    """Measure one continuous EEG segment; volt inputs and stable IDs are explicit.

    References share the EEG sample rate, count and time origin. ECG/EOG use
    signed, zero-lag Pearson correlation. EMG compares high-frequency envelopes.
    """
    data, sfreq, ids, names, refs, cfg = _validate(
        data_volts, sfreq, channel_ids, channel_names, start_sample, start_seconds, references, config)
    start_sample = int(start_sample)
    start_seconds = float(start_seconds)
    nch, n = data.shape
    window_samples = max(1, round(cfg.window_seconds * sfreq))
    offsets = list(range(0, n, window_samples))
    values = {key: [[] for _ in ids] for key in FEATURE_KEYS}
    statuses = {key: [[] for _ in ids] for key in FEATURE_KEYS}
    windows, window_spectra = [], []
    for wi, offset in enumerate(offsets):
        stop = min(offset + window_samples, n)
        block = data[:, offset:stop]
        rms = np.std(block, axis=1) * 1e6
        median_rms = float(np.median(rms))
        f, psd, metadata = _spectrum(block, sfreq, cfg)
        windows.append({"window_index": wi, "start_sample": start_sample + offset,
                        "stop_sample": start_sample + stop, "start_seconds": start_seconds + offset / sfreq,
                        "stop_seconds": start_seconds + stop / sfreq, "sample_count": stop - offset})
        window_spectra.append(metadata)
        reference_envelope = _emg_envelope(refs["emg"][offset:stop], sfreq, cfg) if "emg" in refs else None
        for ci, row in enumerate(block):
            spectral = _spectral_features(f, psd[ci], cfg)
            row_values = {key: spectral.get(key) for key in FEATURE_KEYS}
            row_values.update({"rms_uv": float(rms[ci]), "ptp_uv": float(np.ptp(row) * 1e6),
                               "flat_fraction": float(np.mean(np.abs(np.diff(row)) * 1e6 <= cfg.flat_step_uv)) if len(row) > 1 else None,
                               "large_amplitude_fraction": float(np.mean(np.abs(row - np.median(row)) * 1e6 > cfg.large_amplitude_uv)),
                               "relative_rms": float(rms[ci] / median_rms) if median_rms > 0 else None})
            for key in ("ecg", "eog"):
                row_values[f"{key}_correlation"] = _correlation(row, refs[key][offset:stop]) if key in refs else None
            if reference_envelope is not None:
                envelope = _emg_envelope(row, sfreq, cfg)
                row_values["emg_envelope_correlation"] = _correlation(envelope, reference_envelope)
            for key in FEATURE_KEYS:
                value = row_values[key]
                values[key][ci].append(value)
                if value is not None:
                    status = "computed"
                elif key in {"ecg_correlation", "eog_correlation", "emg_envelope_correlation"}:
                    reference_key = key.split("_")[0]
                    status = "missing_reference" if reference_key not in refs else (
                        "insufficient_bandwidth_or_duration" if reference_key == "emg" and reference_envelope is None else "constant_signal")
                elif key.startswith("alpha_"):
                    status = spectral["alpha_status"]
                elif key.startswith("line_"):
                    status = spectral[f"{key[:-3]}_status"]
                elif key == "high_frequency_fraction":
                    status = spectral.get("high_frequency_status", spectral["status"])
                elif key == "low_frequency_fraction":
                    status = spectral.get("low_frequency_status", spectral["status"])
                elif key == "relative_rms":
                    status = "zero_group_rms"
                elif key == "flat_fraction":
                    status = "insufficient_samples"
                else:
                    status = spectral["status"]
                statuses[key][ci].append(status)
    f, psd, metadata = _spectrum(data, sfreq, cfg)
    summaries = []
    for ci, row in enumerate(data):
        summary = _spectral_features(f, psd[ci], cfg)
        alpha_statuses = statuses["alpha_peak_hz"][ci]
        eligible_alpha = sum(x in {"computed", "no_prominent_peak", "no_power"} for x in alpha_statuses)
        summary.update({"channel_id": ids[ci], "channel_name": names[ci], "channel_index": ci,
                        "rms_uv": float(np.std(row) * 1e6), "ptp_uv": float(np.ptp(row) * 1e6),
                        "alpha_window_fraction": sum(x == "computed" for x in alpha_statuses) / eligible_alpha if eligible_alpha else None,
                        "alpha_assessed_window_count": eligible_alpha,
                        "background_slope": summary["background"]["slope"]})
        summaries.append(summary)
    return {"schema_version": 1, "feature_version": FEATURE_VERSION,
            "data_sha256": hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes(order="C")).hexdigest(),
            "channel_ids": ids, "channel_names": names, "sampling_rate_hz": sfreq,
            "start_sample": start_sample, "stop_sample": start_sample + n,
            "start_seconds": start_seconds, "stop_seconds": start_seconds + n / sfreq,
            "config": asdict(cfg), "measurement_methods": measurement_methods(),
            "windows": windows, "window_features": values,
            "window_feature_status": statuses, "window_spectral_metadata": window_spectra,
            "frequencies_hz": f.tolist(), "psd_v2_hz": psd.tolist(), "spectral_metadata": metadata,
            "channel_features": summaries,
            "reference_status": {key: {"status": "provided" if key in refs else "missing_reference",
                                       "method": "signed zero-lag Pearson correlation" if key != "emg" else
                                       "Pearson correlation of FFT-band-limited Hilbert envelopes; mains bins excluded",
                                       "sample_alignment": "same sampling rate, count and time origin"}
                                 for key in ("ecg", "eog", "emg")},
            "units": {"input": "V", "rms_uv": "uV", "ptp_uv": "uV", "psd_v2_hz": "V^2/Hz",
                      "band_powers_v2": "V^2", "frequency": "Hz", "prominence": "dB"},
            "conventions": {"time_interval": "half_open", "window_axis": "channel, window",
                            "psd_axis": "channel, frequency", "band_integration": "sum PSD * df over [low, high)",
                            "rms_uv": "standard deviation after subtracting the window mean",
                            "large_amplitude_fraction": "absolute deviation from the window median",
                            "flat_fraction": "fraction of adjacent sample differences at or below flat_step_uv"}}
