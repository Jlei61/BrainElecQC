"""Traceable EEG spectra: Thomson line tests and FOOOF parameterization.

The line test follows the sinusoid-in-locally-white-noise model described by
Mitra & Pesaran (1999), Chronux ``ftestc`` and PREP (2015). It measures evidence;
it never subtracts a sinusoid or creates an exclusion mask. Frequency scans,
channels and overlapping time windows form one Bonferroni family per call.
"""

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.signal import detrend, welch
from scipy.signal.windows import dpss
from scipy.stats import f as f_distribution


SPECTRAL_VERSION = "eeg-paper-spectral-1.2"
FOOOF_VERSION = "1.1.1"


@dataclass(frozen=True)
class SpectralQCConfig:
    """Explicit analysis settings; thresholds select review evidence only."""

    line_window_seconds: float = 4.0
    line_step_seconds: float = 1.0
    line_bandwidth_hz: float = 2.0
    line_scan_half_width_hz: float = 2.0
    line_fundamentals_hz: tuple = (50.0, 60.0)
    line_family_alpha: float = 0.01
    welch_seconds: float = 4.0
    welch_overlap_fraction: float = 0.5
    parameter_window_seconds: float = 30.0
    parameter_min_seconds: float = 10.0
    fit_range_hz: tuple = (2.0, 40.0)
    alpha_band_hz: tuple = (8.0, 13.0)
    peak_width_limits_hz: tuple = (1.0, 8.0)
    max_n_peaks: int = 6
    min_peak_height_log10: float = 0.0
    peak_threshold_sd: float = 2.0

    def __post_init__(self):
        pairs = ("fit_range_hz", "alpha_band_hz", "peak_width_limits_hz")
        for key in pairs:
            values = getattr(self, key)
            if (not isinstance(values, (tuple, list)) or len(values) != 2
                    or any(isinstance(x, bool) or not isinstance(x, Real)
                           or not np.isfinite(x) for x in values)
                    or not 0 < values[0] < values[1]):
                raise ValueError(f"{key} must contain two increasing positive values")
        if not (self.fit_range_hz[0] < self.alpha_band_hz[0]
                < self.alpha_band_hz[1] < self.fit_range_hz[1]):
            raise ValueError("alpha_band_hz must be inside fit_range_hz")
        sequence = self.line_fundamentals_hz
        if (not isinstance(sequence, (tuple, list)) or not sequence
                or any(isinstance(x, bool) or not isinstance(x, Real)
                       or not np.isfinite(x) or x <= 0 for x in sequence)
                or len(set(sequence)) != len(sequence)):
            raise ValueError("line_fundamentals_hz must contain unique positive frequencies")
        for key, value in asdict(self).items():
            if key in pairs or key == "line_fundamentals_hz":
                continue
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be a nonnegative finite number")
            if value == 0 and key not in {"min_peak_height_log10", "welch_overlap_fraction"}:
                raise ValueError(f"{key} must be positive")
        if not isinstance(self.max_n_peaks, Integral):
            raise ValueError("max_n_peaks must be an integer")
        if not 0 < self.line_family_alpha < 1 or not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError("line_family_alpha and welch_overlap_fraction are probabilities")
        if self.line_step_seconds > self.line_window_seconds:
            raise ValueError("line_step_seconds must not leave unanalyzed gaps")
        if self.line_bandwidth_hz * self.line_window_seconds < 3:
            raise ValueError("line bandwidth and duration must provide at least two tapers")
        if self.parameter_min_seconds > self.parameter_window_seconds:
            raise ValueError("parameter_min_seconds must not exceed parameter_window_seconds")
        if self.parameter_min_seconds < self.welch_seconds:
            raise ValueError("parameter_min_seconds must cover at least one Welch subwindow")
        if self.line_scan_half_width_hz >= min(self.line_fundamentals_hz):
            raise ValueError("line scans must stay above zero frequency")


def _validate(data_volts, sfreq, channel_ids, channel_names, start_sample, start_seconds, config):
    data = np.asarray(data_volts)
    if (data.ndim != 2 or 0 in data.shape or data.dtype.kind not in "iuf"
            or not np.all(np.isfinite(data)) or np.max(np.abs(data)) > 1e100):
        raise ValueError("data_volts must be finite real channel-by-sample values")
    data = np.asarray(data, dtype=float)
    if (isinstance(sfreq, bool) or not isinstance(sfreq, Real)
            or not np.isfinite(sfreq) or sfreq <= 0):
        raise ValueError("sfreq must be a positive finite sampling rate")
    if (isinstance(start_sample, bool) or not isinstance(start_sample, Integral)
            or start_sample < 0):
        raise ValueError("start_sample must be a nonnegative integer")
    if (isinstance(start_seconds, bool) or not isinstance(start_seconds, Real)
            or not np.isfinite(start_seconds + data.shape[1] / sfreq)):
        raise ValueError("start_seconds and sfreq must give a finite time axis")
    if isinstance(channel_ids, (str, bytes)):
        raise ValueError("channel_ids must contain unique IDs matching channel count")
    ids = list(channel_ids)
    if (len(ids) != len(data) or any(not isinstance(x, str) or not x.strip() for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("channel_ids must contain unique IDs matching channel count")
    if isinstance(channel_names, (str, bytes)):
        raise ValueError("channel_names must match channel count")
    names = ids.copy() if channel_names is None else list(channel_names)
    if len(names) != len(ids) or any(not isinstance(x, str) for x in names):
        raise ValueError("channel_names must match channel count")
    cfg = SpectralQCConfig(**config) if isinstance(config, dict) else config
    cfg = SpectralQCConfig() if cfg is None else cfg
    if not isinstance(cfg, SpectralQCConfig):
        raise ValueError("config must be SpectralQCConfig or a parameter dictionary")
    return data, float(sfreq), ids, names, cfg


def _location(left, right, sfreq, start_sample, start_seconds):
    return {"start_offset": int(left), "stop_offset": int(right),
            "start_sample": int(start_sample + left), "stop_sample": int(start_sample + right),
            "start_seconds": float(start_seconds + left / sfreq),
            "stop_seconds": float(start_seconds + right / sfreq)}


def _channel(index, ids, names):
    return {"channel_index": index, "channel_id": ids[index], "channel_name": names[index]}


def _line_tapers(n_samples, sfreq, bandwidth_hz):
    nw = bandwidth_hz * n_samples / (2 * sfreq)
    count = int(np.floor(2 * nw)) - 1
    if count < 2 or count >= n_samples:
        raise ValueError("line window/rate/bandwidth must provide 2 <= taper count < sample count")
    # Symmetric, L2-normalized DPSS matches MATLAB dpss / PREP's [NW, 2NW-1].
    tapers, eigenvalues = dpss(n_samples, nw, Kmax=count, sym=True, norm=2, return_ratios=True)
    return tapers, {"time_half_bandwidth_product": nw, "n_tapers": count,
                    "full_bandwidth_hz": bandwidth_hz, "dpss_sym": True,
                    "dpss_norm": 2, "eigenvalues": eigenvalues.tolist(),
                    "degrees_of_freedom": [2, 2 * count - 2]}


def _thomson_spectrum(data, sfreq, tapers):
    """Return frequency, F, complex sinusoid coefficient on an unpadded grid.

    Each input row has its mean removed. The caller explicitly detrends its
    analysis copy first. F=(K-1)|A|² sum(H²)/sum|J-AH|², with H the taper sums.
    The real sinusoid's peak amplitude is 2|A|. DC/Nyquist are never tested.
    """
    centered = data - data.mean(axis=-1, keepdims=True)
    transformed = np.fft.rfft(centered[:, None, :] * tapers[None, :, :], axis=-1)
    h = np.sum(tapers, axis=1)
    h[1::2] = 0.0  # Antisymmetric taper integrals are zero analytically.
    h2 = np.dot(h, h)
    amplitude = np.einsum("ckf,k->cf", transformed, h) / h2
    residual = transformed - amplitude[:, None, :] * h[None, :, None]
    denominator = np.sum(np.abs(residual) ** 2, axis=1)
    numerator = (len(tapers) - 1) * np.abs(amplitude) ** 2 * h2
    statistic = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    # Exact zero residual and positive fitted power is an infinite F statistic.
    statistic[(denominator == 0) & (numerator > 0)] = np.inf
    return np.fft.rfftfreq(data.shape[1], 1 / sfreq), statistic, amplitude


def _targets(sfreq, cfg):
    unique = {}
    unavailable = []
    for fundamental in cfg.line_fundamentals_hz:
        if fundamental + cfg.line_scan_half_width_hz >= sfreq / 2:
            unavailable.append({"fundamental_hz": float(fundamental), "status": "insufficient_bandwidth"})
        for harmonic in range(1, int(np.floor((sfreq / 2) / fundamental)) + 1):
            nominal = float(fundamental * harmonic)
            if nominal + cfg.line_scan_half_width_hz >= sfreq / 2:
                continue
            unique.setdefault(nominal, []).append({"fundamental_hz": float(fundamental), "harmonic": harmonic})
    return [{"nominal_hz": nominal, "families": families} for nominal, families in sorted(unique.items())], unavailable


def _line_analysis(data, sfreq, ids, names, cfg, start_sample, start_seconds):
    length = max(1, round(cfg.line_window_seconds * sfreq))
    hop = max(1, round(cfg.line_step_seconds * sfreq))
    starts = list(range(0, data.shape[1] - length + 1, hop))
    targets, unavailable = _targets(sfreq, cfg)
    windows = [_location(left, left + length, sfreq, start_sample, start_seconds) for left in starts]
    output = {"status": "computed", "windows": windows, "observations": [], "channel_summary": [],
              "unavailable_fundamentals": unavailable, "targets": targets,
              "tail_samples": data.shape[1] - (starts[-1] + length if starts else 0),
              "settings": {"window_samples": length, "step_samples": hop,
                           "detrend": "linear per window on analysis copy",
                           "fft_length": length, "frequency_resolution_hz": sfreq / length,
                           "scan_half_width_hz": cfg.line_scan_half_width_hz},
              "multiplicity": {"method": "bonferroni", "family_alpha": cfg.line_family_alpha,
                               "scope": "all channels x complete time windows x unique scanned frequency bins",
                               "hypothesis_count": 0, "per_test_alpha": None}}
    if not starts or not targets:
        output["status"] = "insufficient_duration" if not starts else "insufficient_bandwidth"
        return output
    tapers, details = _line_tapers(length, sfreq, cfg.line_bandwidth_hz)
    output["settings"].update(details)
    frequencies = np.fft.rfftfreq(length, 1 / sfreq)
    scans = [np.flatnonzero(np.abs(frequencies - row["nominal_hz"]) <= cfg.line_scan_half_width_hz)
             for row in targets]
    all_bins = np.unique(np.concatenate(scans))
    count = len(ids) * len(starts) * len(all_bins)
    output["multiplicity"].update(hypothesis_count=int(count), per_test_alpha=cfg.line_family_alpha / count,
                                  channel_count=len(ids), window_count=len(starts),
                                  unique_scanned_bins_per_window=len(all_bins),
                                  scan_bin_counts=[len(indices) for indices in scans])
    for window_index, left in enumerate(starts):
        segment = data[:, left:left + length]
        ac_range = np.ptp(segment, axis=1)
        detrended = detrend(segment, axis=-1, type="linear")
        # A precisely linear or constant input can leave only rounding residuals.
        # This tolerance addresses numerical precision, not an EEG amplitude gate.
        roundoff = 100 * np.finfo(float).eps * np.max(np.abs(segment), axis=1)
        varying = (ac_range > 0) & (np.ptp(detrended, axis=1) > roundoff)
        _, stats, amplitudes = _thomson_spectrum(detrended, sfreq, tapers)
        for channel_index in range(len(ids)):
            for target, indices in zip(targets, scans):
                values = stats[channel_index, indices]
                peak = int(np.argmax(values))
                chosen = indices[peak]
                p_values = f_distribution.sf(values, 2, 2 * len(tapers) - 2)
                p_value = float(p_values[peak]) if varying[channel_index] else None
                corrected = min(1.0, p_value * count) if p_value is not None else None
                row = {**_channel(channel_index, ids, names), **windows[window_index],
                       "window_index": window_index, **target,
                       "status": ("computed" if varying[channel_index] else
                                  "no_ac_power" if ac_range[channel_index] == 0 else "no_residual_power"),
                       "peak_frequency_hz": float(frequencies[chosen]) if varying[channel_index] else None,
                       "f_statistic": float(values[peak]) if np.isfinite(values[peak]) and varying[channel_index] else None,
                       "f_is_infinite": bool(np.isinf(values[peak]) and varying[channel_index]),
                       "p_value": p_value, "p_corrected": corrected,
                       "amplitude_uv": float(2 * np.abs(amplitudes[channel_index, chosen]) * 1e6) if varying[channel_index] else None,
                       "phase_radians": float(np.angle(amplitudes[channel_index, chosen])) if varying[channel_index] else None,
                       "flagged": bool(corrected is not None and corrected < cfg.line_family_alpha),
                       "scan_frequencies_hz": frequencies[indices].tolist(),
                       "scan_f_statistics": [float(x) if np.isfinite(x) else None for x in values] if varying[channel_index] else [],
                       "scan_p_values": p_values.tolist() if varying[channel_index] else []}
                output["observations"].append(row)
    for index in range(len(ids)):
        for target in targets:
            rows = [x for x in output["observations"] if x["channel_index"] == index and x["nominal_hz"] == target["nominal_hz"]]
            assessed = [x for x in rows if x["status"] == "computed"]
            flagged = [x for x in assessed if x["flagged"]]
            coverage = np.zeros(data.shape[1], dtype=bool)
            for row in flagged:
                coverage[row["start_offset"]:row["stop_offset"]] = True
            output["channel_summary"].append({**_channel(index, ids, names), **target,
                "assessed_window_count": len(assessed), "flagged_window_count": len(flagged),
                "flagged_window_fraction": len(flagged) / len(assessed) if assessed else None,
                "flagged_union_seconds": int(coverage.sum()) / sfreq,
                "flagged_union_fraction_of_input": float(coverage.mean()),
                "minimum_corrected_p": min((x["p_corrected"] for x in assessed), default=None)})
    return output


def _welch(data, sfreq, cfg):
    length = min(data.shape[-1], max(2, round(cfg.welch_seconds * sfreq)))
    overlap = min(length - 1, int(length * cfg.welch_overlap_fraction))
    frequencies, power = welch(data, fs=sfreq, window="hann", nperseg=length,
                              noverlap=overlap, nfft=length, detrend="constant",
                              return_onesided=True, scaling="density", average="mean", axis=-1)
    # Mean subtraction can leave a machine-precision residual for a constant
    # value (e.g. 2e-6). Its exact AC power is zero.
    power[np.ptp(data, axis=-1) == 0] = 0.0
    if not np.all(np.isfinite(power)):
        raise ValueError("sample scale or sampling rate exceeds finite PSD range")
    starts = list(range(0, data.shape[-1] - length + 1, length - overlap))
    return frequencies, power, {"method": "scipy.signal.welch", "window": "hann",
        "detrend": "constant", "density_unit": "V^2/Hz", "average": "mean",
        "nperseg": length, "noverlap": overlap, "nfft": length,
        "frequency_resolution_hz": sfreq / length, "periodogram_count": len(starts),
        "periodogram_start_offsets": starts, "covered_samples": starts[-1] + length,
        "tail_samples": data.shape[-1] - starts[-1] - length}


def _fooof_class():
    installed = version("fooof")
    if installed != FOOOF_VERSION:
        raise RuntimeError(f"spectral parameterization requires fooof=={FOOOF_VERSION}; found {installed}")
    # Version is recorded in every result. This pinned API intentionally supports
    # reproducibility with the paper's maintained 1.x implementation.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", DeprecationWarning)
        from fooof import FOOOF
    return FOOOF


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _fit_one(frequencies, power, cfg, duration, estimator, fooof_class):
    base = {"status": "computed", "alpha_status": "unassessed", "alpha_peaks": [], "peaks": [],
            "requested_fit_range_hz": list(cfg.fit_range_hz), "fit_frequencies_hz": [],
            "aperiodic_mode": "fixed", "aperiodic_parameters": None,
            "r_squared": None, "mae_log10_power": None, "rmse_log10_power": None,
            "model_log10_v2_hz": [], "aperiodic_log10_v2_hz": [],
            "observed_log10_v2_hz": [], "residual_log10_power": []}
    if duration < cfg.parameter_min_seconds or estimator["periodogram_count"] < 4:
        base["status"] = "insufficient_duration"
        return base
    if frequencies[-1] < cfg.fit_range_hz[1]:
        base["status"] = "insufficient_bandwidth"
        return base
    band = (frequencies >= cfg.fit_range_hz[0]) & (frequencies <= cfg.fit_range_hz[1])
    if not np.any(power[band] > 0):
        base["status"] = "no_power"
        return base
    if np.any(power[band] <= 0):
        base["status"] = "nonpositive_power_in_fit_band"
        return base
    if cfg.peak_width_limits_hz[0] < 2 * estimator["frequency_resolution_hz"]:
        base["status"] = "insufficient_frequency_resolution"
        return base
    model = fooof_class(peak_width_limits=cfg.peak_width_limits_hz, max_n_peaks=cfg.max_n_peaks,
                        min_peak_height=cfg.min_peak_height_log10,
                        peak_threshold=cfg.peak_threshold_sd, aperiodic_mode="fixed", verbose=False)
    from fooof.core.errors import FitError

    model.set_debug_mode(True)
    try:
        model.fit(frequencies, power, cfg.fit_range_hz)
    except FitError as error:
        base.update(status="fit_failed", fit_error_type=type(error).__name__, fit_error_message=str(error))
        return base
    if not model.has_model:
        base["status"] = "fit_failed"
        return base
    model_values = np.asarray(model.fooofed_spectrum_)
    background = np.asarray(model.get_model("aperiodic", "log"))
    observed = np.asarray(model.power_spectrum)
    residual = observed - model_values
    peaks = [{"center_frequency_hz": float(p[0]), "height_log10_power": float(p[1]),
              "height_db": float(10 * p[1]), "bandwidth_2sigma_hz": float(p[2]),
              "gaussian_height_log10_power": float(g[1]), "gaussian_sigma_hz": float(g[2])}
             for p, g in zip(model.peak_params_, model.gaussian_params_)]
    alpha = [p for p in peaks if cfg.alpha_band_hz[0] <= p["center_frequency_hz"] <= cfg.alpha_band_hz[1]]
    base.update(alpha_status="peak_present" if alpha else "no_fitted_peak", alpha_peaks=alpha, peaks=peaks,
                fit_frequencies_hz=model.freqs.tolist(),
                actual_fit_range_hz=[float(model.freqs[0]), float(model.freqs[-1])],
                aperiodic_parameters={"offset_log10_v2_hz": float(model.aperiodic_params_[0]),
                                      "exponent": float(model.aperiodic_params_[1])},
                r_squared=_finite(model.r_squared_), mae_log10_power=_finite(model.error_),
                rmse_log10_power=float(np.sqrt(np.mean(residual ** 2))),
                model_log10_v2_hz=model_values.tolist(), aperiodic_log10_v2_hz=background.tolist(),
                observed_log10_v2_hz=observed.tolist(), residual_log10_power=residual.tolist())
    return base


def compute_spectral_qc(data_volts, sfreq, channel_ids, channel_names=None, *, start_sample=0,
                        start_seconds=0.0, config=None):
    """Compute paper-based spectral evidence with channel and source-time identity."""
    data, sfreq, ids, names, cfg = _validate(data_volts, sfreq, channel_ids, channel_names,
                                           start_sample, start_seconds, config)
    fooof_class = _fooof_class()
    origin = _location(0, data.shape[1], sfreq, start_sample, start_seconds)
    frequencies, power, estimator = _welch(data, sfreq, cfg)
    parameterization = {"method": "FOOOF", "package_version": FOOOF_VERSION,
                        "curve_unit": "log10(V^2/Hz)", "peak_height_unit": "log10 power above aperiodic background",
                        "full": [], "windows": [],
                        "window_seconds": cfg.parameter_window_seconds,
                        "minimum_seconds": cfg.parameter_min_seconds,
                        "settings": {"aperiodic_mode": "fixed", "peak_width_limits": list(cfg.peak_width_limits_hz),
                                     "max_n_peaks": cfg.max_n_peaks, "min_peak_height": cfg.min_peak_height_log10,
                                     "peak_threshold": cfg.peak_threshold_sd, "fit_range_hz": list(cfg.fit_range_hz),
                                     "alpha_band_hz": list(cfg.alpha_band_hz)}}
    for index in range(len(ids)):
        parameterization["full"].append({**_channel(index, ids, names), **origin,
            "estimator": estimator, **_fit_one(frequencies, power[index], cfg, data.shape[1] / sfreq, estimator, fooof_class)})
    step = max(1, round(cfg.parameter_window_seconds * sfreq))
    for window_index, left in enumerate(range(0, data.shape[1], step)):
        right = min(data.shape[1], left + step)
        f_window, p_window, window_estimator = _welch(data[:, left:right], sfreq, cfg)
        for index in range(len(ids)):
            parameterization["windows"].append({**_channel(index, ids, names),
                **_location(left, right, sfreq, start_sample, start_seconds), "window_index": window_index,
                "estimator": window_estimator,
                **_fit_one(f_window, p_window[index], cfg, (right - left) / sfreq, window_estimator, fooof_class)})
    line = _line_analysis(data, sfreq, ids, names, cfg, start_sample, start_seconds)
    measured = [row for row in line["observations"] if row["status"] == "computed"]
    line["strongest_window_psd"] = None
    if measured:
        row = max(measured, key=lambda r: np.inf if r["f_is_infinite"] else r["f_statistic"])
        left, right, ci = row["start_offset"], row["stop_offset"], row["channel_index"]
        local_f, local_p, local_estimator = _welch(data[ci:ci+1, left:right], sfreq, cfg)
        line["strongest_window_psd"] = {**_channel(ci, ids, names), **_location(left, right, sfreq, start_sample, start_seconds),
            "nominal_hz": row["nominal_hz"], "selection_rule": "maximum F across computed observations",
            "frequencies_hz": local_f.tolist(), "power_v2_hz": local_p[0].tolist(), "estimator": local_estimator}
    result = {"schema_version": SPECTRAL_VERSION,
        "input": {**origin, "sfreq": sfreq, "n_samples": data.shape[1], "channel_ids": ids,
                  "channel_names": names, "unit": "V",
                  "data_sha256": hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes()).hexdigest()},
        "config": asdict(cfg), "software": {key: version(key) for key in ("numpy", "scipy", "fooof")},
        "methods": {
            "line_noise": {"name": "Thomson multitaper F test", "doi": ["10.1016/S0006-3495(99)77236-X", "10.3389/fninf.2015.00016"],
                "reference": "https://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/continuous/ftestc.html",
                "formula": "F=(K-1)*abs(A)^2*sum(H^2)/sum(abs(J-A*H)^2); A=sum(J*H)/sum(H^2)",
                "null_model": "zero deterministic sinusoid amplitude in locally white residual noise",
                "parameter_source": "PREP: 4s windows, 1s step, 2Hz full DPSS bandwidth, K=7, scan +/-2Hz, alpha .01",
                "project_choices": "linear detrending per window; unpadded FFT grid; Bonferroni across the complete input test family",
                "output_action": "review candidates; original samples retained",
                "difference_from_cleanline": "one measurement pass; no iterative subtraction, overlap reconstruction, or robust rereferencing"},
            "psd": {"name": "Welch density estimate", "doi": ["10.1109/TAU.1967.1161901"],
                    "parameter_source": "project configuration: 4s Hann subwindows, 50% overlap, arithmetic averaging"},
            "alpha": {"name": "FOOOF periodic/aperiodic parameterization", "doi": ["10.1038/s41593-020-00744-x"],
                      "reference": "https://github.com/fooof-tools/fooof/tree/v1.1.1",
                      "parameter_source": "FOOOF fixed model and 2-SD peak rule; project fit band 2-40Hz, peak widths 1-8Hz, 6 peaks, 30s windows",
                      "interpretation": "fitted alpha-band peaks with background and fit residual; review with recording state"}},
        "line_noise": line,
        "psd": {"frequencies_hz": frequencies.tolist(), "power_v2_hz": power.tolist(),
                "channel_ids": ids, **origin, "estimator": estimator}, "parameterization": parameterization}
    from .alpha_power import compute_alpha_power

    result["alpha_power"] = compute_alpha_power(data, result)
    return result


def compare_spectral_reference(data_volts, result):
    """Audit an exported result against MNE and direct FOOOF calls.

    MNE independently evaluates the Thomson statistic and sinusoid coefficients.
    It receives our explicit taper and testing settings; its package defaults
    differ. FOOOF is the author implementation used by the production wrapper,
    so its comparison checks data/parameter/result mapping, not independence of
    the spectral model. The source input fingerprint is checked before either.
    """
    import inspect

    from mne.filter import _mt_spectrum_remove
    from mne.time_frequency import psd_array_welch

    data = np.asarray(data_volts, dtype=float)
    expected = result["input"]
    digest = hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes()).hexdigest()
    if (data.ndim != 2 or data.shape != (len(expected["channel_ids"]), expected["n_samples"])
            or digest != expected["data_sha256"]):
        raise ValueError("reference comparison requires the exact spectral-analysis input")
    cfg = SpectralQCConfig(**result["config"])
    sfreq = expected["sfreq"]
    line = result["line_noise"]
    line_audit = {"reference": "mne.filter._mt_spectrum_remove", "reference_version": version("mne"),
                  "reference_source_sha256": hashlib.sha256(inspect.getsource(_mt_spectrum_remove).encode()).hexdigest(),
                  "channel_windows_compared": 0, "unassessed_channel_windows": 0,
                  "analysis_status": line["status"], "observed_peak_amplitude_checks": 0,
                  "flag_mismatch_count": 0, "max_fitted_signal_abs_error_v": 0.0,
                  "f_boundary_checks": 0, "f_boundary_failures": 0,
                  "f_boundary_relative_tolerance": 1e-8,
                  "same_explicit_tapers_and_threshold": True,
                  "comparison_scope": "all complete windows; peak F boundaries at first/middle/last windows for every channel",
                  "difference_from_reference_defaults": "PREP symmetric K=7 DPSS and entire-input Bonferroni family are supplied explicitly"}
    if line["status"] == "computed":
        length = line["settings"]["window_samples"]
        tapers, details = _line_tapers(length, sfreq, cfg.line_bandwidth_hz)
        df1, df2 = details["degrees_of_freedom"]
        threshold = f_distribution.isf(line["multiplicity"]["per_test_alpha"], df1, df2)
        grouped = {}
        for row in line["observations"]:
            grouped.setdefault((row["window_index"], row["channel_index"]), []).append(row)
        boundary_windows = {0, len(line["windows"]) // 2, len(line["windows"]) - 1}
        for window_index, window in enumerate(line["windows"]):
            segment = detrend(data[:, window["start_offset"]:window["stop_offset"]], axis=-1, type="linear")
            radians = 2 * np.pi * np.arange(length) / sfreq
            for channel in range(len(data)):
                rows = grouped[window_index, channel]
                x = segment[channel]
                assessed = [row for row in rows if row["status"] == "computed"]
                if not assessed:
                    line_audit["unassessed_channel_windows"] += 1
                    continue
                _, detected = _mt_spectrum_remove(x, sfreq, None, None, tapers, threshold,
                                                  lambda n: (tapers, threshold))
                for row in assessed:
                    expected_flag = bool(np.any(np.abs(detected - row["nominal_hz"]) <= cfg.line_scan_half_width_hz))
                    line_audit["flag_mismatch_count"] += int(expected_flag != row["flagged"])
                # Check coefficients even when the line test found no significant
                # candidates: a negative result must still have numerical evidence.
                if assessed:
                    unique_peaks = {row["peak_frequency_hz"]: row for row in assessed}
                    tested = np.asarray(list(unique_peaks))
                    residual, _ = _mt_spectrum_remove(x, sfreq, tested, np.zeros(len(tested)), tapers,
                                                      threshold, lambda n: (tapers, threshold))
                    reconstructed = np.sum([row["amplitude_uv"] * 1e-6 * np.cos(
                        frequency * radians + row["phase_radians"]) for frequency, row in unique_peaks.items()], axis=0)
                    error = float(np.max(np.abs((x - residual) - reconstructed)))
                    line_audit["max_fitted_signal_abs_error_v"] = max(line_audit["max_fitted_signal_abs_error_v"], error)
                    line_audit["observed_peak_amplitude_checks"] += len(unique_peaks)
                if window_index in boundary_windows:
                    for row in assessed:
                        if row["f_statistic"] is None or row["f_statistic"] <= 0:
                            continue
                        for multiplier, should_find in ((1 - 1e-8, True), (1 + 1e-8, False)):
                            boundary = row["f_statistic"] * multiplier
                            _, found = _mt_spectrum_remove(x, sfreq, None, None, tapers, boundary,
                                                           lambda n: (tapers, boundary))
                            found_target = bool(np.any(found == row["peak_frequency_hz"]))
                            line_audit["f_boundary_checks"] += 1
                            line_audit["f_boundary_failures"] += int(found_target != should_find)
                line_audit["channel_windows_compared"] += 1
    line_audit["passed"] = (line_audit["flag_mismatch_count"] == 0
                            and line_audit["f_boundary_failures"] == 0
                            and line_audit["max_fitted_signal_abs_error_v"] < 1e-15)

    estimator = result["psd"]["estimator"]
    reference_power, reference_freqs = psd_array_welch(data, sfreq, n_fft=estimator["nfft"],
        n_per_seg=estimator["nperseg"], n_overlap=estimator["noverlap"], average="mean", window="hann",
        remove_dc=True, verbose=False)
    psd_difference = float(np.max(np.abs(np.asarray(result["psd"]["power_v2_hz"]) - reference_power)))
    roundoff_floor = np.maximum(1e-30, 64 * np.finfo(float).eps * np.max(reference_power, axis=1, keepdims=True))
    psd_audit = {"reference": "mne.time_frequency.psd_array_welch", "reference_version": version("mne"),
                 "kernel_note": "MNE and this module both use SciPy spectral estimators",
                 "max_abs_error_v2_hz": psd_difference,
                 "rtol": 1e-12, "roundoff_atol_per_channel_v2_hz": roundoff_floor[:, 0].tolist(),
                 "atol_definition": "max(1e-30, 64*machine_epsilon*channel_max_PSD)",
                 "allclose": bool(np.array_equal(reference_freqs, result["psd"]["frequencies_hz"])
                     and np.allclose(reference_power, result["psd"]["power_v2_hz"], rtol=1e-12, atol=roundoff_floor))}
    local = line.get("strongest_window_psd")
    if local is not None:
        est = local["estimator"]
        reference, freq = psd_array_welch(data[local["channel_index"], local["start_offset"]:local["stop_offset"]], sfreq,
            n_fft=est["nfft"], n_per_seg=est["nperseg"], n_overlap=est["noverlap"], average="mean", window="hann", remove_dc=True, verbose=False)
        close = np.array_equal(freq, local["frequencies_hz"]) and np.allclose(reference, local["power_v2_hz"], rtol=1e-12, atol=1e-30)
        psd_audit["local_window_max_abs_error_v2_hz"] = float(np.max(np.abs(reference-local["power_v2_hz"])))
        psd_audit["local_window_allclose"] = bool(close)
        psd_audit["allclose"] = bool(psd_audit["allclose"] and close)
    fooof_class = _fooof_class()
    parameter_audit = {"author_package_version": FOOOF_VERSION,
                       "comparison_role": "direct author-package reuse; independent call verifies wrapper input/output mapping",
                       "compared_full_channel_fits": 0, "skipped_channel_statuses": [],
                       "max_model_error_log10_power": 0.0, "peak_count_mismatches": 0,
                       "fit_failure_status_checks": 0, "fit_failure_status_mismatches": 0}
    for row in result["parameterization"]["full"]:
        if row["status"] not in {"computed", "fit_failed"}:
            parameter_audit["skipped_channel_statuses"].append({"channel_id": row["channel_id"], "status": row["status"]})
            continue
        direct = fooof_class(peak_width_limits=cfg.peak_width_limits_hz, max_n_peaks=cfg.max_n_peaks,
            min_peak_height=cfg.min_peak_height_log10, peak_threshold=cfg.peak_threshold_sd,
            aperiodic_mode="fixed", verbose=False)
        direct.fit(np.asarray(result["psd"]["frequencies_hz"]),
                   np.asarray(result["psd"]["power_v2_hz"])[row["channel_index"]], cfg.fit_range_hz)
        if row["status"] == "fit_failed":
            parameter_audit["fit_failure_status_checks"] += 1
            parameter_audit["fit_failure_status_mismatches"] += int(direct.has_model)
            continue
        error = float(np.max(np.abs(np.asarray(row["model_log10_v2_hz"]) - direct.fooofed_spectrum_)))
        parameter_audit["max_model_error_log10_power"] = max(parameter_audit["max_model_error_log10_power"], error)
        parameter_audit["peak_count_mismatches"] += int(len(row["peaks"]) != direct.n_peaks_)
        parameter_audit["compared_full_channel_fits"] += 1
    parameter_audit["passed"] = (parameter_audit["max_model_error_log10_power"] < 1e-10
                                 and parameter_audit["peak_count_mismatches"] == 0
                                 and parameter_audit["fit_failure_status_mismatches"] == 0)
    return {"passed": bool(line_audit["passed"] and psd_audit["allclose"] and parameter_audit["passed"]),
            "input_sha256_matches": True, "line": line_audit, "psd": psd_audit,
            "parameterization": parameter_audit}
