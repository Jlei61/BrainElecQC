"""Channel-resolved high-frequency bursts and simultaneous EMG evidence.

The envelope calculation follows MNE's ``annotate_muscle_zscore`` for a
single EEG channel. Frequency selection, spectral safeguards and auxiliary
EMG corroboration are explicit BrainElecQC screening policies. Signal arrays
are volts; every exported interval uses the original half-open sample clock.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral, Real

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class EEGMuscleConfig:
    lower_frequency_hz: float = 30.0
    upper_frequency_hz: float = 100.0
    upper_sampling_fraction: float = 0.45
    z_threshold: float = 4.0
    reference_z_threshold: float = 2.0
    envelope_correlation_threshold: float = 0.5
    window_seconds: float = 1.0
    step_seconds: float = 0.25
    min_high_score_seconds: float = 0.15
    min_power_ratio: float = 3.0
    min_effective_frequency_bins: float = 5.0
    min_non_line_power_fraction: float = 0.2
    min_high_frequency_fraction: float = 0.01
    line_exclusion_half_width_hz: float = 3.0
    max_difference_energy_fraction: float = 0.25
    constant_min_seconds: float = 0.10
    constant_tolerance_volts: float = 1e-15
    exclusion_padding_seconds: float = 0.5
    envelope_smoothing_hz: float = 4.0
    correlation_context_seconds: float = 2.0
    min_duration_seconds: float = 10.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.lower_frequency_hz >= self.upper_frequency_hz:
            raise ValueError("lower_frequency_hz must be below upper_frequency_hz")
        if not 0 < self.upper_sampling_fraction < 0.5:
            raise ValueError("upper_sampling_fraction must be below Nyquist")
        for key in ("envelope_correlation_threshold", "min_non_line_power_fraction", "min_high_frequency_fraction", "max_difference_energy_fraction"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be at most one")
        if self.step_seconds > self.window_seconds or self.min_high_score_seconds > self.window_seconds:
            raise ValueError("step and high-score duration must fit the analysis window")
        if self.min_duration_seconds < 10:
            raise ValueError("min_duration_seconds must be at least 10 for stable envelope estimation")


def _fingerprint(data):
    return sha256(np.asarray(data, dtype="<f8").tobytes()).hexdigest()


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _range(first, stop, sfreq, start_sample, start_seconds):
    return {"start_sample": int(start_sample + first), "stop_sample": int(start_sample + stop),
            "start_seconds": float(start_seconds + first / sfreq),
            "stop_seconds": float(start_seconds + stop / sfreq)}


def _line_frequencies(sfreq, upper):
    return sorted({float(base * harmonic) for base in (50, 60)
                   for harmonic in range(1, int(upper // base) + 1)
                   if base * harmonic + 3 < sfreq / 2})


def _envelope(row, sfreq, band, line_frequencies, cfg):
    """MNE primitives, including its Hilbert FFT padding, keep reference exact."""
    import mne

    raw = mne.io.RawArray(row[np.newaxis].copy(), mne.create_info(["signal"], sfreq, "eeg"), verbose=False)
    if line_frequencies:
        raw.notch_filter(line_frequencies, notch_widths=4.0, verbose=False)
    raw.filter(*band, fir_design="firwin", pad="reflect_limited", verbose=False)
    raw.apply_hilbert(envelope=True, verbose=False)
    envelope = raw.get_data()[0]
    scale = float(envelope.std())
    if scale <= np.finfo(float).eps * max(float(np.max(np.abs(envelope))), 1e-30):
        return envelope, np.zeros(len(row))
    score = (envelope - envelope.mean()) / scale
    score = mne.filter.filter_data(score, sfreq, None, cfg.envelope_smoothing_hz, verbose=False)
    return envelope, score


def _guard_mask(row, sfreq, cfg, explicit):
    """Guard constant runs and supplied continuity intervals including ringing."""
    mask = np.zeros(len(row), bool)
    pad = int(np.ceil(cfg.exclusion_padding_seconds * sfreq))
    minimum = max(2, int(np.ceil(cfg.constant_min_seconds * sfreq)))
    for first, last in _runs(np.abs(np.diff(row)) <= cfg.constant_tolerance_volts):
        if last - first + 1 >= minimum:
            mask[max(0, first - pad):min(len(row), last + 1 + pad)] = True
    for first, stop in explicit:
        mask[max(0, first - pad):min(len(row), stop + pad)] = True
    mask[:pad] = True
    mask[-pad:] = True
    return mask


def _spectral_metrics(row, sfreq, band, line_frequencies, cfg):
    frequencies, power = welch(row, fs=sfreq, window="hann", nperseg=len(row),
                               noverlap=0, detrend="constant", scaling="density")
    active = (frequencies >= band[0]) & (frequencies <= band[1])
    outside_line = active.copy()
    for frequency in line_frequencies:
        outside_line &= np.abs(frequencies - frequency) > cfg.line_exclusion_half_width_hz
    kept = power[outside_line]
    total = float(np.sum(power[active]))
    clean = float(np.sum(kept))
    total_eeg = float(np.sum(power[(frequencies >= 2) & (frequencies <= band[1])]))
    effective = clean ** 2 / float(np.sum(kept ** 2)) if np.any(kept) else 0.0
    df = float(frequencies[1] - frequencies[0])
    difference_energy = np.diff(row) ** 2
    concentration = (float(np.max(difference_energy) / np.sum(difference_energy))
                     if np.any(difference_energy) else 0.0)
    return {"high_frequency_power_uv2": clean * df * 1e12,
            "non_line_power_fraction": clean / total if total else 0.0,
            "high_frequency_fraction": clean / total_eeg if total_eeg else 0.0,
            "effective_frequency_bins": effective,
            "frequency_resolution_hz": df,
            "max_difference_energy_fraction": concentration}


def compute_muscle_qc(data_volts, sfreq, channel_ids, channel_names=None, *, references=None,
                      reference_channels=None, start_sample=0, start_seconds=0.0,
                      config=None, excluded_intervals=None):
    """Measure high-frequency bursts with optional simultaneous EMG support.

    ``references['emg']`` is a native-rate, aligned one-dimensional array.
    ``excluded_intervals`` contains channel IDs (or indices) and original
    ``start_sample``/``stop_sample`` values, e.g. continuity candidates. The
    output retains MNE-equivalent per-channel scores before the guards so
    plotting and reference checks can inspect every gate independently.
    """
    cfg = config if isinstance(config, EEGMuscleConfig) else EEGMuscleConfig(**({} if config is None else config))
    raw = np.asarray(data_volts)
    if raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "fiu":
        raise ValueError("data_volts must be a nonempty numeric channels-by-samples array")
    data = np.asarray(raw, dtype=float)
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive and finite")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, Real) or not np.isfinite(start_seconds):
        raise ValueError("start_seconds must be finite")
    ids = [] if isinstance(channel_ids, (str, bytes)) else list(channel_ids)
    if len(ids) != len(data) or any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("channel_ids must contain distinct nonempty strings")
    names = ids.copy() if channel_names is None else ([] if isinstance(channel_names, (str, bytes)) else list(channel_names))
    if len(names) != len(ids) or any(not isinstance(v, str) for v in names):
        raise ValueError("channel_names must contain one string per channel")
    sfreq, start_sample, start_seconds = float(sfreq), int(start_sample), float(start_seconds)
    n = data.shape[1]
    if not np.isfinite(start_seconds + n / sfreq):
        raise ValueError("stop time must be finite")
    references = {} if references is None else references
    if not isinstance(references, dict) or set(references) - {"ecg", "eog", "emg"}:
        raise ValueError("references accepts ecg, eog and emg arrays")
    reference = None
    if "emg" in references:
        original = np.asarray(references["emg"])
        if original.ndim != 1 or len(original) != n or original.dtype.kind not in "fiu":
            raise ValueError("EMG reference must share the EEG sample count and clock")
        reference = np.asarray(original, float)
    if reference_channels is not None and not isinstance(reference_channels, dict):
        raise ValueError("reference_channels must map kinds to channel metadata")
    source = dict((reference_channels or {}).get("emg", {}))
    source.setdefault("raw_name", "EMG")
    source.setdefault("channel_id", None)
    band = [cfg.lower_frequency_hz, min(cfg.upper_frequency_hz, cfg.upper_sampling_fraction * sfreq)]
    lines = _line_frequencies(sfreq, band[1])
    result = {"schema_version": 1, "method_version": "eeg-muscle-review-1.0",
              "data_sha256": _fingerprint(data), "channel_ids": ids, "channel_names": names,
              "sample_rate_hz": sfreq, "unit": "V", "interval_convention": "half_open",
              "input_range": _range(0, n, sfreq, start_sample, start_seconds),
              "parameters": {**asdict(cfg), "effective_filter_band_hz": band,
                             "notch_frequencies_hz": lines, "notch_width_hz": 4.0,
                             "envelope_zscore": "mean and population standard deviation over this input per channel",
                             "hilbert_padding": "MNE Raw.apply_hilbert default",
                             "reference_correlation": "zero-lag Pearson correlation of 4 Hz smoothed envelope z-scores",
                             "policy_source": "BrainElecQC screening policy; MNE provides envelope z-score calculation",
                             "line_families": [50, 60]},
              "method_sources": ["https://doi.org/10.3389/fnhum.2013.00138",
                                 "https://mne.tools/stable/generated/mne.preprocessing.annotate_muscle_zscore.html"],
              "status": "ok", "channels": [], "windows": [], "candidates": [],
              "reference": {"status": "missing_reference" if reference is None else "ok",
                            "source_reference": source,
                            "valid_seconds": 0.0, "excluded_seconds": n / sfreq,
                            "data_sha256": None if reference is None else _fingerprint(reference)},
              "sample_metrics": {"envelope_uv": [], "z_score": [], "excluded": [],
                                 "reference_envelope_uv": None, "reference_z_score": None},
              "review_status": "pending"}
    if band[1] - band[0] < 20:
        result["status"] = "insufficient_sampling_rate"
        return result
    if n / sfreq < cfg.min_duration_seconds:
        result["status"] = "insufficient_duration"
        return result
    import mne

    result["engine_version"] = mne.__version__
    explicit = [[] for _ in ids]
    for interval in [] if excluded_intervals is None else excluded_intervals:
        cid = interval.get("channel_id")
        if cid is not None:
            if cid not in ids:
                raise ValueError("excluded interval refers to an unknown channel_id")
            channel = ids.index(cid)
            if "channel_index" in interval and interval["channel_index"] != channel:
                raise ValueError("excluded interval channel index disagrees with channel_id")
        else:
            channel = interval.get("channel_index")
            if isinstance(channel, bool) or not isinstance(channel, Integral) or not 0 <= channel < len(ids):
                raise ValueError("excluded interval requires a valid channel_id or channel_index")
        left, right = interval.get("start_sample"), interval.get("stop_sample")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (left, right)) or left >= right:
            raise ValueError("excluded interval requires increasing integer sample bounds")
        if right > start_sample and left < start_sample + n:
            explicit[channel].append((max(0, left - start_sample), min(n, right - start_sample)))
    ref_envelope = ref_score = None
    ref_guard = np.zeros(n, bool)
    if reference is not None:
        if not np.all(np.isfinite(reference)):
            result["reference"]["status"] = "nonfinite_reference"
        elif np.std(reference) <= cfg.constant_tolerance_volts:
            result["reference"]["status"] = "constant_reference"
        else:
            ref_envelope, ref_score = _envelope(reference, sfreq, band, lines, cfg)
            ref_guard = _guard_mask(reference, sfreq, cfg, [])
            result["reference"]["valid_seconds"] = float((~ref_guard).sum() / sfreq)
            result["reference"]["excluded_seconds"] = float(ref_guard.sum() / sfreq)
            result["reference"]["excluded_intervals"] = [_range(a, b, sfreq, start_sample, start_seconds) for a, b in _runs(ref_guard)]
            result["sample_metrics"]["reference_envelope_uv"] = (ref_envelope * 1e6).tolist()
            result["sample_metrics"]["reference_z_score"] = ref_score.tolist()
    width, step = max(1, round(cfg.window_seconds * sfreq)), max(1, round(cfg.step_seconds * sfreq))
    positions = list(range(0, n - width + 1, step))
    result["parameters"]["window_samples"] = width
    result["parameters"]["step_samples"] = step
    result["windows"] = [{"window_index": i, **_range(a, a + width, sfreq, start_sample, start_seconds)}
                         for i, a in enumerate(positions)]
    for channel, row in enumerate(data):
        summary = {"channel_index": channel, "channel_id": ids[channel], "channel_name": names[channel],
                   "status": "ok", "window_metrics": [], "excluded_seconds": n / sfreq, "valid_seconds": 0.0,
                   "candidate_seconds": 0.0, "emg_supported_seconds": 0.0}
        result["channels"].append(summary)
        if not np.all(np.isfinite(row)):
            summary["status"] = "nonfinite_input"
            result["sample_metrics"]["envelope_uv"].append(None)
            result["sample_metrics"]["z_score"].append(None)
            result["sample_metrics"]["excluded"].append(None)
            continue
        envelope, score = _envelope(row, sfreq, band, lines, cfg)
        guard = _guard_mask(row, sfreq, cfg, explicit[channel])
        summary["excluded_seconds"] = float(guard.sum() / sfreq)
        summary["valid_seconds"] = float((~guard).sum() / sfreq)
        summary["excluded_intervals"] = [_range(a, b, sfreq, start_sample, start_seconds) for a, b in _runs(guard)]
        result["sample_metrics"]["envelope_uv"].append((envelope * 1e6).tolist())
        result["sample_metrics"]["z_score"].append(score.tolist())
        result["sample_metrics"]["excluded"].append(guard.tolist())
        measurements = [_spectral_metrics(row[a:a + width], sfreq, band, lines, cfg) for a in positions]
        baseline_values = [v["high_frequency_power_uv2"] for a, v in zip(positions, measurements) if not guard[a:a + width].any()]
        baseline = float(np.median(baseline_values)) if baseline_values else 0.0
        summary["median_high_frequency_power_uv2"] = baseline
        selected = {"high_frequency_burst": np.zeros(n, bool), "emg_supported_burst": np.zeros(n, bool)}
        for index, (a, metrics) in enumerate(zip(positions, measurements)):
            b = a + width
            ratio = metrics["high_frequency_power_uv2"] / baseline if baseline > 0 else 0.0
            high_duration = float(np.count_nonzero(score[a:b] > cfg.z_threshold) / sfreq)
            context_a = max(0, a - round(cfg.correlation_context_seconds * sfreq / 2))
            context_b = min(n, b + round(cfg.correlation_context_seconds * sfreq / 2))
            correlation = None
            ref_peak = None
            support = False
            if ref_envelope is not None:
                valid = ~(guard[context_a:context_b] | ref_guard[context_a:context_b])
                x, y = score[context_a:context_b][valid], ref_score[context_a:context_b][valid]
                if len(x) >= sfreq and x.std() > 0 and y.std() > 0:
                    correlation = float(np.corrcoef(x, y)[0, 1])
                ref_peak = float(np.max(ref_score[a:b]))
                support = (correlation is not None and correlation >= cfg.envelope_correlation_threshold
                           and ref_peak >= cfg.reference_z_threshold and not ref_guard[a:b].any())
            gates = {"continuity_clear": not bool(guard[a:b].any()),
                     "high_envelope": high_duration >= cfg.min_high_score_seconds,
                     "power_increase": ratio >= cfg.min_power_ratio,
                     "broadband": metrics["effective_frequency_bins"] >= cfg.min_effective_frequency_bins,
                     "outside_line": metrics["non_line_power_fraction"] >= cfg.min_non_line_power_fraction,
                     "high_frequency_contribution": metrics["high_frequency_fraction"] >= cfg.min_high_frequency_fraction,
                     "distributed_changes": metrics["max_difference_energy_fraction"] <= cfg.max_difference_energy_fraction}
            flag = all(gates.values())
            kind = ("emg_supported_burst" if support else "high_frequency_burst") if flag else None
            if flag:
                selected[kind][a:b] = True
            summary["window_metrics"].append({"window_index": index, **result["windows"][index], **metrics,
                                               "power_ratio": ratio, "peak_z_score": float(np.max(score[a:b])),
                                               "high_score_seconds": high_duration,
                                               "mean_envelope_uv": float(envelope[a:b].mean() * 1e6),
                                               "reference_peak_z_score": ref_peak,
                                               "reference_envelope_correlation": correlation,
                                               "correlation_range": _range(context_a, context_b, sfreq, start_sample, start_seconds),
                                               "gates": gates, "flagged": flag, "kind": kind})
        # A sample supported by EMG receives the stronger evidence label once.
        selected["high_frequency_burst"] &= ~selected["emg_supported_burst"]
        summary["candidate_seconds"] = float(np.count_nonzero(selected["high_frequency_burst"] | selected["emg_supported_burst"]) / sfreq)
        summary["emg_supported_seconds"] = float(selected["emg_supported_burst"].sum() / sfreq)
        for kind, mask in selected.items():
            for a, b in _runs(mask):
                rows = [v for v in summary["window_metrics"] if v["kind"] == kind and
                        v["start_sample"] < start_sample + b and v["stop_sample"] > start_sample + a]
                result["candidates"].append({"candidate_id": f"muscle:{channel}:{start_sample+a}:{kind}",
                    "channel_index": channel, "channel_id": ids[channel], "channel_name": names[channel],
                    "kind": kind, "action": "review", "review_status": "pending",
                    "evidence_level": "EMG同步支持" if kind == "emg_supported_burst" else "宽带高频突发",
                    "score_category": "muscle" if kind == "emg_supported_burst" else "channel_quality",
                    "window_indices": [r["window_index"] for r in rows],
                    **_range(a, b, sfreq, start_sample, start_seconds),
                    "peak_z_score": max(r["peak_z_score"] for r in rows),
                    "max_power_ratio": max(r["power_ratio"] for r in rows),
                    "max_reference_envelope_correlation": max((r["reference_envelope_correlation"] for r in rows
                        if r["reference_envelope_correlation"] is not None), default=None)})
    if any(c["status"] != "ok" for c in result["channels"]):
        result["status"] = "partial_channels"
    return result


def compare_muscle_to_mne(data_volts, result):
    """Independently call public MNE once per channel after identical notches."""
    import mne

    data = np.asarray(data_volts, float)
    if _fingerprint(data) != result["data_sha256"]:
        raise ValueError("data fingerprint differs from muscle result")
    if len(result["sample_metrics"]["z_score"]) != len(data):
        raise ValueError("muscle result has no available per-channel scores")
    sfreq = result["sample_rate_hz"]
    p = result["parameters"]
    if p["envelope_smoothing_hz"] != 4:
        raise ValueError("MNE comparison requires its fixed 4 Hz score smoothing")
    rows = []
    for index, values in enumerate(data):
        saved = result["sample_metrics"]["z_score"][index]
        if saved is None or not np.std(values):
            rows.append({"channel_id": result["channel_ids"][index], "status": "unavailable"})
            continue
        raw = mne.io.RawArray(values[np.newaxis].copy(), mne.create_info(["signal"], sfreq, "eeg"), verbose=False)
        if p["notch_frequencies_hz"]:
            raw.notch_filter(p["notch_frequencies_hz"], notch_widths=p["notch_width_hz"], verbose=False)
        _, score = mne.preprocessing.annotate_muscle_zscore(raw, threshold=p["z_threshold"], ch_type="eeg",
            min_length_good=0, filter_freq=p["effective_filter_band_hz"], verbose=False)
        rows.append({"channel_id": result["channel_ids"][index], "status": "compared",
                     "score_max_abs_error": float(np.max(np.abs(score - np.asarray(saved))))})
    return {"reference": "mne.preprocessing.annotate_muscle_zscore; per-channel, identical notch input",
            "engine_version": mne.__version__, "channels": rows,
            "scope": "Envelope z-score numerical agreement; broadband/continuity/EMG gates are project screening policies"}
