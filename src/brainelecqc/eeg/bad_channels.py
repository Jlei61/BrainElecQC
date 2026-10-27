"""Traceable PREP non-spatial channel measurements and review candidates.

The input montage/reference is retained. Statistics are calculated independently
of PyPREP; ``compare_with_pyprep`` runs the separately installed reference package.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.stats import median_abs_deviation


@dataclass(frozen=True)
class BadChannelConfig:
    """PREP thresholds plus explicitly identified project applicability gates."""

    highpass_hz: float = 1.0
    window_seconds: float = 1.0
    deviation_z: float = 5.0
    hf_noise_z: float = 5.0
    correlation_threshold: float = 0.4
    bad_window_fraction: float = 0.01
    flat_threshold_volts: float = 1e-15
    min_group_channels: int = 4

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.correlation_threshold >= 1 or self.bad_window_fraction >= 1:
            raise ValueError("correlation and fraction thresholds must be below 1")
        if int(self.min_group_channels) != self.min_group_channels or self.min_group_channels < 4:
            raise ValueError("min_group_channels must be an integer >= 4")


def _config(config):
    if config is not None and not isinstance(config, (BadChannelConfig, dict)):
        raise ValueError("config must be BadChannelConfig or a parameter dictionary")
    return config if isinstance(config, BadChannelConfig) else BadChannelConfig(**(config or {}))


def _iqr(values, axis=None):
    # Hazen plotting positions are the MATLAB quantile convention used by PREP.
    q = np.quantile(values, [0.25, 0.75], axis=axis, method="hazen")
    return q[1] - q[0]


def _lowpass_kernel(sfreq):
    """Frequency-sampling definition of PyPREP 0.5.0's 100-order FIR."""
    grid = 512
    knots = np.round(grid * np.array([0, 90 / sfreq, 100 / sfreq, 1]))
    response = PchipInterpolator(knots, [1, 1, 0, 0])(np.arange(grid + 1))
    response = response * np.exp(-1j * np.pi * 50 * np.arange(grid + 1) / grid)
    return np.fft.irfft(response, n=2 * grid)[:101] * np.hamming(101)


def _prepare(data, sfreq, cfg):
    from mne.filter import filter_data

    high = filter_data(
        data.copy(), sfreq, l_freq=cfg.highpass_hz, h_freq=None, verbose=False,
    )
    low = signal.filtfilt(_lowpass_kernel(sfreq), [1.0], high, axis=1) if sfreq > 100 else high.copy()
    return high, low


def _score(values, *, mad=False):
    center = float(np.median(values))
    spread = float(median_abs_deviation(values) * 1.4826 if mad else _iqr(values) * 0.7413)
    # A zero cohort spread is a property of the comparison group. It cannot
    # support an outlier z-score; retain raw measurements for human review.
    if not np.isfinite(spread) or spread <= np.finfo(float).eps * max(abs(center), np.finfo(float).tiny):
        return np.full(np.shape(values), np.nan), center, spread
    return (values - center) / spread, center, spread


def _json_array(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return float(values) if np.isfinite(values) else None
    return [_json_array(item) for item in values]


def _range(first, stop, sfreq, start_sample, start_seconds):
    return {
        "start_sample": int(start_sample + first), "stop_sample": int(start_sample + stop),
        "start_seconds": float(start_seconds + first / sfreq),
        "stop_seconds": float(start_seconds + stop / sfreq),
    }


def _runs(mask):
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def _inputs(data_volts, sfreq, channel_ids, channel_names, groups, cfg, start_sample, start_seconds):
    data = np.asarray(data_volts)
    if data.dtype.kind not in "iuf":
        raise ValueError("data_volts must contain real numeric values")
    data = np.asarray(data, dtype=float)
    if isinstance(channel_ids, (str, bytes)) or isinstance(channel_names, (str, bytes)):
        raise ValueError("channel IDs and names must be sequences of strings")
    ids = list(channel_ids)
    names = list(channel_names) if channel_names is not None else ids.copy()
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("data_volts must have nonempty channel x sample shape")
    if len(ids) != data.shape[0] or len(names) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("channel IDs must be unique and match the data rows")
    if any(not isinstance(item, str) or not item for item in ids + names):
        raise ValueError("channel IDs and names must be nonempty strings")
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 2 * cfg.highpass_hz:
        raise ValueError("sfreq must exceed twice the highpass frequency")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, Real) or not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if isinstance(groups, Mapping):
        if set(groups) != set(ids):
            raise ValueError("groups mapping must contain exactly the channel IDs")
        labels = [groups[item] for item in ids]
    else:
        labels = list(groups) if groups is not None else ["unconfirmed"] * len(ids)
    if len(labels) != len(ids) or any(not isinstance(item, str) or not item for item in labels):
        raise ValueError("groups must provide one nonempty reference/montage group per channel")
    return data, ids, names, labels, groups is not None


def compute_bad_channels(
    data_volts, sfreq, channel_ids, channel_names=None, *, start_sample=0,
    start_seconds=0.0, groups=None, config=None,
):
    """Measure PREP channel criteria, preserving channel IDs and sample ranges.

    ``groups`` is a sequence of explicit common-reference/comparable EEG group
    labels, or an ID -> label mapping. Supplying groups declares a comparison
    assumption; acquisition-reference evidence stays in source metadata.
    Fewer than four usable channels retain descriptive values
    and defer cohort-threshold candidates. Four is a project eligibility gate,
    rather than a validated minimum montage size from the PREP paper.
    """
    cfg = _config(config)
    data, ids, names, labels, declared = _inputs(
        data_volts, sfreq, channel_ids, channel_names, groups, cfg, start_sample, start_seconds,
    )
    n_chan, n_samples = data.shape
    win_size = int(cfg.window_seconds * sfreq)
    if win_size < 2:
        raise ValueError("window_seconds must contain at least two samples")
    n_windows = n_samples // win_size
    bounds = [_range(w * win_size, (w + 1) * win_size, sfreq, start_sample, start_seconds) for w in range(n_windows)]
    duration_ok = n_samples >= max(304, int(np.ceil(3.3 * sfreq / cfg.highpass_hz)) + 1)
    arrays = {key: np.full((n_chan, n_windows), np.nan) for key in (
        "amplitude_uv", "amplitude_z", "hf_ratio", "hf_z", "max_abs_correlation", "dropout",
    )}
    amp, amp_z, hf_ratio, hf_z = (np.full(n_chan, np.nan) for _ in range(4))
    finite = np.all(np.isfinite(data), axis=1)
    raw_std, raw_mad = np.full(n_chan, np.nan), np.full(n_chan, np.nan)
    raw_std[finite] = np.std(data[finite], axis=1)
    raw_mad[finite] = median_abs_deviation(data[finite], axis=1)
    flat = np.zeros(n_chan, dtype=bool)
    flat[finite] = (raw_std[finite] < cfg.flat_threshold_volts) | (raw_mad[finite] < cfg.flat_threshold_volts)
    usable = finite & ~flat
    high = np.full_like(data, np.nan)
    low = np.full_like(data, np.nan)
    if duration_ok and np.any(usable):
        high[usable], low[usable] = _prepare(data[usable], sfreq, cfg)
        amp[usable] = _iqr(high[usable], axis=1) * 0.7413
        if sfreq > 100:
            denominator = median_abs_deviation(low[usable], axis=1)
            numerator = median_abs_deviation(high[usable] - low[usable], axis=1)
            hf_ratio[usable] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    group_info = []
    eligible = np.zeros(n_chan, dtype=bool)
    for label in dict.fromkeys(labels):
        indices = np.flatnonzero(np.array(labels) == label)
        active = indices[usable[indices]]
        reason = "ready"
        if not duration_ok:
            reason = "duration_shorter_than_analysis_filter"
        elif not declared:
            reason = "reference_group_unconfirmed"
        elif len(active) < cfg.min_group_channels:
            reason = "small_comparison_group"
        eligible[indices] = reason == "ready"
        info = {"group_id": label, "channel_ids": [ids[i] for i in indices], "usable_channel_count": len(active),
                "thresholds_active": reason == "ready", "status": reason,
                "reference_group_declared": declared, "grouping_basis": "caller_declared" if declared else "unreported"}
        group_info.append(info)
        if not duration_ok or not len(active):
            continue
        amp_z[active], amp_center, amp_sd = _score(amp[active])
        info.update(amplitude_center_uv=amp_center * 1e6, amplitude_spread_uv=amp_sd * 1e6)
        noise_center = noise_sd = np.nan
        if sfreq > 100 and np.all(np.isfinite(hf_ratio[active])):
            hf_z[active], noise_center, noise_sd = _score(hf_ratio[active], mad=True)
            info.update(hf_ratio_center=noise_center, hf_ratio_spread=noise_sd)
        for w in range(n_windows):
            cut = slice(w * win_size, (w + 1) * win_size)
            raw_window, low_window = high[active, cut], low[active, cut]
            amplitude = _iqr(raw_window, axis=1) * 0.7413
            arrays["amplitude_uv"][active, w] = amplitude * 1e6
            if amp_sd > 0:
                arrays["amplitude_z"][active, w] = (amplitude - amp_center) / amp_sd
            low_mad = median_abs_deviation(low_window, axis=1)
            valid_window = low_mad > 0
            arrays["dropout"][active, w] = ~valid_window
            valid = active[valid_window]
            arrays["max_abs_correlation"][active[~valid_window], w] = 0
            if sfreq > 100:
                noise = median_abs_deviation(raw_window[valid_window] - low_window[valid_window], axis=1) / low_mad[valid_window]
                arrays["hf_ratio"][valid, w] = noise
                if noise_sd > 0:
                    arrays["hf_z"][valid, w] = (noise - noise_center) / noise_sd
            if len(valid) >= 2:
                corr = np.abs(np.corrcoef(low_window[valid_window]))
                np.fill_diagonal(corr, 0)
                arrays["max_abs_correlation"][valid, w] = np.quantile(corr, 0.98, axis=0, method="hazen")
    channels, candidates = [], []

    def add_candidate(i, kind, first=0, stop=n_samples, window_indices=None):
        reasons = {"nonfinite_input": "原始输入含非有限数值", "flat_signal": "全段 MAD 或标准差低于 PREP 平线阈值",
                   "amplitude_deviation": "相同采集参考组内稳健幅度离群", "high_frequency_noise": "相同采集参考组内高频残差比例离群",
                   "low_correlation": "低频信号与同组通道相关偏低", "intermittent_dropout": "低频分析信号出现平线窗"}
        candidates.append({"channel_id": ids[i], "channel_name": names[i], "channel_index": i,
                           "group_id": labels[i], "kind": kind, "review_status": "pending",
                           "reason": reasons[kind], "evidence": {"method_id": "prep_nonspatial_review",
                               "amplitude_z": _json_array(amp_z[i]), "hf_z": _json_array(hf_z[i]),
                               "window_metrics_key": "max_abs_correlation" if kind == "low_correlation" else kind,
                               "parameters": asdict(cfg)},
                           "action": "review", "window_indices": window_indices or [],
                           **_range(first, stop, sfreq, start_sample, start_seconds)})

    for i, channel_id in enumerate(ids):
        corr = arrays["max_abs_correlation"][i]
        observed_corr = np.isfinite(corr)
        corr_fraction = float(np.mean(corr[observed_corr] < cfg.correlation_threshold)) if observed_corr.any() else None
        drop = arrays["dropout"][i]
        drop_fraction = float(np.mean(drop[np.isfinite(drop)])) if np.isfinite(drop).any() else None
        flags = []
        if not finite[i]:
            flags.append("nonfinite_input")
            for first, stop in _runs(~np.isfinite(data[i])):
                add_candidate(i, "nonfinite_input", first, stop)
        elif flat[i]:
            flags.append("flat_signal")
            add_candidate(i, "flat_signal")
        elif eligible[i]:
            if np.isfinite(amp_z[i]) and abs(amp_z[i]) > cfg.deviation_z:
                flags.append("amplitude_deviation")
                add_candidate(i, "amplitude_deviation")
            if np.isfinite(hf_z[i]) and hf_z[i] > cfg.hf_noise_z:
                flags.append("high_frequency_noise")
                add_candidate(i, "high_frequency_noise")
            if corr_fraction is not None and corr_fraction > cfg.bad_window_fraction:
                flags.append("low_correlation")
                for first, stop in _runs(observed_corr & (corr < cfg.correlation_threshold)):
                    add_candidate(i, "low_correlation", first * win_size, stop * win_size, list(range(first, stop)))
            if drop_fraction is not None and drop_fraction > cfg.bad_window_fraction:
                flags.append("intermittent_dropout")
                for first, stop in _runs(drop == 1):
                    add_candidate(i, "intermittent_dropout", first * win_size, stop * win_size, list(range(first, stop)))
        channels.append({"channel_id": channel_id, "channel_name": names[i], "channel_index": i,
                         "group_id": labels[i], "nonfinite_count": int(np.sum(~np.isfinite(data[i]))),
                         "flat": bool(flat[i]), "raw_std_uv": _json_array(raw_std[i] * 1e6),
                         "raw_mad_uv": _json_array(raw_mad[i] * 1e6), "robust_amplitude_uv": _json_array(amp[i] * 1e6),
                         "amplitude_z": _json_array(amp_z[i]), "hf_ratio": _json_array(hf_ratio[i]),
                         "hf_z": _json_array(hf_z[i]), "low_correlation_fraction": corr_fraction,
                         "dropout_fraction": drop_fraction, "flags": flags})
    return {
        "method": {"id": "prep_nonspatial_review", "version": "1.0.0",
                   "paper_doi": "10.3389/fninf.2015.00016", "reference_implementation": "pyprep 0.5.0",
                   "scope": "single-pass criteria in explicit input reference groups",
                   "reference_source": "https://github.com/sappelhoff/pyprep/blob/0.5.0/pyprep/find_noisy_channels.py",
                   "formulas": {"amplitude": "0.7413 * IQR_Hazen(highpass_1Hz_signal)",
                                "amplitude_z": "(amplitude - group median) / (0.7413 * group IQR_Hazen)",
                                "hf_ratio": "MAD(highpass_signal - lowpass_signal) / MAD(lowpass_signal)",
                                "hf_z": "(hf_ratio - group median) / (1.4826 * group MAD)",
                                "correlation": "98th Hazen percentile of absolute lowpass Pearson correlation; diagonal zero",
                                "window_fraction": "count(correlation < 0.4) / evaluated full windows"},
                   "threshold_source": "PREP / PyPREP defaults; min_group_channels is a project eligibility gate",
                   "differences": ["all complete windows retained; PyPREP 0.5.0 omits its final complete window",
                                   "zero cohort spread yields an unavailable z-score and retains raw measurements",
                                   "nonfinite channels are isolated before filtering", "HF normalization follows PyPREP's 1.4826 MAD"]},
        "parameters": asdict(cfg), "channel_ids": ids, "channel_names": names,
        "data_sha256": sha256(np.asarray(data, dtype="<f8").tobytes()).hexdigest(),
        "sample_rate": float(sfreq), "input_range": _range(0, n_samples, sfreq, start_sample, start_seconds),
        "preprocessing": {"highpass_hz": cfg.highpass_hz, "highpass_implementation": "MNE FIR default",
                          "lowpass": "100-order frequency-sampling FIR, 45–50 Hz transition, forward-backward" if sfreq > 100 else "input below 100 Hz sample rate",
                          "hf_status": "available" if sfreq > 100 else "sample_rate_at_or_below_100hz",
                          "analysis_only": True, "input_reference_retained": True},
        "groups": group_info, "channels": channels, "windows": bounds,
        "window_metrics": {key: _json_array(value) for key, value in arrays.items()},
        "unwindowed_tail": _range(n_windows * win_size, n_samples, sfreq, start_sample, start_seconds),
        "candidates": candidates, "automatic_exclusions": [],
    }


def compare_with_pyprep(data_volts, sfreq, channel_ids, channel_names=None, *, groups=None, config=None):
    """Compare independent measurements against installed PyPREP on identical data.

    The reference calls its own high-pass, low-pass and statistics. MNE's FIR
    high-pass primitive is shared; the downstream statistics are independent.
    Full-window outputs are compared over the reference implementation's scope.
    """
    import mne
    from pyprep import NoisyChannels

    cfg = _config(config)
    ours = compute_bad_channels(data_volts, sfreq, channel_ids, channel_names, groups=groups, config=cfg)
    data = np.asarray(data_volts, dtype=float)
    comparisons = []
    for group in ours["groups"]:
        indices = [ours["channel_ids"].index(item) for item in group["channel_ids"]]
        indices = [i for i in indices if not ours["channels"][i]["flat"] and not ours["channels"][i]["nonfinite_count"]]
        if len(indices) < 2 or group["status"] == "duration_shorter_than_analysis_filter":
            comparisons.append({"group_id": group["group_id"], "status": "insufficient_usable_data"})
            continue
        if cfg.highpass_hz != 1:
            raise ValueError("PyPREP default high-pass comparison requires highpass_hz=1")
        raw = mne.io.RawArray(data[indices].copy(), mne.create_info([ours["channel_ids"][i] for i in indices], sfreq, "eeg"), verbose=False)
        with mne.use_log_level("ERROR"):
            reference = NoisyChannels(raw, do_detrend=True, matlab_strict=False, ransac=False)
            reference.find_bad_by_deviation(cfg.deviation_z)
            reference.find_bad_by_hfnoise(cfg.hf_noise_z)
            reference.find_bad_by_correlation(cfg.window_seconds, cfg.correlation_threshold, cfg.bad_window_fraction)
        extra = reference._extra_info
        expected_corr = extra["bad_by_correlation"]["max_correlations"]
        common = expected_corr.shape[1]
        pairs = {
            "amplitude_z": ([ours["channels"][i]["amplitude_z"] for i in indices], extra["bad_by_deviation"]["robust_channel_deviations"]),
            "correlation": (np.asarray(ours["window_metrics"]["max_abs_correlation"], dtype=float)[indices, :common], expected_corr),
            "window_amplitude_uv": (np.asarray(ours["window_metrics"]["amplitude_uv"], dtype=float)[indices, :common], extra["bad_by_deviation"]["channel_amplitudes"].T * 1e6),
            "window_dropout": (np.asarray(ours["window_metrics"]["dropout"], dtype=float)[indices, :common], extra["bad_by_dropout"]["dropouts"]),
        }
        if sfreq > 100:
            pairs["hf_z"] = ([ours["channels"][i]["hf_z"] for i in indices], extra["bad_by_hf_noise"]["hf_noise_zscores"])
            pairs["window_hf_ratio"] = (np.asarray(ours["window_metrics"]["hf_ratio"], dtype=float)[indices, :common], extra["bad_by_hf_noise"]["noise_levels"].T)
        checks = {}
        for name, (actual, expected) in pairs.items():
            actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            finite = np.isfinite(actual) & np.isfinite(expected)
            checks[name] = {"max_abs_difference": float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else None,
                            "matches": bool(np.allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True))}
        reference_bads = reference.get_bads(as_dict=True)
        candidate_comparison = {"status": "descriptive_group", "full_window_differences": {}}
        if group["thresholds_active"]:
            category_map = {"amplitude_deviation": "bad_by_deviation", "high_frequency_noise": "bad_by_hf_noise",
                            "low_correlation": "bad_by_correlation", "intermittent_dropout": "bad_by_dropout"}
            candidate_comparison["status"] = "compared"
            candidate_comparison["common_window_matches"] = {}
            for flag, reference_key in category_map.items():
                output_ids = {ours["channel_ids"][i] for i in indices if flag in ours["channels"][i]["flags"]}
                reference_ids = set(reference_bads[reference_key])
                common_ids = output_ids
                if flag in {"low_correlation", "intermittent_dropout"}:
                    key = "max_abs_correlation" if flag == "low_correlation" else "dropout"
                    values = np.asarray(ours["window_metrics"][key], dtype=float)[indices, :common]
                    windows_flagged = values < cfg.correlation_threshold if flag == "low_correlation" else values == 1
                    common_ids = {ours["channel_ids"][i] for j, i in enumerate(indices) if np.mean(windows_flagged[j]) > cfg.bad_window_fraction}
                candidate_comparison["common_window_matches"][flag] = common_ids == reference_ids
                candidate_comparison["full_window_differences"][flag] = {
                    "only_in_output": sorted(output_ids - reference_ids), "only_in_reference": sorted(reference_ids - output_ids),
                }
        comparisons.append({"group_id": group["group_id"], "status": "compared", "channel_ids": [ours["channel_ids"][i] for i in indices],
                            "reference_windows": common, "output_windows": len(ours["windows"]),
                            "additional_output_windows": ours["windows"][common:],
                            "comparison_scope": "common complete windows", "checks": checks,
                            "candidate_comparison": candidate_comparison,
                            "reference_candidate_channels": reference_bads})
    compared_checks = [check for group in comparisons for check in group.get("checks", {}).values()]
    return {"reference": "pyprep", "reference_version": version("pyprep"), "mne_version": version("mne"),
            "data_sha256": ours["data_sha256"], "groups": comparisons,
            "compared_group_count": sum(group["status"] == "compared" for group in comparisons),
            "shared_primitive": "MNE default 1 Hz high-pass FIR; independently computed downstream criteria",
            "all_compared_measurements_match": bool(compared_checks) and all(check["matches"] for check in compared_checks)}
