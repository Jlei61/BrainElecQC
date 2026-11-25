"""Window-local evidence in physical volts, without filtering or interpolation.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import math

import numpy as np
from scipy.signal import welch


def _longest(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0


def _number(value):
    return float(value) if np.isfinite(value) else None


def _rails(channel):
    """File encoding bounds in volts; header calibration does not confirm ADC rails."""
    fields = ("physical_min", "physical_max", "scale_to_si", "digital_min", "digital_max")
    if any(channel.get(field) is None for field in fields):
        return None
    if any(isinstance(channel[field], bool) or not isinstance(channel[field], (int, float))
           or not math.isfinite(channel[field]) for field in fields):
        raise ValueError("File calibration fields must be finite numbers")
    low, high = sorted([channel["physical_min"] * channel["scale_to_si"],
                        channel["physical_max"] * channel["scale_to_si"]])
    codes = channel["digital_max"] - channel["digital_min"]
    if low == high or codes <= 0 or channel["scale_to_si"] <= 0:
        raise ValueError("Invalid file calibration")
    return low, high, (high - low) / codes


def compute_window_metrics(values, frequency, channel, config):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    valid = values[finite]
    voltage = channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage"
    metrics = {"unit": channel.get("unit"), "sample_count": len(values), "finite_sample_count": int(finite.sum()),
               "finite_fraction": float(finite.mean()), "missing_fraction": float(1 - finite.mean()),
               "flat_fraction": None, "longest_flat_seconds": None, "clipping_fraction": None,
               "longest_clipping_seconds": None, "rms_v": None, "mad_v": None, "robust_range_v": None,
               "offset_v": None, "drift_v_per_second": None, "max_jump_v": None,
               "kurtosis": None, "line_length_v_per_second": None, "line_noise_db": None,
               "line_noise_fraction": None, "line_peak_width_hz": None, "high_frequency_ratio": None,
               "line_harmonics": [],
               "voltage_status": "computed" if voltage and len(valid) else "insufficient_data",
               "clipping_status": "insufficient_data", "spectral_status": "insufficient_data",
               "boundary_dwell_status": "insufficient_data", "boundary_dwell_fraction": None,
               "longest_boundary_dwell_seconds": None, "boundary_dwell_score_effect": "none",
               "adc_range_status": "unconfirmed",
               "background_fit_status": "not_implemented", "alpha_peak_status": "not_scored"}
    flags = ["nonfinite"] if not finite.all() else []
    spectrum = {"status": "insufficient_data", "reason": "unknown_voltage_unit" if not voltage else "too_short_or_nonfinite_or_constant",
                "frequency_hz": [], "power_v2_hz": []}
    if not voltage:
        return metrics, flags, spectrum
    if len(valid):
        offset = np.median(valid)
        metrics.update(rms_v=_number(np.sqrt(np.mean(valid ** 2))), mad_v=_number(np.median(np.abs(valid - offset))),
                       robust_range_v=_number(np.percentile(valid, 99) - np.percentile(valid, 1)), offset_v=_number(offset))
    if len(values) < 2:
        return metrics, flags, spectrum
    differences = np.diff(values)
    valid_pairs = finite[:-1] & finite[1:]
    flat = valid_pairs & (np.abs(differences) <= config["flat_tolerance_v"])
    longest_flat = (_longest(flat) + 1) / frequency if flat.any() else 0.0
    metrics.update(flat_fraction=float(flat.sum() / max(1, valid_pairs.sum())), longest_flat_seconds=longest_flat)
    if longest_flat >= config["flat_min_seconds"] or flat.all():
        flags.append("flat")
    if valid_pairs.any():
        metrics.update(max_jump_v=_number(np.max(np.abs(differences[valid_pairs]))),
                       line_length_v_per_second=_number(np.abs(differences[valid_pairs]).sum() / (len(values) / frequency)))
    if len(valid) >= 3:
        centered = valid - valid.mean()
        variance = np.mean(centered ** 2)
        metrics["kurtosis"] = _number(np.mean(centered ** 4) / variance ** 2) if variance > 0 else None
        times = np.flatnonzero(finite) / frequency
        times -= times.mean()
        metrics["drift_v_per_second"] = _number(np.dot(times, centered) / np.dot(times, times))
    rails = _rails(channel)
    if rails:
        low, high, step = rails
        clipped = finite & ((np.abs(values - low) <= step * .25) | (np.abs(values - high) <= step * .25))
        metrics.update(clipping_status="computed", clipping_fraction=float(clipped.mean()),
                       longest_clipping_seconds=_longest(clipped) / frequency,
                       boundary_dwell_status="computed", boundary_dwell_fraction=float(clipped.mean()),
                       longest_boundary_dwell_seconds=_longest(clipped) / frequency,
                       boundary_source="file_encoding_limits", boundary_low_v=low, boundary_high_v=high,
                       encoding_step_v=step)
        if metrics["clipping_fraction"] >= config["clipping_fraction"] or metrics["longest_clipping_seconds"] >= config["clipping_min_seconds"]:
            # Retain the old metric names for report readers, but only surface
            # unconfirmed file bounds as review evidence (no scoring flag).
            flags.append("boundary_dwell")
    if finite.all() and len(values) >= 8 and len(values) / frequency >= 1 and np.ptp(values) > 0:
        spectrum = compute_spectrum(values, frequency, config)
        metrics.update({key: spectrum[key] for key in ("line_noise_db", "line_noise_fraction", "line_peak_width_hz", "high_frequency_ratio")})
        metrics["spectral_status"] = "computed"
        metrics["line_harmonics"] = spectrum["harmonics"]
        if spectrum["line_candidate"]:
            flags.append("line_noise")
    return metrics, flags, spectrum


def compute_spectrum(values, frequency, config):
    nfft = max(8, min(8192, int(round(2 * frequency))))
    freq, power = welch(values, fs=frequency, window="hann", nperseg=min(len(values), nfft),
                        nfft=nfft, noverlap=min(len(values), nfft) // 2, detrend="constant", scaling="density")
    resolution = frequency / nfft
    total = float(np.sum(power[freq >= 1]) * resolution)
    harmonics = []
    for center in np.arange(config["line_frequency"], frequency / 2 - 5, config["line_frequency"]):
        peak_mask = np.abs(freq - center) <= max(1, resolution)
        neighbor = (np.abs(freq - center) >= 2) & (np.abs(freq - center) <= 5)
        if not peak_mask.any() or neighbor.sum() < 2 or resolution > 2:
            continue
        indices = np.flatnonzero(peak_mask)
        index = int(indices[np.argmax(power[indices])])
        background = float(np.median(power[neighbor]))
        peak = float(power[index])
        floor = max(float(np.max(power)) * 1e-15, np.finfo(float).tiny)
        ratio = 10 * math.log10(max(peak, floor) / max(background, floor))
        left = right = index
        while left > 0 and power[left - 1] >= peak * .5:
            left -= 1
        while right < len(power) - 1 and power[right + 1] >= peak * .5:
            right += 1
        width = (right - left + 1) * resolution
        fraction = float(power[peak_mask].sum() * resolution / total) if total > 0 else None
        harmonics.append({"frequency_hz": float(center), "peak_frequency_hz": float(freq[index]),
                          "prominence_db": ratio, "power_fraction": fraction, "half_power_width_hz": width,
                          "candidate": bool(ratio >= config["line_noise_db"] and fraction is not None
                                            and fraction >= config["line_noise_fraction"] and width <= 3)})
    strongest = max(harmonics, key=lambda item: item["prominence_db"], default=None)
    return {"status": "computed", "reason": None, "frequency_hz": freq.tolist(), "power_v2_hz": power.tolist(),
            "method": "welch_hann_constant_detrend", "frequency_resolution_hz": resolution,
            "nperseg": min(len(values), nfft), "nfft": nfft, "harmonics": harmonics,
            "line_noise_db": strongest["prominence_db"] if strongest else None,
            "line_noise_fraction": sum(h["power_fraction"] or 0 for h in harmonics) if harmonics else None,
            "line_peak_width_hz": strongest["half_power_width_hz"] if strongest else None,
            "line_candidate": any(h["candidate"] for h in harmonics),
            "high_frequency_ratio": float(power[freq >= 80].sum() * resolution / total) if total > 0 and freq[-1] >= 80 else None}


def compute_relations(data, mappings, stream_id, segment_id, start, stop):
    """Same-clock pair evidence only; never establishes a damaged physical contact."""
    relations = []
    eligible = [i for i, item in enumerate(mappings) if item["unit"] == "V"
                and (item["channel_type"] == "seeg" or (item["channel_type"] == "unknown"
                     and item["candidate"]["channel_type"] not in ("ecg", "eog", "emg", "stim")))]
    for pos, i in enumerate(eligible):
        a = data[i]
        if not np.isfinite(a).all() or np.ptp(a) == 0:
            continue
        for j in eligible[pos + 1:]:
            b = data[j]
            if not np.isfinite(b).all() or np.ptp(b) == 0:
                continue
            exact = np.array_equal(a, b)
            correlation = 1.0 if exact else float(np.corrcoef(a, b)[0, 1])
            shaft_a = mappings[i]["shaft"] or mappings[i]["candidate"]["shaft"]
            shaft_b = mappings[j]["shaft"] or mappings[j]["candidate"]["shaft"]
            if exact or abs(correlation) >= .995:
                kind = "duplicate_signal_candidate" if exact else "high_correlation_candidate"
            elif shaft_a and shaft_a == shaft_b:
                kind = "same_shaft_correlation_evidence"
            else:
                continue
            relations.append({"kind": kind, "channel_ids": [mappings[i]["channel_id"], mappings[j]["channel_id"]],
                              "stream_id": stream_id, "segment_id": segment_id, "start_sample": start,
                              "stop_sample": stop, "correlation": correlation, "score_effect": "none"})
    return relations
