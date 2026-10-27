"""Auxiliary ECG/EOG event detection and sample-exact EEG review evidence.

Jas et al. (2018), section 3.5.2, motivates event detection followed by
epoching and averaging to inspect physiological contributions. Detection is
delegated to MNE's public reference implementation. Summary statistics and
split-half descriptions are project review evidence. All EEG epochs retain
their calibrated samples and recorded reference.
"""

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral, Real
import warnings

import numpy as np

from .features import _numeric_array


@dataclass(frozen=True)
class EEGEventConfig:
    ecg_l_freq_hz: float = 5.0
    ecg_h_freq_hz: float = 35.0
    ecg_qrs_threshold: str | float = "auto"
    eog_l_freq_hz: float = 1.0
    eog_h_freq_hz: float = 10.0
    eog_threshold_volts: float | None = None
    filter_length_seconds: float = 10.0
    min_duration_seconds: float = 10.0
    epoch_tmin_seconds: float = -0.5
    epoch_tmax_seconds: float = 0.5
    max_events_for_average: int = 1000
    max_epoch_elements_per_channel: int = 2_000_000
    event_count_window_seconds: float = 30.0

    def __post_init__(self):
        for key in ("ecg_l_freq_hz", "ecg_h_freq_hz", "eog_l_freq_hz", "eog_h_freq_hz",
                    "filter_length_seconds", "min_duration_seconds", "event_count_window_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("max_events_for_average", "max_epoch_elements_per_channel"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), Integral) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("epoch_tmin_seconds", "epoch_tmax_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not self.epoch_tmin_seconds < 0 < self.epoch_tmax_seconds:
            raise ValueError("epoch bounds must span the event at zero")
        if self.ecg_l_freq_hz >= self.ecg_h_freq_hz or self.eog_l_freq_hz >= self.eog_h_freq_hz:
            raise ValueError("event filter low cutoff must be below high cutoff")
        if self.min_duration_seconds < 3:
            raise ValueError("min_duration_seconds must cover the ECG detector's first three seconds")
        value = self.ecg_qrs_threshold
        if value != "auto" and (isinstance(value, bool) or not isinstance(value, Real)
                                or not np.isfinite(value) or not 0 < value <= 1):
            raise ValueError("ecg_qrs_threshold must be auto or a number in (0, 1]")
        value = self.eog_threshold_volts
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not np.isfinite(value) or value <= 0):
            raise ValueError("eog_threshold_volts must be positive, finite, or None")


def _config(value):
    if value is None:
        return EEGEventConfig()
    if isinstance(value, dict):
        return EEGEventConfig(**value)
    if not isinstance(value, EEGEventConfig):
        raise ValueError("config must be EEGEventConfig or a parameter dictionary")
    return value


def _fingerprint(data):
    return hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def _correlation(first, second):
    x, y = first - np.mean(first), second - np.mean(second)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else None


def _summary(row, offsets, ids, epoch_offsets):
    if not len(offsets):
        return {"event_count": 0, "mean_volts": [], "median_volts": [], "sem_volts": None,
                "chronological_half_correlation": None, "odd_even_correlation": None,
                "mean_peak_to_peak_uv": None, "median_event_peak_to_peak_uv": None,
                "event_metrics": []}
    epochs = row[np.asarray(offsets)[:, None] + epoch_offsets[None, :]]
    mean = np.mean(epochs, axis=0)
    count = len(epochs)
    sem = np.std(epochs, axis=0, ddof=1) / np.sqrt(count) if count > 1 else None
    mid = count // 2
    return {
        "event_count": count, "mean_volts": mean.tolist(),
        "median_volts": np.median(epochs, axis=0).tolist(),
        "sem_volts": sem.tolist() if sem is not None else None,
        "chronological_half_correlation": _correlation(np.mean(epochs[:mid], axis=0),
                                                       np.mean(epochs[mid:], axis=0)) if count >= 4 else None,
        "odd_even_correlation": _correlation(np.mean(epochs[::2], axis=0),
                                              np.mean(epochs[1::2], axis=0)) if count >= 4 else None,
        "mean_peak_to_peak_uv": float(np.ptp(mean) * 1e6),
        "median_event_peak_to_peak_uv": float(np.median(np.ptp(epochs, axis=1)) * 1e6),
        "event_metrics": [{"event_id": event_id, "peak_to_peak_uv": float(np.ptp(epoch) * 1e6),
                           "ac_rms_uv": float(np.std(epoch) * 1e6)}
                          for event_id, epoch in zip(ids, epochs)],
    }


def _detect(reference, sfreq, kind, cfg):
    import mne

    raw = mne.io.RawArray(reference[None], mne.create_info([kind], sfreq, [kind]), verbose=False)
    parameters = {"ch_name": kind, "l_freq": getattr(cfg, f"{kind}_l_freq_hz"),
                  "h_freq": getattr(cfg, f"{kind}_h_freq_hz"),
                  "filter_length": f"{cfg.filter_length_seconds:g}s", "tstart": 0.0,
                  "reject_by_annotation": False, "verbose": False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "ecg":
            events, _, pulse = mne.preprocessing.find_ecg_events(
                raw, qrs_threshold=cfg.ecg_qrs_threshold, **parameters)
            details = {"selected_threshold_fraction": None if cfg.ecg_qrs_threshold == "auto"
                       else float(cfg.ecg_qrs_threshold),
                       "threshold_selection": "MNE auto selection; selected fraction is not returned by its public API"
                       if cfg.ecg_qrs_threshold == "auto" else "explicit configured fraction",
                       "configured_threshold_fraction": cfg.ecg_qrs_threshold,
                       "auto_candidate_fractions": np.arange(0.3, 1.1, 0.05).tolist()
                       if cfg.ecg_qrs_threshold == "auto" else []}
        else:
            events = mne.preprocessing.find_eog_events(raw, thresh=cfg.eog_threshold_volts, **parameters)
            pulse = None
            filtered = mne.filter.filter_data(reference, sfreq, cfg.eog_l_freq_hz, cfg.eog_h_freq_hz,
                                             filter_length=f"{cfg.filter_length_seconds:g}s",
                                             l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                                             phase="zero-double", fir_window="hann", fir_design="firwin2",
                                             verbose=False)
            details = {"effective_threshold_volts": float(np.ptp(filtered) / 4)
                       if cfg.eog_threshold_volts is None else float(cfg.eog_threshold_volts),
                       "filtered_reference_peak_to_peak_uv": float(np.ptp(filtered) * 1e6),
                       "preselection_filter_band_hz": [2.0, min(45.0, sfreq / 2 - 0.75)],
                       "peak_polarity": "positive" if np.max(filtered - np.mean(filtered)) >
                       abs(np.min(filtered - np.mean(filtered))) else "negative"}
    return events[:, 0].astype(int), pulse, list(dict.fromkeys(str(w.message) for w in caught)), details


def compute_event_qc(data_volts, sfreq, channel_ids, channel_names=None, *, references=None,
                     reference_channels=None, start_sample=0, start_seconds=0.0, config=None):
    """Return ECG/EOG event tables and native-rate locked averages as JSON values.

    All provided auxiliary arrays share the EEG sample clock. ``start_sample``
    identifies the first input sample in the source stream; ``start_seconds``
    independently locates it in recording time. Epoch endpoints follow MNE's
    inclusive tmax convention internally and are exported as half-open bounds.
    Large event sets use deterministic evenly spaced chronological events;
    every detected event remains in the table with its selection state.
    """
    cfg = _config(config)
    data = _numeric_array(data_volts, 2, "data_volts")
    if 0 in data.shape:
        raise ValueError("data_volts must contain channels and samples")
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive and finite")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, Real) or not np.isfinite(start_seconds):
        raise ValueError("start_seconds must be finite")
    if isinstance(channel_ids, (str, bytes)):
        raise ValueError("channel_ids must contain distinct nonempty strings")
    ids = list(channel_ids)
    if len(ids) != len(data) or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("channel_ids must contain one distinct nonempty string per EEG channel")
    if isinstance(channel_names, (str, bytes)):
        raise ValueError("channel_names must contain one string per EEG channel")
    names = ids.copy() if channel_names is None else list(channel_names)
    if len(names) != len(ids) or any(not isinstance(name, str) for name in names):
        raise ValueError("channel_names must contain one string per EEG channel")
    references = {} if references is None else references
    if not isinstance(references, dict) or set(references) - {"ecg", "eog", "emg"}:
        raise ValueError("references accepts ecg, eog, and emg arrays")
    refs = {}
    for kind, value in references.items():
        ref = _numeric_array(value, 1, f"references[{kind}]")
        if len(ref) != data.shape[1]:
            raise ValueError(f"references[{kind}] must share the EEG sample count and clock")
        refs[kind] = ref
    reference_channels = {} if reference_channels is None else reference_channels
    if not isinstance(reference_channels, dict):
        raise ValueError("reference_channels must map auxiliary kinds to channel metadata")
    sfreq, start_sample, start_seconds = float(sfreq), int(start_sample), float(start_seconds)
    n = data.shape[1]
    if not np.isfinite(start_seconds + n / sfreq):
        raise ValueError("sample rate and time origin must produce a finite stop time")
    left, right = round(cfg.epoch_tmin_seconds * sfreq), round(cfg.epoch_tmax_seconds * sfreq)
    epoch_offsets = np.arange(left, right + 1, dtype=int)
    if len(epoch_offsets) > cfg.max_epoch_elements_per_channel:
        raise ValueError("one epoch exceeds max_epoch_elements_per_channel; shorten the epoch")
    by_kind = {}
    needs_engine = bool(set(refs) & {"ecg", "eog"})
    if needs_engine:
        try:
            import mne
        except ImportError as exc:
            raise RuntimeError("ECG/EOG event review requires MNE; install brainelecqc[readers]") from exc
        engine_version = mne.__version__
    else:
        engine_version = None
    for kind in ("ecg", "eog"):
        source = dict(reference_channels.get(kind, {}))
        source.setdefault("raw_name", kind.upper())
        source.setdefault("channel_id", None)
        detector = {
            "function": f"mne.preprocessing.find_{kind}_events", "engine_version": engine_version,
            "filter_band_hz": [getattr(cfg, f"{kind}_l_freq_hz"), getattr(cfg, f"{kind}_h_freq_hz")],
            "filter_length": f"{cfg.filter_length_seconds:g}s", "filter_phase": "zero-double",
            "fir_window": "hann", "fir_design": "firwin2", "transition_bandwidth_hz": [0.5, 0.5],
            "tstart_seconds": 0.0, "reject_by_annotation": False,
            "threshold": cfg.ecg_qrs_threshold if kind == "ecg" else cfg.eog_threshold_volts,
            "threshold_rule": ("MNE qrs_threshold auto selects a threshold using 40–160 detected events/min"
                               if kind == "ecg" and cfg.ecg_qrs_threshold == "auto" else
                               "MNE peak finder uses one quarter of the filtered EOG range"
                               if kind == "eog" and cfg.eog_threshold_volts is None else "explicit configured threshold"),
            "threshold_source": "MNE reference implementation default" if (
                (kind == "ecg" and cfg.ecg_qrs_threshold == "auto") or
                (kind == "eog" and cfg.eog_threshold_volts is None)) else "run configuration",
        }
        block = {"source_reference": source, "detector": detector, "status": "missing_reference",
                 "event_table": [], "selected_event_ids": [], "detected_event_count": 0,
                 "complete_epoch_count": 0, "boundary_excluded_count": 0, "averaged_event_count": 0,
                 "relative_times_seconds": (epoch_offsets / sfreq).tolist(), "channel_summaries": [],
                 "reference_summary": None, "warnings": [], "review_state": "pending"}
        by_kind[kind] = block
        if kind not in refs:
            continue
        ref = refs[kind]
        block["reference_sha256"] = _fingerprint(ref)
        if n / sfreq < cfg.min_duration_seconds:
            block.update(status="insufficient_duration", required_duration_seconds=cfg.min_duration_seconds)
            continue
        if sfreq / 2 <= detector["filter_band_hz"][1] + 0.5:
            block.update(status="insufficient_sampling_rate", required_nyquist_hz=detector["filter_band_hz"][1] + 0.5)
            continue
        if np.ptp(ref) == 0:
            block["status"] = "constant_reference"
            continue
        event_offsets, pulse, captured, detector_details = _detect(ref, sfreq, kind, cfg)
        detector.update(detector_details)
        block["warnings"] = captured
        valid = event_offsets[(event_offsets + left >= 0) & (event_offsets + right < n)]
        limit = min(cfg.max_events_for_average, cfg.max_epoch_elements_per_channel // len(epoch_offsets))
        selected = valid if len(valid) <= limit else valid[np.linspace(0, len(valid) - 1, limit, dtype=int)]
        selected_set = set(selected.tolist())
        selected_ids = [f"{kind}:{start_sample + int(offset)}" for offset in selected]
        for ei, offset in enumerate(event_offsets):
            offset = int(offset)
            complete = offset + left >= 0 and offset + right < n
            a, b = max(0, offset + left), min(n, offset + right + 1)
            block["event_table"].append({
                "event_id": f"{kind}:{start_sample + offset}", "sample": start_sample + offset,
                "sample_offset": offset, "time_seconds": start_seconds + offset / sfreq,
                "epoch_start_sample": start_sample + a, "epoch_stop_sample": start_sample + b,
                "epoch_start_seconds": start_seconds + a / sfreq,
                "epoch_stop_seconds": start_seconds + b / sfreq,
                "complete_epoch": bool(complete), "included_in_average": offset in selected_set,
                "status": "boundary" if not complete else "averaged" if offset in selected_set else "event_limit",
                "raw_reference_volts_at_event": float(ref[offset]),
                "interval_from_previous_seconds": float((offset - event_offsets[ei - 1]) / sfreq) if ei else None,
            })
        intervals = np.diff(event_offsets) / sfreq
        count_window = max(1, round(cfg.event_count_window_seconds * sfreq))
        block.update(
            status="computed" if len(selected) else "no_complete_epochs" if len(event_offsets) else "no_events",
            selected_event_ids=selected_ids, detected_event_count=int(len(event_offsets)),
            complete_epoch_count=int(len(valid)), boundary_excluded_count=int(len(event_offsets) - len(valid)),
            averaged_event_count=int(len(selected)), selection_rule="all complete epochs" if len(selected) == len(valid)
            else "evenly spaced chronological complete epochs within configured memory/event limit",
            detected_events_per_minute=float(len(event_offsets) * 60 * sfreq / n),
            engine_average_pulse_per_minute=float(pulse) if pulse is not None else None,
            inter_event_interval_seconds={"minimum": float(np.min(intervals)) if len(intervals) else None,
                                          "median": float(np.median(intervals)) if len(intervals) else None,
                                          "maximum": float(np.max(intervals)) if len(intervals) else None},
            overlapping_complete_epoch_pairs=int(np.sum(np.diff(valid) < len(epoch_offsets))),
            event_count_windows=[{"start_sample": start_sample + begin, "stop_sample": start_sample + min(n, begin + count_window),
                                  "start_seconds": start_seconds + begin / sfreq,
                                  "stop_seconds": start_seconds + min(n, begin + count_window) / sfreq,
                                  "event_count": int(np.sum((event_offsets >= begin) & (event_offsets < begin + count_window)))}
                                 for begin in range(0, n, count_window)],
            event_time_span_seconds=[float(start_seconds + event_offsets[0] / sfreq),
                                     float(start_seconds + event_offsets[-1] / sfreq)] if len(event_offsets) else [],
            review_checks=["核对辅助导联上的事件位置、漏检与重复检出", "逐事件查看 EEG 与辅助导联的同步波形",
                           "比较单次、平均和中位数，核对共同参考及状态变化"],
        )
        block["channel_summaries"] = [
            {"channel_id": channel_id, "channel_name": name, "channel_index": ci,
             **_summary(row, selected, selected_ids, epoch_offsets)}
            for ci, (row, channel_id, name) in enumerate(zip(data, ids, names))]
        block["reference_summary"] = _summary(ref, selected, selected_ids, epoch_offsets)
    return {
        "schema_version": 1, "method_version": "eeg-event-review-1.0", "data_sha256": _fingerprint(data),
        "reference_engine": {"package": "mne", "version": engine_version, "role": "public detector implementation reuse"},
        "method": {"paper": "Jas et al. (2018), section 3.5.2", "doi": "10.3389/fnins.2018.00530",
                   "url": "https://doi.org/10.3389/fnins.2018.00530",
                   "role": "event detection, epoching and averaging for physiological artifact review",
                   "project_statistics": "median, SEM and split-half correlations describe variability and repeatability"},
        "config": asdict(cfg), "channel_ids": ids, "channel_names": names, "sampling_rate_hz": sfreq,
        "start_sample": start_sample, "stop_sample": start_sample + n, "start_seconds": start_seconds,
        "stop_seconds": start_seconds + n / sfreq, "by_kind": by_kind,
        "epoch_processing": {"eeg_filter": None, "baseline": None, "reject": None, "projection": False,
                             "selection_basis": "reference events and complete time coverage",
                             "native_sample_offsets": [left, right], "sample_interval": "half_open",
                             "tmax_inclusive": True, "units": "V", "sem": "sample SD / sqrt(event count); descriptive dispersion",
                             "split_half": "Pearson r of mean traces; computed with at least four events",
                             "sample_alignment": "EEG and auxiliary arrays share rate, sample count and time origin"},
    }


def compare_events_to_mne(data_volts, references, result):
    """Verify integration against direct MNE event calls and MNE Epochs.

    This audits indices, sample clocks and array extraction around the reused
    detector. It compares arithmetic with the reference engine and carries no
    event-label accuracy estimate.
    """
    import mne

    cfg = EEGEventConfig(**result["config"])
    data = _numeric_array(data_volts, 2, "data_volts")
    if _fingerprint(data) != result["data_sha256"]:
        raise ValueError("data fingerprint differs from the event result")
    sfreq = result["sampling_rate_hz"]
    first = result["start_sample"]
    comparison = {}
    for kind, block in result["by_kind"].items():
        if block["status"] not in {"computed", "no_complete_epochs", "no_events"}:
            comparison[kind] = {"status": block["status"]}
            continue
        ref = _numeric_array(references[kind], 1, f"references[{kind}]")
        if _fingerprint(ref) != block["reference_sha256"]:
            raise ValueError(f"{kind} reference fingerprint differs from the event result")
        raw = mne.io.RawArray(np.vstack([data, ref]),
                             mne.create_info([f"EEG{j}" for j in range(len(data))] + [kind], sfreq,
                                             ["eeg"] * len(data) + [kind]), first_samp=first, verbose=False)
        kwargs = dict(ch_name=kind, l_freq=getattr(cfg, f"{kind}_l_freq_hz"),
                      h_freq=getattr(cfg, f"{kind}_h_freq_hz"), filter_length=f"{cfg.filter_length_seconds:g}s",
                      tstart=0.0, reject_by_annotation=False, verbose=False)
        if kind == "ecg":
            events = mne.preprocessing.find_ecg_events(raw, qrs_threshold=cfg.ecg_qrs_threshold, **kwargs)[0]
        else:
            events = mne.preprocessing.find_eog_events(raw, thresh=cfg.eog_threshold_volts, **kwargs)
        actual = np.asarray([row["sample"] for row in block["event_table"]], dtype=int)
        indices_equal = np.array_equal(events[:, 0], actual)
        selected = [row["sample"] for row in block["event_table"] if row["included_in_average"]]
        errors = {"mean_max_abs_error_volts": None, "median_max_abs_error_volts": None,
                  "sem_max_abs_error_volts": None, "relative_time_max_abs_error_seconds": None}
        if selected:
            selected_events = events[np.isin(events[:, 0], selected)]
            epochs = mne.Epochs(raw, selected_events, tmin=cfg.epoch_tmin_seconds, tmax=cfg.epoch_tmax_seconds,
                                baseline=None, proj=False, reject=None, flat=None, preload=True,
                                reject_by_annotation=False, picks=list(range(len(data))), verbose=False)
            values = epochs.get_data(copy=True)
            mean = epochs.average().data
            actual_mean = np.asarray([r["mean_volts"] for r in block["channel_summaries"]])
            actual_median = np.asarray([r["median_volts"] for r in block["channel_summaries"]])
            errors["mean_max_abs_error_volts"] = float(np.max(np.abs(mean - actual_mean)))
            errors["median_max_abs_error_volts"] = float(np.max(np.abs(np.median(values, axis=0) - actual_median)))
            if len(values) > 1:
                actual_sem = np.asarray([r["sem_volts"] for r in block["channel_summaries"]])
                errors["sem_max_abs_error_volts"] = float(np.max(np.abs(np.std(values, axis=0, ddof=1) /
                                                                        np.sqrt(len(values)) - actual_sem)))
            errors["relative_time_max_abs_error_seconds"] = float(np.max(np.abs(
                epochs.times - np.asarray(block["relative_times_seconds"]))))
        comparison[kind] = {"status": "compared", "event_indices_equal": bool(indices_equal),
                            "reference_event_count": int(len(events)), "averaged_event_count": len(selected), **errors}
    return {"reference_engine": f"MNE {mne.__version__}", "scope": "detector integration, epoch extraction and summary arithmetic",
            "by_kind": comparison}
