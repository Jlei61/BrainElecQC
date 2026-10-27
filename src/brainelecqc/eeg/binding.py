"""Validate that saved measurements, decisions and figures address one input."""
from numbers import Integral

import numpy as np

from .masks import apply_eeg_decisions, data_sha256, validate_binding


def validate_result_identity(result):
    """Check hashes, channel order, source clock, masks and auxiliary evidence.

    Used before both numeric export and plotting. Regenerated bundles can omit
    ``masked_data_volts`` and ``channels``; their identity still comes from the
    features, manifest, decisions and method blocks.
    """
    try:
        _validate(result)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Incomplete EEG result identity: {exc}") from exc
    return True


def _validate(result):
    f = result["features"]
    data = validate_binding(result["raw_data_volts"], f)
    ids, names = f["channel_ids"], f["channel_names"]
    if len(names) != len(ids):
        raise ValueError("Feature channel names disagree with channel identity")
    origin, stop, clock, fs = f["start_sample"], f["stop_sample"], f["start_seconds"], f["sampling_rate_hz"]
    digest = data_sha256(data)

    def same(actual, expected, location):
        if actual != expected:
            raise ValueError(f"EEG result identity mismatch at {location}")

    def time(actual, expected, location):
        if isinstance(actual, bool) or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError(f"EEG sample clock mismatch at {location}")

    def channel(row, location):
        if row["channel_id"] not in ids:
            raise ValueError(f"Unknown channel identity at {location}")
        index = ids.index(row["channel_id"])
        if "channel_index" in row:
            if isinstance(row["channel_index"], bool) or not isinstance(row["channel_index"], Integral):
                raise ValueError(f"Channel index identity must be an integer at {location}")
            same(row["channel_index"], index, f"{location}.channel_index")
        if "channel_name" in row:
            same(row["channel_name"], names[index], f"{location}.channel_name")

    def interval(row, location, *, full=False):
        a, b = row["start_sample"], row["stop_sample"]
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (a, b)) or not origin <= a < b <= stop:
            raise ValueError(f"Sample interval identity mismatch at {location}")
        if full:
            same((a, b), (origin, stop), f"{location}.input_range")
        time(row["start_seconds"], clock + (a - origin) / fs, f"{location}.start_seconds")
        if "stop_seconds" in row:
            time(row["stop_seconds"], clock + (b - origin) / fs, f"{location}.stop_seconds")

    def ordered(rows, location):
        same([row["channel_id"] for row in rows], ids, f"{location}.channel_ids")
        for row in rows:
            channel(row, location)

    def common(block, location, *, hash_key="data_sha256", rate_key="sampling_rate_hz"):
        same(block[hash_key], digest, f"{location}.fingerprint")
        same(block["channel_ids"], ids, f"{location}.channel_ids")
        if "channel_names" in block:
            same(block["channel_names"], names, f"{location}.channel_names")
        time(block[rate_key], fs, f"{location}.sampling_rate")

    interval(f, "features", full=True)
    ordered(f["channel_features"], "feature summaries")
    for row in f["windows"]:
        interval(row, "feature window")
    manifest = result["manifest"]
    common(manifest, "manifest")
    interval(manifest, "manifest", full=True)
    same(manifest["unit"], "V", "manifest.unit")
    same(f["units"]["input"], "V", "features.unit")
    decisions = result["decisions"]
    common(decisions, "decisions")
    interval(decisions, "decisions", full=True)
    ordered(decisions["channel_summary"], "decision channel summaries")
    for row in decisions["review_queue"] + decisions["automatic_exclusions"]:
        channel(row, "decision interval")
        interval(row, "decision interval")
    if decisions.get("decision_mode") == "review_only" and decisions["automatic_exclusions"]:
        raise ValueError("Review-only decisions must retain the input samples")
    masked, expected_mask = apply_eeg_decisions(data, decisions, channel_ids=ids)
    keep = np.asarray(result["keep_mask"])
    if keep.dtype != np.bool_ or not np.array_equal(keep, expected_mask):
        raise ValueError("Keep mask disagrees with the input-bound decisions")
    if "masked_data_volts" in result and not np.array_equal(np.asarray(result["masked_data_volts"]), masked, equal_nan=True):
        raise ValueError("Retained sample values disagree with the input-bound decisions")
    if "channels" in result:
        ordered(result["channels"], "source channels")
        same([row["raw_name"] for row in result["channels"]], names, "source channel names")
    if "method_results" not in result:
        return
    methods = result["method_results"]
    if methods.get("header") is not None:
        same(methods["header"]["source_sha256"], manifest["source_sha256"], "header.source_sha256")
        header = methods["header"]
        if header.get("auto_points") is not None:
            from .header_qc import CHECKS
            same([(c["id"], c["label"], c["max_points"]) for c in header["checks"]], CHECKS, "header.checks")
            for c in header["checks"]:
                same(c["points"], c["max_points"] if c["status"] == "pass" else 0, "header.points")
            same(sum(c["points"] for c in header["checks"]), header["auto_points"], "header.total")
    bad = methods["bad_channels"]
    common(bad, "PREP", rate_key="sample_rate")
    interval(bad["input_range"], "PREP", full=True)
    ordered(bad["channels"], "PREP channel summaries")
    for row in bad["windows"]:
        interval(row, "PREP window")
    for values in bad["window_metrics"].values():
        if np.asarray(values).shape != (len(ids), len(bad["windows"])):
            raise ValueError("PREP channel/window axes disagree with identity")
    for row in bad["candidates"]:
        channel(row, "PREP candidate")
        interval(row, "PREP candidate")
    continuity = methods.get("continuity")
    if "continuity" in methods:
        common(continuity, "continuity")
        same(continuity["channel_names"], names, "continuity.channel_names")
        same(continuity["unit"], "V", "continuity.unit")
        interval(continuity["input_range"], "continuity", full=True)
        ordered(continuity["measurements"], "continuity channel measurements")
        for row in continuity["candidates"]:
            channel(row, "continuity candidate")
            interval(row, "continuity candidate")
            same(row["sample_count"], row["stop_sample"] - row["start_sample"], "continuity.sample_count")
            time(row["duration_seconds"], row["sample_count"] / fs, "continuity.duration_seconds")
    muscle = methods.get("muscle")
    if "muscle" in methods:
        common(muscle, "muscle", rate_key="sample_rate_hz")
        same(muscle["channel_names"], names, "muscle.channel_names")
        same(muscle["unit"], "V", "muscle.unit")
        interval(muscle["input_range"], "muscle", full=True)
        measured = muscle["status"] not in {"insufficient_sampling_rate", "insufficient_duration"}
        metrics = muscle["sample_metrics"]
        if measured:
            ordered(muscle["channels"], "muscle channel measurements")
            for key in ("envelope_uv", "z_score", "excluded"):
                if len(metrics[key]) != len(ids):
                    raise ValueError(f"Muscle {key} channel axis disagrees with identity")
                for index, values in enumerate(metrics[key]):
                    unavailable = muscle["channels"][index]["status"] == "nonfinite_input"
                    if unavailable:
                        if values is not None:
                            raise ValueError(f"Muscle {key} must mark unavailable channel values explicitly")
                        continue
                    array = np.asarray(values)
                    if array.shape != (data.shape[1],):
                        raise ValueError(f"Muscle {key} sample axis disagrees with identity")
                    if key == "excluded":
                        if array.dtype != np.bool_:
                            raise ValueError("Muscle exclusion mask must contain booleans")
                    elif array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                        raise ValueError(f"Muscle {key} contains invalid sample values")
        else:
            for key in ("channels", "windows", "candidates"):
                same(muscle[key], [], f"muscle.unavailable.{key}")
            for key in ("envelope_uv", "z_score", "excluded"):
                same(metrics[key], [], f"muscle.unavailable.{key}")
        for index, row in enumerate(muscle["windows"]):
            same(row["window_index"], index, "muscle.window_index")
            interval(row, "muscle window")
        for row in muscle["channels"]:
            for excluded in row.get("excluded_intervals", []):
                interval(excluded, "muscle excluded interval")
            expected_count = 0 if row["status"] == "nonfinite_input" else len(muscle["windows"])
            same(len(row["window_metrics"]), expected_count, "muscle channel window count")
            for index, window in enumerate(row["window_metrics"]):
                interval(window, "muscle channel window")
                same(window["window_index"], index, "muscle channel window index")
                expected = muscle["windows"][index]
                same((window["start_sample"], window["stop_sample"]),
                     (expected["start_sample"], expected["stop_sample"]), "muscle channel window samples")
                interval(window["correlation_range"], "muscle correlation range")
        for row in muscle["candidates"]:
            channel(row, "muscle candidate")
            interval(row, "muscle candidate")
            indices = row["window_indices"]
            if (not indices or len(indices) != len(set(indices)) or
                    any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < len(muscle["windows"]) for v in indices)):
                raise ValueError("Muscle candidate window indices disagree with identity")
        reference = muscle["reference"]
        provided = "emg" in result["references"]
        if provided != (reference["data_sha256"] is not None):
            raise ValueError("EMG reference availability disagrees with muscle evidence")
        if provided:
            auxiliary = np.asarray(result["references"]["emg"])
            if auxiliary.shape != (data.shape[1],) or auxiliary.dtype.kind not in "fiu":
                raise ValueError("EMG reference clock/shape disagrees with EEG")
            same(data_sha256(auxiliary), reference["data_sha256"], "muscle.EMG.reference_fingerprint")
            declared = f.get("reference_channels", {}).get("emg")
            if declared is not None:
                for key in ("channel_id", "raw_name", "source_index"):
                    same(reference["source_reference"].get(key), declared.get(key), f"muscle.EMG.source_reference.{key}")
        for row in reference.get("excluded_intervals", []):
            interval(row, "EMG excluded interval")
        has_reference_metrics = measured and provided and reference["status"] == "ok"
        for key in ("reference_envelope_uv", "reference_z_score"):
            if has_reference_metrics:
                values = np.asarray(metrics[key])
                if values.shape != (data.shape[1],) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                    raise ValueError(f"Muscle {key} sample axis/values disagree with EMG identity")
            elif metrics[key] is not None:
                raise ValueError(f"Muscle {key} availability disagrees with EMG analysis status")
    spectral = methods["spectral"]
    common(spectral["input"], "spectral", rate_key="sfreq")
    interval(spectral["input"], "spectral", full=True)
    same(spectral["input"]["n_samples"], data.shape[1], "spectral.n_samples")
    psd = spectral["psd"]
    same(psd["channel_ids"], ids, "PSD.channel_ids")
    interval(psd, "PSD", full=True)
    if np.asarray(psd["power_v2_hz"]).shape != (len(ids), len(psd["frequencies_hz"])):
        raise ValueError("PSD channel/frequency axes disagree with identity")
    alpha = spectral.get("alpha_power")
    if alpha is not None:
        common(alpha, "alpha power", rate_key="sfreq")
        ordered(alpha["full"], "alpha power summaries")
        estimator = psd["estimator"]
        same(alpha["alpha_band_hz"], list(spectral["config"]["alpha_band_hz"]), "alpha band")
        same(alpha["reference_band_hz"], list(spectral["config"]["fit_range_hz"]), "alpha reference band")
        interval(alpha["input_interval"], "alpha input", full=True)
        for row in alpha["full"]:
            interval(row, "alpha full")
            same((row["start_sample"], row["stop_sample"]),
                 (origin, origin + estimator["covered_samples"]), "alpha Welch coverage")
        starts = estimator["periodogram_start_offsets"]
        same(len(alpha["windows"]), len(starts)*len(ids), "alpha window count")
        for window_index, left in enumerate(starts):
            rows = alpha["windows"][window_index*len(ids):(window_index+1)*len(ids)]
            ordered(rows, "alpha window channels")
            for row in rows:
                interval(row, "alpha window")
                same(row["window_index"], window_index, "alpha.window_index")
                same((row["start_sample"], row["stop_sample"]),
                     (origin + left, origin + left + estimator["nperseg"]), "alpha window samples")
    for row in spectral["line_noise"]["windows"]:
        interval(row, "line window")
    for row in spectral["line_noise"]["observations"]:
        channel(row, "line observation")
        interval(row, "line observation")
        expected = spectral["line_noise"]["windows"][row["window_index"]]
        same((row["start_sample"], row["stop_sample"]), (expected["start_sample"], expected["stop_sample"]), "line.window_index")
    local = spectral["line_noise"].get("strongest_window_psd")
    if local is not None:
        channel(local, "local line PSD")
        interval(local, "local line PSD")
        if len(local["power_v2_hz"]) != len(local["frequencies_hz"]):
            raise ValueError("Local PSD frequency axis disagrees with identity")
    ordered(spectral["parameterization"]["full"], "spectral parameterization")
    for scope in ("full", "windows"):
        for row in spectral["parameterization"][scope]:
            channel(row, "spectral fit")
            interval(row, "spectral fit", full=scope == "full")
    event = methods["events"]
    common(event, "events")
    interval(event, "events", full=True)
    epoch_left, epoch_right = event["epoch_processing"]["native_sample_offsets"]
    expected_relative = np.arange(epoch_left, epoch_right + 1) / fs
    for kind, block in event["by_kind"].items():
        if not np.array_equal(np.asarray(block["relative_times_seconds"]), expected_relative):
            raise ValueError(f"{kind} epoch relative clock disagrees with sample offsets")
        provided = kind in result["references"]
        if provided != ("reference_sha256" in block):
            raise ValueError(f"{kind} reference availability disagrees with event evidence")
        if not provided:
            continue
        auxiliary = np.asarray(result["references"][kind])
        if auxiliary.shape != (data.shape[1],) or not np.all(np.isfinite(auxiliary)):
            raise ValueError(f"{kind} reference clock/shape disagrees with EEG")
        same(data_sha256(auxiliary), block["reference_sha256"], f"{kind}.reference_fingerprint")
        declared = f.get("reference_channels", {}).get(kind)
        if declared is not None:
            for key in ("channel_id", "raw_name", "source_index"):
                same(block["source_reference"].get(key), declared.get(key), f"{kind}.source_reference.{key}")
        table = block["event_table"]
        event_ids = [row["event_id"] for row in table]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"Duplicate {kind} event identities")
        for row in table:
            sample = row["sample"]
            if isinstance(sample, bool) or not isinstance(sample, Integral) or not origin <= sample < stop:
                raise ValueError(f"{kind} event sample is outside its input")
            same(row["sample_offset"], sample - origin, f"{kind}.sample_offset")
            time(row["time_seconds"], clock + (sample - origin) / fs, f"{kind}.event_time")
            complete = sample + epoch_left >= origin and sample + epoch_right < stop
            same(row["complete_epoch"], complete, f"{kind}.complete_epoch")
            if not complete and row["included_in_average"]:
                raise ValueError(f"Incomplete {kind} boundary epoch was included in the average")
            same((row["epoch_start_sample"], row["epoch_stop_sample"]),
                 (max(origin, sample + epoch_left), min(stop, sample + epoch_right + 1)), f"{kind}.epoch_offsets")
            interval({key: row[f"epoch_{key}"] for key in ("start_sample", "stop_sample", "start_seconds", "stop_seconds")}, f"{kind}.epoch")
        selected = [row["event_id"] for row in table if row["included_in_average"]]
        same(block["selected_event_ids"], selected, f"{kind}.selected_event_ids")
        if block["channel_summaries"]:
            ordered(block["channel_summaries"], f"{kind} summaries")
            count = len(block["relative_times_seconds"])
            for row in block["channel_summaries"]:
                if row["event_count"] != len(selected) or len(row["mean_volts"]) != (count if selected else 0):
                    raise ValueError(f"{kind} averaged event axes disagree with identity")
