"""Read a selected continuous EEG interval from the immutable recording store."""
import math
import re

import numpy as np

from ..signals import describe_artifact, verified_catalog, assert_unchanged
from ..windows import MAX_WINDOW_BYTES, samples


def channel_selection(channels, eeg_indices=None, reference_indices=None):
    """Explicit source indices take priority; otherwise use declared EEG labels/types."""
    by_index = {c["source_index"]: c for c in channels}
    if eeg_indices is None:
        eeg_indices = [c["source_index"] for c in channels if c.get("channel_type") == "eeg"
                       or re.match(r"^EEG(?:\s|[-_:]|$)", c["raw_name"], re.I)]
    if (not isinstance(eeg_indices, list) or not eeg_indices or len(set(eeg_indices)) != len(eeg_indices)
            or any(type(i) is not int or i not in by_index for i in eeg_indices)):
        raise ValueError("Select distinct EEG source indices with --eeg-channels or config.eeg_indices")
    if reference_indices is None:
        reference_indices = {}
        for kind in ("ecg", "eog", "emg"):
            pattern = r"^(?:ECG|EKG)(?:\s|[-_:]|$)" if kind == "ecg" else rf"^{kind}(?:\s|[-_:]|$)"
            found = [c["source_index"] for c in channels if c.get("channel_type") == kind
                     or re.match(pattern, c["raw_name"], re.I)]
            if found:
                reference_indices[kind] = found[0]
    if not isinstance(reference_indices, dict) or set(reference_indices)-{"ecg", "eog", "emg"}:
        raise ValueError("reference_indices maps ecg, eog or emg to source indices")
    for index in reference_indices.values():
        if type(index) is not int or index not in by_index or index in eeg_indices:
            raise ValueError("References must identify auxiliary channels in the same stream")
    selected = [by_index[i] for i in eeg_indices]
    refs = {kind: by_index[i] for kind, i in reference_indices.items()}
    if any(c.get("unit") != "V" for c in selected+list(refs.values())):
        raise ValueError("Resolve EEG/reference physical units to volts before running QC")
    return selected, refs


def read_eeg_input(artifact_id, *, stream_id=None, segment_id=None, start_seconds=None,
                   duration_seconds=None, eeg_indices=None, reference_indices=None):
    catalog = describe_artifact(artifact_id)
    streams = [s for s in catalog["streams"] if stream_id is None or s["stream_id"] == stream_id]
    if len(streams) != 1:
        raise ValueError("Choose one stream_id from describe_artifact() for this QC run")
    stream = streams[0]
    segments = [s for s in stream["segments"] if segment_id is None or s["segment_id"] == segment_id]
    if len(segments) != 1:
        raise ValueError("Choose one continuous segment_id from describe_artifact() for this QC run")
    segment, fs = segments[0], stream["sampling_frequency_hz"]
    base = segment["recording_start_seconds"]
    start_seconds = base if start_seconds is None else start_seconds
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, (int, float)) or not math.isfinite(start_seconds):
        raise ValueError("start_seconds must be finite")
    if duration_seconds is not None and (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float))
                                         or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive and finite")
    start = segment["start_sample"]+round((start_seconds-base)*fs)
    stop = segment["stop_sample"] if duration_seconds is None else start+round(duration_seconds*fs)
    if not segment["start_sample"] <= start < stop <= segment["stop_sample"]:
        raise ValueError("Select a nonempty interval entirely inside the continuous segment")
    eeg, references = channel_selection(stream["channels"], eeg_indices, reference_indices)
    ordered = list(dict.fromkeys(c["channel_id"] for c in eeg+list(references.values())))
    if len(ordered)*(stop-start)*8 > 256*1024*1024:
        raise ValueError("QC interval exceeds 256 MiB; specify a shorter duration_seconds and process successive intervals")
    data = np.empty((len(ordered), stop-start), dtype="<f8")
    step, metadata = max(1, MAX_WINDOW_BYTES//(len(ordered)*8)), None
    for offset in range(start, stop, step):
        part_meta, part = samples(artifact_id, stream["stream_id"], segment["segment_id"],
                                  ordered, offset, min(stop, offset+step), "physical")
        metadata = part_meta if metadata is None else metadata
        data[:, offset-start:offset-start+part.shape[1]] = part
    metadata.update(stop_sample=stop, shape=list(data.shape))
    bounds = {}
    for c in eeg:
        if all(c.get(k) is not None for k in ("physical_min", "physical_max", "scale_to_si", "digital_to_physical_scale")):
            lo, hi = sorted([c["physical_min"]*c["scale_to_si"], c["physical_max"]*c["scale_to_si"]])
            bounds[c["channel_id"]] = {"lower_volts": lo, "upper_volts": hi,
                "tolerance_volts": abs(c["digital_to_physical_scale"]*c["scale_to_si"])*0.25,
                "source": "file_declared_calibration_bounds"}
    from .header_qc import inspect_input_header
    entry = verified_catalog(artifact_id)
    header = None
    if catalog["data_kind"] == "raw_samples":
        source = next(s for s in catalog["sources"] if s["source_id"] == stream["source_id"])
        header = inspect_input_header(entry["root"]/"files"/source["relative_path"], expected_sha256=source["sha256"])
    assert_unchanged(entry)
    return {"data_volts": data[:len(eeg)], "channels": eeg, "metadata": metadata,
            "header_qc": header,
            "calibration_bounds": bounds, "references": {k: data[ordered.index(c["channel_id"])] for k, c in references.items()},
            "reference_channels": {k: {"channel_id": c["channel_id"], "raw_name": c["raw_name"],
                                      "source_index": c["source_index"]} for k, c in references.items()},
            "selection": {"eeg_indices": [c["source_index"] for c in eeg],
                          "reference_indices": {k: c["source_index"] for k, c in references.items()}}}
