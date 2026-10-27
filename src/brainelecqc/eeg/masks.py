"""Identity-bound channel/sample masks with the original sample clock."""
import hashlib
from numbers import Integral, Real

import numpy as np


def data_sha256(data):
    return hashlib.sha256(np.asarray(data, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def true_runs(values):
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def validate_metadata(metadata):
    ids = metadata["channel_ids"]
    start, stop = metadata["start_sample"], metadata["stop_sample"]
    fs, time = metadata["sampling_rate_hz"], metadata["start_seconds"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(cid, str) or not cid.strip() for cid in ids)
            or len(set(ids)) != len(ids)
            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in (start, stop))
            or not 0 <= start < stop
            or any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (fs, time))
            or fs <= 0):
        raise ValueError("QC metadata requires ordered distinct IDs, integer sample indices and a finite positive sample clock")
    return ids, start, stop


def validate_binding(data, metadata):
    data = np.asarray(data)
    ids, start, stop = validate_metadata(metadata)
    if (data.ndim != 2 or data.dtype.kind not in "iuf" or not np.all(np.isfinite(data))
            or list(data.shape) != [len(ids), stop - start]
            or len(set(ids)) != len(ids) or data_sha256(data) != metadata["data_sha256"]):
        raise ValueError("QC input fingerprint, channel count or sample range disagrees with the measured data")
    return np.asarray(data, dtype=np.float64)


def apply_eeg_decisions(data_volts, decisions, *, channel_ids):
    """Return (masked copy, keep mask); excluded samples are NaN at original indices."""
    data = validate_binding(data_volts, decisions)
    if list(channel_ids) != decisions["channel_ids"]:
        raise ValueError("Channel order must match the decisions")
    keep = np.ones(data.shape, dtype=bool)
    origin, stop = decisions["start_sample"], decisions["stop_sample"]
    rows = {cid: i for i, cid in enumerate(channel_ids)}
    for item in decisions["automatic_exclusions"]:
        a, b = item["start_sample"], item["stop_sample"]
        if (item["channel_id"] not in rows or any(isinstance(v, bool) or not isinstance(v, Integral) for v in (a, b))
                or not origin <= a < b <= stop or item["decision_source"] != "automatic"):
            raise ValueError("Exclusion must identify a channel and a valid original sample interval")
        keep[rows[item["channel_id"]], a-origin:b-origin] = False
    masked = data.copy()
    masked[~keep] = np.nan
    return masked, keep


def retained_intervals(keep_mask, metadata):
    """Yield channel-specific continuous intervals, preserving half-open sample indices."""
    ids, start, stop = validate_metadata(metadata)
    keep_mask = np.asarray(keep_mask)
    if keep_mask.dtype != np.bool_ or keep_mask.shape != (len(ids), stop-start):
        raise ValueError("Keep mask must have boolean channel × sample shape matching the metadata")
    fs, origin = metadata["sampling_rate_hz"], metadata["start_sample"]
    for cid, row in zip(metadata["channel_ids"], keep_mask):
        for a, b in true_runs(row):
            yield {"channel_id": cid, "start_sample": origin+a, "stop_sample": origin+b,
                   "start_seconds": metadata["start_seconds"]+a/fs,
                   "stop_seconds": metadata["start_seconds"]+b/fs, "status": "retain"}
