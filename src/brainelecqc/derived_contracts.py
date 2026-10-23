"""First processed-voltage format: explicit linear channel lineage, unchanged sample clock."""
from copy import deepcopy
import math


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalize_stream(spec, parent, artifact_id, index):
    """Validate actual clock/sample lineage; resampling needs a later explicit mapping."""
    if not isinstance(spec, dict):
        raise ValueError("Derived stream must be an object")
    stream_key = spec.get("stream_key")
    if not isinstance(stream_key, str) or not stream_key.strip():
        raise ValueError("Derived stream_key is required")
    original = next((s for s in parent["streams"] if s["stream_id"] == spec.get("parent_stream_id")), None)
    if original is None:
        raise ValueError("Derived stream must name a stream of its verified parent artifact")
    fs = spec.get("sampling_frequency_hz")
    if not _number(fs) or fs != original["sampling_frequency_hz"]:
        raise ValueError("Derived v1 keeps the parent sampling rate; resampling needs an explicit new format")
    origin, count = spec.get("sample_index_origin"), spec.get("sample_count")
    if not _integer(origin) or not _integer(count) or count == 0:
        raise ValueError("Derived sample origin/count must be explicit nonnegative/positive integers")
    reference = spec.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Derived reference must be explicit")
    channel_specs, segment_specs = spec.get("channels"), spec.get("segments")
    if not isinstance(channel_specs, list) or not 1 <= len(channel_specs) <= 512:
        raise ValueError("Derived stream must declare 1..512 channels")
    if not isinstance(segment_specs, list) or not segment_specs:
        raise ValueError("Derived stream must declare its segments")
    source_id, stream_id = f"{artifact_id}:npy:{index}", f"{artifact_id}:stream:{index}"
    by_id = {c["channel_id"]: c for c in original["channels"]}
    channels = []
    for i, channel in enumerate(channel_specs):
        if not isinstance(channel, dict) or channel.get("unit") != "V" or not isinstance(channel.get("raw_name"), str):
            raise ValueError("Derived voltage needs explicit names and SI V units")
        members, weights = channel.get("source_channel_ids"), channel.get("weights")
        if (not isinstance(members, list) or not members or any(not isinstance(c, str) for c in members)
                or len(set(members)) != len(members) or any(c not in by_id or by_id[c].get("unit") != "V" for c in members)):
            raise ValueError("Derived channel lineage must refer to known voltage channels in the parent stream")
        if not isinstance(weights, list) or len(weights) != len(members) or not all(_number(w) for w in weights):
            raise ValueError("Derived linear channel weights must be finite and match ordered source channels")
        channels.append({"channel_id": f"{stream_id}:ch:{i}", "raw_name": channel["raw_name"],
                         "unit": "V", "quantity": "voltage", "source_channel_ids": list(members),
                         "weights": list(weights), "channel_type": "unknown"})
    segments, cursor, previous_parent_stop = [], origin, None
    for i, segment in enumerate(segment_specs):
        if not isinstance(segment, dict):
            raise ValueError("Derived segment must be an object")
        start, stop = segment.get("start_sample"), segment.get("stop_sample")
        pstart, pstop = segment.get("parent_start_sample"), segment.get("parent_stop_sample")
        pseg = next((s for s in original["segments"] if s["segment_id"] == segment.get("parent_segment_id")), None)
        if (not all(_integer(n) for n in (start, stop, pstart, pstop)) or start != cursor or stop <= start
                or pseg is None or pstop - pstart != stop - start or pstart < pseg["start_sample"]
                or pstop > pseg["stop_sample"] or previous_parent_stop is not None and pstart < previous_parent_stop):
            raise ValueError("Derived ranges must cover stored samples once and map within ordered parent segments")
        onset = pseg["recording_start_seconds"] + (pstart - pseg["start_sample"]) / fs
        if (segment.get("clock_mapping") != "identity" or not _number(segment.get("recording_start_seconds"))
                or not math.isclose(segment["recording_start_seconds"], onset, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Derived segment time must retain the exact parent clock and gaps")
        segments.append({"segment_id": f"{stream_id}:segment:{i}", "start_sample": start, "stop_sample": stop,
                         "recording_start_seconds": onset, "parent_stream_id": original["stream_id"],
                         "parent_segment_id": pseg["segment_id"], "parent_start_sample": pstart,
                         "parent_stop_sample": pstop, "clock_mapping": "identity"})
        cursor, previous_parent_stop = stop, pstop
    if cursor != origin + count:
        raise ValueError("Derived segments do not cover the declared sample count")
    return {"stream_key": stream_key, "stream_id": stream_id, "source_id": source_id,
            "parent_stream_id": original["stream_id"], "clock_id": original["clock_id"],
            "clock_alignment": original["clock_alignment"], "sampling_frequency_hz": fs,
            "sample_index_origin": origin, "sample_count": count, "channels": channels,
            "channel_order": [c["channel_id"] for c in channels], "segments": segments,
            "reference": reference, "source_relative_path": f"voltage-{index}.npy"}


def validate_provenance(processing, software_versions):
    if (not isinstance(processing, list) or not processing or
            any(not isinstance(p, dict) or not isinstance(p.get("operation"), str) or not p["operation"].strip()
                or not isinstance(p.get("parameters"), dict) for p in processing)):
        raise ValueError("Processed voltage must record operations and their parameters")
    if (not isinstance(software_versions, dict) or not software_versions or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in software_versions.items())):
        raise ValueError("Processed voltage must record software versions")
    return deepcopy(processing), deepcopy(software_versions)
