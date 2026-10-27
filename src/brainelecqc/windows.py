"""Bounded raw windows and explicitly separate min/max overviews."""

import numpy as np

from .signals import verified_catalog, assert_unchanged

MAX_WINDOW_BYTES = 16 * 1024 * 1024
MAX_OVERVIEW_BINS = 4096
MAX_OVERVIEW_INPUT_BYTES = 256 * 1024 * 1024


def _validate_request(channel_ids, start_sample, stop_sample, representation):
    if (not isinstance(channel_ids, list) or not channel_ids or len(channel_ids) > 512 or
            any(not isinstance(c, str) for c in channel_ids) or len(set(channel_ids)) != len(channel_ids)):
        raise ValueError("Select 1..512 distinct ordered channel IDs")
    if type(start_sample) is not int or type(stop_sample) is not int or not 0 <= start_sample < stop_sample:
        raise ValueError("Use a nonempty half-open integer sample range [start_sample, stop_sample)")
    if representation not in ("physical", "digital"):
        raise ValueError("representation must explicitly be physical or digital")


def _select(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation):
    _validate_request(channel_ids, start_sample, stop_sample, representation)
    entry = verified_catalog(artifact_id)
    stream = next((s for s in entry["catalog"]["streams"] if s["stream_id"] == stream_id), None)
    if stream is None:
        raise LookupError("Stream not found; inspect signals for unsupported or unreadable sources")
    segment = next((s for s in stream["segments"] if s["segment_id"] == segment_id), None)
    if segment is None:
        raise LookupError("Segment not found in this stream")
    if start_sample < segment["start_sample"] or stop_sample > segment["stop_sample"]:
        raise ValueError("Window must lie within one segment; no gap interpolation or tail padding")
    by_id = {c["channel_id"]: c for c in stream["channels"]}
    if any(c not in by_id for c in channel_ids):
        raise ValueError("Requested channel does not belong to this stream")
    channels = [by_id[c] for c in channel_ids]
    n_samples = stop_sample - start_sample
    catalog = entry["catalog"]
    metadata = {key: catalog[key] for key in ("schema_version", "reader_version", "recording_id",
                                             "artifact_id", "content_sha256", "data_kind", "processing")}
    metadata.update(stream_id=stream_id, segment_id=segment_id, channel_ids=channel_ids,
                    units=["count" if representation == "digital" else c.get("unit") for c in channels],
                    shape=[len(channel_ids), n_samples], dtype="<f8", order="C",
                    start_sample=start_sample, stop_sample=stop_sample, representation=representation,
                    sampling_frequency_hz=stream["sampling_frequency_hz"], clock_id=stream["clock_id"],
                    clock_alignment=stream["clock_alignment"],
                    recording_start_seconds=segment["recording_start_seconds"] +
                    (start_sample - segment["start_sample"]) / stream["sampling_frequency_hz"],
                    source_relative_path=stream["source_relative_path"],
                    source_sha256=stream["source_sha256"], interval_convention="half_open",
                    sample_index_origin=stream["sample_index_origin"])
    for key in ("parent_artifact_id", "parent_content_sha256", "software_versions"):
        if key in catalog:
            metadata[key] = catalog[key]
    return entry, metadata


def samples(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation):
    _validate_request(channel_ids, start_sample, stop_sample, representation)
    if len(channel_ids) * max(0, stop_sample - start_sample) * 8 > MAX_WINDOW_BYTES:
        raise ValueError("Window exceeds 16 MiB; request smaller sample ranges or fewer channels")
    entry, metadata = _select(artifact_id, stream_id, segment_id, channel_ids,
                              start_sample, stop_sample, representation)
    data = entry["readers"][stream_id].read_window(stream_id, segment_id, channel_ids,
                                                   start_sample, stop_sample, representation=representation)
    data = np.asarray(data, dtype="<f8", order="C")
    if list(data.shape) != metadata["shape"]:
        raise ValueError("Reader returned a shape inconsistent with requested channel/sample order")
    assert_unchanged(entry)
    return metadata, data


def overview(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample,
             representation, bin_samples):
    _validate_request(channel_ids, start_sample, stop_sample, representation)
    if type(bin_samples) is not int or bin_samples <= 0:
        raise ValueError("bin_samples must be a positive integer")
    count = (stop_sample - start_sample + bin_samples - 1) // bin_samples
    if not 1 <= count <= MAX_OVERVIEW_BINS:
        raise ValueError("Overview must contain 1..4096 bins; increase bin_samples or shorten the range")
    if count * len(channel_ids) * 16 > MAX_WINDOW_BYTES:
        raise ValueError("Overview result exceeds 16 MiB; request fewer channels or bins")
    if (stop_sample - start_sample) * len(channel_ids) * 8 > MAX_OVERVIEW_INPUT_BYTES:
        raise ValueError("Overview scans at most 256 MiB per request; split into segment subranges")
    entry, metadata = _select(artifact_id, stream_id, segment_id, channel_ids,
                              start_sample, stop_sample, representation)
    output = np.empty((len(channel_ids), count, 2), dtype="<f8")
    starts, stops = [], []
    # Scan bounded pieces even when a single overview bin spans a very long record.
    step = max(1, (1024 * 1024) // (8 * len(channel_ids)))
    reader = entry["readers"][stream_id]
    for index, start in enumerate(range(start_sample, stop_sample, bin_samples)):
        stop = min(start + bin_samples, stop_sample)
        low, high = np.full(len(channel_ids), np.inf), np.full(len(channel_ids), -np.inf)
        for part in range(start, stop, step):
            data = reader.read_window(stream_id, segment_id, channel_ids, part, min(part + step, stop),
                                      representation=representation)
            low, high = np.minimum(low, data.min(axis=1)), np.maximum(high, data.max(axis=1))
        output[:, index, 0], output[:, index, 1] = low, high
        starts.append(start)
        stops.append(stop)
    metadata.update(source_data_kind=metadata["data_kind"], data_kind="minmax_overview", shape=list(output.shape),
                    axes=["channel", "bin", "extremum"], extrema=["min", "max"],
                    bin_samples=bin_samples, bin_start_samples=starts, bin_stop_samples=stops)
    assert_unchanged(entry)
    return metadata, output
