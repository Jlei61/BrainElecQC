"""Direct BrainVision binary windows with index-based names and calibration."""
from copy import deepcopy
from datetime import datetime
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from .edf import EphysReaderError, _identity, _UNIT_SCALES
from .sidecars import brainvision_metadata, text_header


class BrainVisionReader:
    def __init__(self, path, source_id):
        if not isinstance(source_id, str) or not source_id:
            raise EphysReaderError("source_id must be a stable non-empty string")
        self.path, self.source_id = Path(path), source_id
        if self.path.is_symlink() or not self.path.is_file():
            raise EphysReaderError("Input must be a regular source file")
        try:
            metadata = brainvision_metadata(self.path)
            config = metadata["config"]
            self._frequency = 1e6 / float(config["Common Infos"]["SamplingInterval"])
            self._dtype = np.dtype({"INT_16": "<i2", "UINT_16": "<u2", "IEEE_FLOAT_32": "<f4"}[config["Binary Infos"]["BinaryFormat"]])
            orientation = config["Common Infos"]["DataOrientation"].upper()
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            raise EphysReaderError("Invalid BrainVision metadata: {}".format(exc)) from exc
        if not math.isfinite(self._frequency) or self._frequency <= 0 or orientation not in ("MULTIPLEXED", "VECTORIZED"):
            raise EphysReaderError("Invalid BrainVision sampling frequency or binary orientation")
        self._count = metadata["expected_samples"]
        if self._count <= 0:
            raise EphysReaderError("Empty BrainVision sample payload")
        self._signatures = {p: _identity(p.stat()) for p in metadata["files"]}
        self._stream_id = source_id + ":samples"
        self._warnings = []
        channels = []
        for i, fields in enumerate(metadata["channel_fields"]):
            if len(fields) < 4:
                raise EphysReaderError("Each BrainVision channel requires name, reference, resolution and unit")
            resolution = float(fields[2])
            if not math.isfinite(resolution) or resolution == 0:
                raise EphysReaderError("Invalid BrainVision channel resolution")
            unit = fields[3]
            scale = _UNIT_SCALES.get(unit)
            channels.append({"channel_id": "{}:ch:{}".format(source_id, i), "source_index": i,
                             "raw_name": metadata["names"][i], "reference_name": fields[1].replace("\\1", ","),
                             "channel_type": "unknown", "classification_status": "unreviewed",
                             "physical_unit": unit, "unit": "V" if scale is not None else unit,
                             "quantity": "voltage" if scale is not None else "unknown", "scale_to_si": scale,
                             "digital_to_physical_scale": resolution, "digital_to_physical_offset": 0,
                             "unit_status": "known_voltage" if scale is not None else "unknown_quantity"})
            if scale is None:
                self._warnings.append({"code": "unknown_physical_quantity", "severity": "degraded",
                                       "source_index": i, "reason": "Source unit is retained; voltage metrics are unavailable."})
        shape = (self._count, len(channels)) if orientation == "MULTIPLEXED" else (len(channels), self._count)
        mapped = np.memmap(metadata["data"], dtype=self._dtype, mode="r", shape=shape)
        self._data = mapped.T if orientation == "MULTIPLEXED" else mapped
        segments, events = self._markers(metadata["markers"])
        self._stream = {"stream_id": self._stream_id, "source_id": source_id, "clock_id": source_id,
                        "sampling_frequency_hz": self._frequency, "sample_index_origin": 0, "sample_count": self._count,
                        "channels": channels, "segments": segments, "reference": "as_stored",
                        "time_source": "explicit_brainvision_markers" if len(segments) > 1 else "source_sample_clock",
                        "processing_applied": [], "channel_order": "source_header"}
        self._description = {"reader_version": "brainvision-binary-1", "source_id": source_id, "format": "brainvision",
                             "clock": {"clock_id": source_id, "origin": "file_local_start", "timezone": "unknown",
                                       "cross_source_alignment": "unverified"},
                             "streams": [self._stream], "events": events, "warnings": self._warnings,
                             "scientific_validity": "pending", "processing_applied": []}
        self._assert_unchanged()

    def _markers(self, path):
        points, markers = [], []
        for line in text_header(path).splitlines():
            if not line.startswith("Mk") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields = value.split(",")
            if len(fields) < 5:
                raise EphysReaderError("Incomplete BrainVision marker")
            position, size, channel = int(fields[2]) - 1, int(fields[3]), int(fields[4])
            if position < 0 or size < 0 or channel < 0:
                raise EphysReaderError("Negative BrainVision marker position, size or channel")
            marker = {"event_id": self.source_id + ":" + key, "source_id": self.source_id,
                      "clock_id": self.source_id, "source_marker_id": key,
                      "source_sample_index": position, "source_position_one_based": position + 1,
                      "source_size_samples": size, "source_channel_one_based": channel,
                      "marker_type": fields[0].replace("\\1", ","), "text": fields[1].replace("\\1", ","),
                      "duration_seconds": size / self._frequency, "source_marker_text": value,
                      "interpretation_status": "source_marker"}
            if fields[0] == "New Segment":
                stamp = datetime.strptime(fields[5], "%Y%m%d%H%M%S%f") if len(fields) > 5 and fields[5] else None
                points.append((position, stamp))
            markers.append(marker)
        if len({m["event_id"] for m in markers}) != len(markers):
            raise EphysReaderError("Duplicate BrainVision marker IDs")
        if not points:
            points = [(0, None)]
        if points[0][0] != 0 or (len(points) > 1 and any(stamp is None for _, stamp in points)):
            raise EphysReaderError("Multiple BrainVision segments require source timestamps and a first segment at sample one")
        segments, last_end = [], None
        for i, (start, stamp) in enumerate(points):
            stop = points[i + 1][0] if i + 1 < len(points) else self._count
            onset = (stamp - points[0][1]).total_seconds() if stamp is not None else 0.
            if start >= stop or stop > self._count or (last_end is not None and onset < last_end - 1e-9):
                raise EphysReaderError("Overlapping or inconsistent BrainVision segment timestamps")
            segments.append({"segment_id": "{}:segment:{}".format(self._stream_id, i), "start_sample": start,
                             "stop_sample": stop, "recording_start_seconds": onset,
                             "source_timestamp": stamp.isoformat() if stamp else None, "timezone": "unknown"})
            last_end = onset + (stop - start) / self._frequency
        for marker in markers:
            sample = marker["source_sample_index"]
            segment = next((s for s in segments if s["start_sample"] <= sample < s["stop_sample"]), None)
            marker["sample_mappings"] = [] if segment is None else [{"stream_id": self._stream_id,
                "segment_id": segment["segment_id"], "sample_position": sample, "sample_index": sample, "alignment": "exact_sample"}]
            marker["onset_seconds"] = (segment["recording_start_seconds"] + (sample - segment["start_sample"]) / self._frequency
                                       if segment else None)
        return segments, markers

    def _assert_unchanged(self):
        for path, identity in self._signatures.items():
            if path.is_symlink() or _identity(path.stat()) != identity:
                raise EphysReaderError("Source bundle changed; reverify its input fingerprint")

    def describe(self):
        self._assert_unchanged()
        return deepcopy(self._description)

    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
        self._assert_unchanged()
        if representation not in ("physical", "digital") or (representation == "digital" and self._dtype.kind == "f"):
            raise EphysReaderError("Representation requires physical or integer ADC values; float storage has no ADC codes")
        if (isinstance(start_sample, bool) or isinstance(stop_sample, bool) or
                not isinstance(start_sample, Integral) or not isinstance(stop_sample, Integral)):
            raise EphysReaderError("Sample bounds must be integers")
        segment = next((s for s in self._stream["segments"] if s["segment_id"] == segment_id), None)
        if stream_id != self._stream_id or segment is None or not segment["start_sample"] <= start_sample < stop_sample <= segment["stop_sample"]:
            raise EphysReaderError("Window must be a non-empty interval within one known segment")
        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or any(not isinstance(c, str) for c in channel_ids)
                or len(set(channel_ids)) != len(channel_ids)):
            raise EphysReaderError("Select distinct ordered channel IDs")
        lookup = {c["channel_id"]: i for i, c in enumerate(self._stream["channels"])}
        if any(c not in lookup for c in channel_ids):
            raise EphysReaderError("Unknown channel ID")
        if len(channel_ids) * (stop_sample - start_sample) * 8 > 256 * 1024 * 1024:
            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
        indices = [lookup[c] for c in channel_ids]
        result = np.array(self._data[indices, start_sample:stop_sample], dtype="<f8", order="C", copy=True)
        if representation == "physical":
            for row, index in enumerate(indices):
                channel = self._stream["channels"][index]
                result[row] *= channel["digital_to_physical_scale"]
                result[row] *= channel["scale_to_si"] if channel["scale_to_si"] is not None else 1.
        if not np.isfinite(result).all():
            raise EphysReaderError("Non-finite BrainVision sample values")
        self._assert_unchanged()
        return result
