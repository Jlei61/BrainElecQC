"""Read versioned derived voltages through bounded, read-only NumPy mappings."""
from copy import deepcopy
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .edf import EphysReaderError, _identity


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum or value > 2 ** 63 - 1:
        raise EphysReaderError("{} must be an integer >= {}".format(label, minimum))
    return int(value)


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise EphysReaderError("{} must be an explicit non-empty string".format(label))
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EphysReaderError("{} must be a finite number".format(label))
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EphysReaderError("{} exceeds float64 range".format(label)) from exc
    if not math.isfinite(result):
        raise EphysReaderError("{} must be a finite number".format(label))
    return result


class NpyVoltageReader:
    """Only C-order little-endian float64 volts, with explicit parent mappings.

    Construction validates the descriptor and file layout, without scanning
    samples. Finite values are checked on the requested window. Publication is
    responsible for the complete provenance and the artifact's content hash.
    """

    def __init__(self, path, stream):
        self.path = Path(path)
        self._stream = deepcopy(stream)
        try:
            self._validate_stream()
        except (KeyError, TypeError) as exc:
            raise EphysReaderError("Incomplete derived stream descriptor: {}".format(exc)) from exc
        self._signature = _identity(self.path.stat())
        try:
            self._data = np.load(self.path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise EphysReaderError("Cannot map derived voltage NPY: {}".format(exc)) from exc
        if not isinstance(self._data, np.memmap):
            if hasattr(self._data, "close"):
                self._data.close()
            raise EphysReaderError("Derived voltage must be a single NPY array")
        expected = (len(self._stream["channels"]), self._stream["sample_count"])
        if (self._data.dtype.str != "<f8" or self._data.shape != expected or
                not self._data.flags.c_contiguous):
            raise EphysReaderError("Derived voltage requires C-contiguous <f8 with shape channel x stored sample")
        if self._data.offset + self._data.nbytes != self._signature[2]:
            raise EphysReaderError("NPY payload length does not exactly match its array shape")
        self._assert_unchanged()

    def _validate_stream(self):
        stream = self._stream
        if not isinstance(stream, dict):
            raise EphysReaderError("Derived stream descriptor must be an object")
        for name in ("stream_id", "source_id", "clock_id"):
            _text(stream[name], name)
        frequency = _number(stream["sampling_frequency_hz"], "sampling_frequency_hz")
        if frequency <= 0:
            raise EphysReaderError("sampling_frequency_hz must be positive")
        origin = _integer(stream["sample_index_origin"], "sample_index_origin")
        count = _integer(stream["sample_count"], "sample_count", 1)
        channels = stream["channels"]
        if not isinstance(channels, list) or not channels:
            raise EphysReaderError("Derived channels must be a non-empty ordered list")
        seen = set()
        for channel in channels:
            channel_id = _text(channel["channel_id"], "channel_id")
            if channel_id in seen or channel["unit"] != "V" or not isinstance(channel["raw_name"], str):
                raise EphysReaderError("Derived channels require unique IDs, preserved names and explicit SI unit V")
            seen.add(channel_id)
            parents = channel["source_channel_ids"]
            if not isinstance(parents, list) or not parents:
                raise EphysReaderError("Derived channel requires explicit source_channel_ids")
            for parent in parents:
                _text(parent, "source_channel_id")
            if len(set(parents)) != len(parents):
                raise EphysReaderError("Duplicate source_channel_ids are ambiguous")
        segments = stream["segments"]
        if not isinstance(segments, list) or not segments:
            raise EphysReaderError("Derived segments must be a non-empty list")
        next_sample, previous_time_stop, segment_ids = origin, None, set()
        for segment in segments:
            segment_id = _text(segment["segment_id"], "segment_id")
            start = _integer(segment["start_sample"], "start_sample")
            stop = _integer(segment["stop_sample"], "stop_sample")
            onset = _number(segment["recording_start_seconds"], "recording_start_seconds")
            for field in ("parent_stream_id", "parent_segment_id"):
                _text(segment[field], field)
            parent_start = _integer(segment["parent_start_sample"], "parent_start_sample")
            parent_stop = _integer(segment["parent_stop_sample"], "parent_stop_sample")
            if (segment_id in segment_ids or start != next_sample or stop <= start or
                    parent_stop - parent_start != stop - start or segment["clock_mapping"] != "identity"):
                raise EphysReaderError("Segments must exactly cover stored samples with explicit identity parent mappings")
            if previous_time_stop is not None and onset < previous_time_stop and not math.isclose(onset, previous_time_stop, rel_tol=0, abs_tol=1e-12):
                raise EphysReaderError("Derived segment clocks overlap or move backward")
            segment_ids.add(segment_id)
            next_sample = stop
            previous_time_stop = onset + (stop - start) / frequency
            if not math.isfinite(previous_time_stop):
                raise EphysReaderError("Derived segment clock exceeds float64 range")
        if next_sample != origin + count:
            raise EphysReaderError("Segments do not cover the declared stored sample count")

    def _assert_unchanged(self):
        try:
            unchanged = _identity(self.path.stat()) == self._signature
        except OSError as exc:
            raise EphysReaderError("Derived voltage file changed or disappeared") from exc
        if not unchanged:
            raise EphysReaderError("Derived voltage file changed; reverify artifact fingerprint")

    def describe(self):
        self._assert_unchanged()
        return {"format": "NPY_FLOAT64_VOLTAGE", "source_id": self._stream["source_id"],
                "streams": [deepcopy(self._stream)], "events": [], "warnings": [],
                "value_validation": "finite_per_requested_window", "scientific_validity": "pending"}

    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
        self._assert_unchanged()
        if representation != "physical":
            raise EphysReaderError("Derived voltages support physical representation only; ADC codes are unavailable")
        start = _integer(start_sample, "start_sample")
        stop = _integer(stop_sample, "stop_sample")
        if stream_id != self._stream["stream_id"]:
            raise EphysReaderError("Unknown derived stream_id")
        segment = next((s for s in self._stream["segments"] if s["segment_id"] == segment_id), None)
        if segment is None or not segment["start_sample"] <= start < stop <= segment["stop_sample"]:
            raise EphysReaderError("Window must be a non-empty half-open interval within the selected segment")
        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or
                any(not isinstance(c, str) for c in channel_ids) or len(set(channel_ids)) != len(channel_ids)):
            raise EphysReaderError("Select distinct ordered channel IDs")
        lookup = {channel["channel_id"]: index for index, channel in enumerate(self._stream["channels"])}
        if any(c not in lookup for c in channel_ids):
            raise EphysReaderError("Channel is not part of the selected derived stream")
        if len(channel_ids) * (stop - start) * 8 > 256 * 1024 * 1024:
            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
        origin = self._stream["sample_index_origin"]
        # Advanced row indexing copies only the bounded column slice, never all
        # stored samples. Return an owned C-order array; callers cannot edit input.
        result = np.array(self._data[[lookup[c] for c in channel_ids], start - origin:stop - origin],
                          dtype="<f8", order="C", copy=True)
        if not np.isfinite(result).all():
            raise EphysReaderError("Requested derived voltage window contains non-finite values")
        self._assert_unchanged()
        return result
