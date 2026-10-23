"""Windowed EDF/BDF decoding, without re-reference, filtering or resampling.

Byte layouts and TAL conventions follow EDF+ and BDF+; executable byte-authored
fixtures cover the conventions, including int24 sign, calibration and gaps.
Only annotation blocks are scanned during construction. Numerical samples are
read on demand. Clock storage grows with discontinuities, not signal length.
"""
import base64
import copy
from decimal import Decimal, InvalidOperation
import logging
import math
from numbers import Integral
import os
from pathlib import Path
import re

import numpy as np


class EphysReaderError(ValueError):
    """Input cannot be interpreted under the explicitly supported contract."""


_NUMBER = rb"[+-](?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_DURATION = rb"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_JOINED_TIMEKEEPING = re.compile(rb"^(" + _NUMBER + rb"\x14\x14)(?=" + _NUMBER + rb"[\x14\x15])")
_UNIT_SCALES = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}
_LOGGER = logging.getLogger(__name__)


def _decimal(value, label):
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise EphysReaderError("Invalid {}: {!r}".format(label, value)) from exc
    if not result.is_finite():
        raise EphysReaderError("Non-finite {}".format(label))
    if not math.isfinite(float(result)):
        raise EphysReaderError("{} exceeds supported float64 range".format(label))
    return result


def _integer(value, label):
    number = _decimal(value, label)
    if number != number.to_integral_value():
        raise EphysReaderError("{} must be an integer".format(label))
    return int(number)


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class EdfBdfReader:
    """Immutable-source reader. Caller verifies the S1 content fingerprint.

    Sample coordinates are cumulative *stored* samples in each rate stream.
    Gaps split segments; they never insert samples. Physical output uses volts
    only for recognized voltage units, retaining unknown physical units as-is.
    """

    def __init__(self, path, source_id):
        if not isinstance(source_id, str) or not source_id:
            raise EphysReaderError("source_id must be a stable non-empty string")
        self.path = Path(path)
        self.source_id = source_id
        self._warnings = []
        self._warning_index = {}
        self._events = []
        self._record_spans = []
        self._streams = []
        self._stream_lookup = {}
        with self.path.open("rb") as handle:
            self._signature = _identity(os.fstat(handle.fileno()))
            self._read_header(handle)
            self._index_time(handle)
            self._assert_unchanged(handle)
        self._build_streams()
        self._map_events()
        for warning in self._warnings:
            _LOGGER.warning("ephys_reader_degraded code=%s occurrences=%s trigger=%s",
                            warning["code"], warning["occurrences"], warning["first_context"])

    def _warn(self, code, reason, **context):
        # Aggregate repeat deviations, while retaining a concrete first location.
        if code in self._warning_index:
            self._warning_index[code]["occurrences"] += 1
            return
        warning = {"code": code, "severity": "degraded", "reason": reason,
                   "occurrences": 1, "first_context": context}
        self._warning_index[code] = warning
        self._warnings.append(warning)

    def _text(self, value):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError:
            try:
                text = value.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = value.decode("latin-1")
                encoding = "latin-1-byte-preserving"
            self._warn("header_non_ascii", "Header is not standard ASCII; original byte values are preserved.",
                       decoding=encoding)
        return text.rstrip(" ")

    def _read_header(self, handle):
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise EphysReaderError("Truncated EDF/BDF fixed header")
        if fixed[:8] == b"\xffBIOSEMI":
            self._base_format, self._width = "BDF", 3
        elif fixed[:8] == b"0       ":
            self._base_format, self._width = "EDF", 2
        else:
            raise EphysReaderError("Unsupported EDF/BDF version signature")
        self._header_bytes = _integer(self._text(fixed[184:192]), "header length")
        self._reserved = self._text(fixed[192:236])
        self._declared_records = _integer(self._text(fixed[236:244]), "record count")
        self._duration = _decimal(self._text(fixed[244:252]), "record duration")
        n_signals = _integer(self._text(fixed[252:256]), "signal count")
        if not 1 <= n_signals <= 4096 or self._header_bytes != 256 * (n_signals + 1):
            raise EphysReaderError("Invalid signal count or EDF/BDF header length")
        if self._duration < 0 or self._declared_records < -1 or (self._duration and not float(self._duration)):
            raise EphysReaderError("Negative record duration or invalid record count")
        self._format = self._base_format
        if self._reserved.startswith((self._base_format + "+C", self._base_format + "+D")):
            self._format = self._reserved[:5]
        elif self._reserved.startswith(("EDF+", "BDF+")):
            raise EphysReaderError("Unsupported or contradictory EDF/BDF continuity marker")
        self._clock = {"clock_id": self.source_id, "origin": "file_header_local_start",
                       "startdate": self._text(fixed[168:176]), "starttime": self._text(fixed[176:184]),
                       "timezone": "unknown", "cross_source_alignment": "unverified"}
        raw = handle.read(self._header_bytes - 256)
        if len(raw) != self._header_bytes - 256:
            raise EphysReaderError("Truncated EDF/BDF signal header")
        fields = [("raw_name", 16), ("transducer", 80), ("physical_unit", 8),
                  ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                  ("digital_max", 8), ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]
        self._channels = [{} for _ in range(n_signals)]
        offset = 0
        for field, width in fields:
            for index in range(n_signals):
                value = raw[offset + index * width:offset + (index + 1) * width]
                self._channels[index][field] = self._text(value)
            offset += width * n_signals
        self._annotation_indices = []
        self._channel_offsets = []
        offset = 0
        limit = 2 ** (self._width * 8 - 1)
        for index, channel in enumerate(self._channels):
            samples = _integer(channel["samples_per_record"], "samples per record")
            if not 1 <= samples <= 100000000:
                raise EphysReaderError("Invalid samples per record in channel {}".format(index))
            channel["samples_per_record"] = samples
            self._channel_offsets.append(offset)
            offset += samples * self._width
            channel.update(channel_id="{}:ch:{}".format(self.source_id, index),
                           source_index=index, channel_type="unknown", classification_status="unreviewed")
            if channel["raw_name"] in ("EDF Annotations", "BDF Annotations"):
                if samples * self._width > 16 * 1024 * 1024:
                    raise EphysReaderError("Annotation record exceeds supported 16 MiB limit")
                self._annotation_indices.append(index)
                continue
            for key in ("physical_min", "physical_max", "digital_min", "digital_max"):
                channel[key] = float(_decimal(channel[key], "channel {} {}".format(index, key)))
            if (channel["digital_min"] < -limit or channel["digital_max"] >= limit or
                    channel["digital_min"] >= channel["digital_max"] or
                    channel["digital_min"] % 1 or channel["digital_max"] % 1 or
                    channel["physical_min"] == channel["physical_max"]):
                raise EphysReaderError("Invalid calibration in channel {}".format(index))
            gain = ((channel["physical_max"] - channel["physical_min"]) /
                    (channel["digital_max"] - channel["digital_min"]))
            channel["digital_to_physical_scale"] = gain
            channel["digital_to_physical_offset"] = channel["physical_min"] - gain * channel["digital_min"]
            if not math.isfinite(gain) or not math.isfinite(channel["digital_to_physical_offset"]):
                raise EphysReaderError("Calibration exceeds supported float64 range")
            scale = _UNIT_SCALES.get(channel["physical_unit"])
            channel.update(quantity="voltage" if scale is not None else "unknown",
                           unit="V" if scale is not None else channel["physical_unit"],
                           scale_to_si=scale, unit_status="known_voltage" if scale is not None else "unknown_quantity")
            if scale is None:
                self._warn("unknown_physical_quantity", "Physical values retain their source unit; voltage metrics are unavailable.",
                           source_index=index, physical_unit=channel["physical_unit"])
        self._record_bytes = offset
        payload = self._signature[2] - self._header_bytes
        if payload < 0 or payload % self._record_bytes:
            raise EphysReaderError("File length does not contain complete EDF/BDF data records")
        self._records = payload // self._record_bytes
        if self._declared_records == -1:
            self._warn("record_count_from_complete_payload", "Unspecified record count derived from exact complete payload length.",
                       record_count=self._records)
        elif self._records != self._declared_records:
            raise EphysReaderError("File length disagrees with declared EDF/BDF record count")
        if self._duration == 0 and len(self._annotation_indices) != len(self._channels):
            raise EphysReaderError("Zero-duration numerical signals have no supported sampling frequency")
        if "+" in self._format and not self._annotation_indices:
            raise EphysReaderError("EDF+/BDF+ requires a timekeeping annotation channel")

    def _parse_tals(self, raw, record, channel_index):
        chunks = raw.split(b"\0")
        parsed = []
        byte_offset = 0
        for chunk_index, original in enumerate(chunks):
            if not original:
                byte_offset += 1
                continue
            if chunk_index == len(chunks) - 1:
                raise EphysReaderError("Unterminated TAL in record {}".format(record))
            pieces = [(original, byte_offset)]
            match = _JOINED_TIMEKEEPING.match(original) if not parsed else None
            if match:
                split = match.end()
                first_onset = _decimal(original.split(b"\x14", 1)[0].decode("ascii"), "TAL onset")
                candidate = original[split:].split(b"\x14", 1)[0].split(b"\x15", 1)[0]
                next_onset = _decimal(candidate.decode("ascii"), "candidate TAL onset")
                # A text annotation such as '+123' can be valid. Only offer the
                # known vendor interpretation for an event inside this record;
                # even then retain both interpretations because bytes alone
                # cannot prove a NUL was omitted.
                if not first_onset <= next_onset < first_onset + self._duration:
                    match = None
            if match:
                pieces = [(original[:split], byte_offset), (original[split:], byte_offset + split)]
                self._warn("timekeeping_missing_nul_separator",
                           "Ambiguous vendor recovery: split first empty TAL from an in-record timestamp; the original bytes and standard interpretation remain available.",
                           source_record_index=record, annotation_channel_index=channel_index)
            for chunk, position in pieces:
                parts = chunk.split(b"\x14")
                if len(parts) < 3 or parts[-1] != b"":
                    raise EphysReaderError("Malformed TAL delimiters in record {}".format(record))
                stamp = parts[0].split(b"\x15")
                if len(stamp) > 2 or re.fullmatch(_NUMBER, stamp[0]) is None:
                    raise EphysReaderError("Invalid TAL onset in record {}".format(record))
                if len(stamp) == 2 and re.fullmatch(_DURATION, stamp[1]) is None:
                    raise EphysReaderError("Invalid TAL duration in record {}".format(record))
                onset_text = stamp[0].decode("ascii")
                duration_text = stamp[1].decode("ascii") if len(stamp) == 2 else None
                alternative = None
                if match:
                    alternative = {"onset_seconds": float(first_onset), "onset_text": str(first_onset),
                                   "annotation_texts_base64": [base64.b64encode(text).decode("ascii")
                                                               for text in original.split(b"\x14")[1:-1] if text]}
                parsed.append({"onset": _decimal(onset_text, "TAL onset"), "onset_text": onset_text,
                               "duration_text": duration_text, "texts": parts[1:-1],
                               "raw": chunk + b"\0", "raw_source": original + b"\0", "byte_offset": position,
                               "interpretation_status": "vendor_recovered_ambiguous" if match else "standard",
                               "alternative_standard_interpretation": alternative})
            byte_offset += len(original) + 1
        return parsed

    def _index_time(self, handle):
        if not self._annotation_indices:
            if self._records:
                self._record_spans = [{"start": 0, "stop": self._records, "onset": Decimal(0)}]
            self._time_source = "continuous_file_format"
            return
        self._time_source = "per_record_TAL"
        if "+" not in self._format:
            self._warn("annotations_without_plus_marker", "TAL timestamps used although header lacks a plus-format marker.")
        previous_onset = None
        for record in range(self._records):
            record_onset = None
            for channel_index in self._annotation_indices:
                channel = self._channels[channel_index]
                offset = self._header_bytes + record * self._record_bytes + self._channel_offsets[channel_index]
                handle.seek(offset)
                tals = self._parse_tals(handle.read(channel["samples_per_record"] * self._width), record, channel_index)
                if channel_index == self._annotation_indices[0]:
                    if not tals or not tals[0]["texts"] or tals[0]["texts"][0] != b"" or tals[0]["duration_text"] is not None:
                        raise EphysReaderError("Missing first timekeeping TAL in record {}".format(record))
                    record_onset = tals[0]["onset"]
                for tal_index, tal in enumerate(tals):
                    for annotation_index, raw_text in enumerate(tal["texts"]):
                        if not raw_text:
                            continue
                        try:
                            text = raw_text.decode("utf-8")
                            encoding_status = "utf-8"
                        except UnicodeDecodeError:
                            text, encoding_status = None, "undecoded_non_utf8"
                            self._warn("annotation_non_utf8", "Annotation is not UTF-8; original bytes retained without guessing an encoding.",
                                       source_record_index=record, annotation_channel_index=channel_index)
                        if len(self._events) >= 1000000:
                            raise EphysReaderError("Event count exceeds supported metadata limit of 1000000")
                        self._events.append({"event_id": "{}:ann:{}:{}:{}:{}".format(self.source_id, channel_index, record, tal_index, annotation_index),
                                             "source_id": self.source_id, "clock_id": self.source_id,
                                             "annotation_stream_id": "{}:annotations:{}".format(self.source_id, channel_index),
                                             "source_record_index": record, "annotation_channel_index": channel_index,
                                             "tal_index": tal_index, "annotation_index": annotation_index,
                                             "source_byte_offset_in_annotation_record": tal["byte_offset"],
                                             "onset_text": tal["onset_text"], "onset_seconds": float(tal["onset"]),
                                             "duration_text": tal["duration_text"],
                                             "duration_seconds": float(tal["duration_text"]) if tal["duration_text"] is not None else None,
                                             "text": text, "encoding_status": encoding_status,
                                             "interpretation_status": tal["interpretation_status"],
                                             "alternative_standard_interpretation": tal["alternative_standard_interpretation"],
                                             "raw_text_base64": base64.b64encode(raw_text).decode("ascii"),
                                             "raw_tal_base64": base64.b64encode(tal["raw"]).decode("ascii"),
                                             "raw_source_tal_base64": base64.b64encode(tal["raw_source"]).decode("ascii")})
            if previous_onset is not None and record_onset < previous_onset + self._duration:
                raise EphysReaderError("Overlapping or backward data record timestamps at record {}".format(record))
            contiguous = previous_onset is not None and record_onset == previous_onset + self._duration
            if contiguous:
                self._record_spans[-1]["stop"] = record + 1
            else:
                self._record_spans.append({"start": record, "stop": record + 1, "onset": record_onset})
                if previous_onset is not None and self._format.endswith("+C"):
                    self._warn("continuous_flag_has_gaps", "Header says continuous; per-record TAL timestamps contain gaps, which remain separate segments.",
                               source_record_index=record)
            previous_onset = record_onset

    def _build_streams(self):
        for index, channel in enumerate(self._channels):
            if index in self._annotation_indices:
                continue
            samples = channel["samples_per_record"]
            stream_id = "{}:rate:{}".format(self.source_id, samples)
            if stream_id not in self._stream_lookup:
                segments = [{"segment_id": "{}:segment:{}".format(stream_id, span_index),
                             "start_sample": span["start"] * samples, "stop_sample": span["stop"] * samples,
                             "recording_start_seconds": float(span["onset"]),
                             "recording_start_seconds_text": str(span["onset"]),
                             "source_start_record": span["start"], "source_stop_record": span["stop"]}
                            for span_index, span in enumerate(self._record_spans)]
                stream = {"stream_id": stream_id, "source_id": self.source_id, "clock_id": self.source_id,
                          "sampling_frequency_hz": float(Decimal(samples) / self._duration),
                          "samples_per_record": samples, "sample_index_origin": 0,
                          "sample_count": self._records * samples, "channels": [], "segments": segments,
                          "time_source": self._time_source, "reference": "as_stored",
                          "processing_applied": [], "channel_order": "source_header_within_rate_group"}
                if not math.isfinite(stream["sampling_frequency_hz"]) or stream["sampling_frequency_hz"] <= 0:
                    raise EphysReaderError("Sampling frequency exceeds supported float64 range")
                self._streams.append(stream)
                self._stream_lookup[stream_id] = stream
            self._stream_lookup[stream_id]["channels"].append(channel)

    def _map_events(self):
        for event in self._events:
            onset = Decimal(event["onset_text"])
            mappings = []
            for stream in self._streams:
                samples = stream["samples_per_record"]
                for span_index, span in enumerate(self._record_spans):
                    duration = (span["stop"] - span["start"]) * self._duration
                    if span["onset"] <= onset < span["onset"] + duration:
                        position = span["start"] * samples + (onset - span["onset"]) * samples / self._duration
                        mappings.append({"stream_id": stream["stream_id"],
                                         "segment_id": stream["segments"][span_index]["segment_id"],
                                         "sample_position": float(position),
                                         "sample_index": int(position) if position == position.to_integral_value() else None,
                                         "interpretation_status": event["interpretation_status"],
                                         "alignment": "exact_sample" if position == position.to_integral_value() else "between_samples"})
                        break
            event["sample_mappings"] = mappings

    def _assert_unchanged(self, handle=None):
        current = os.fstat(handle.fileno()) if handle is not None else self.path.stat()
        if _identity(current) != self._signature or (handle is not None and _identity(self.path.stat()) != self._signature):
            raise EphysReaderError("Source file changed since reader construction; reverify input fingerprint")

    def describe(self):
        self._assert_unchanged()
        return copy.deepcopy({"reader_version": "edf-bdf-1", "source_id": self.source_id,
                              "format": self._format, "clock": self._clock,
                              "header_channel_count": len(self._channels), "record_count": self._records,
                              "record_duration_seconds": float(self._duration),
                              "annotation_channel_indices": self._annotation_indices,
                              "streams": self._streams, "events": self._events, "warnings": self._warnings,
                              "processing_applied": [], "scientific_validity": "pending"})

    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
        self._assert_unchanged()
        if representation not in ("physical", "digital"):
            raise EphysReaderError("representation must be physical or digital")
        if (isinstance(start_sample, bool) or isinstance(stop_sample, bool) or
                not isinstance(start_sample, Integral) or not isinstance(stop_sample, Integral)):
            raise EphysReaderError("Sample bounds must be integers")
        if not isinstance(stream_id, str) or not isinstance(segment_id, str):
            raise EphysReaderError("Stream and segment IDs must be strings")
        stream = self._stream_lookup.get(stream_id)
        if stream is None:
            raise EphysReaderError("Unknown stream_id")
        segment = next((span for span in stream["segments"] if span["segment_id"] == segment_id), None)
        if (segment is None or start_sample < segment["start_sample"] or stop_sample > segment["stop_sample"] or
                start_sample >= stop_sample):
            raise EphysReaderError("Window must be a non-empty half-open interval within the selected segment")
        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or
                any(not isinstance(channel_id, str) for channel_id in channel_ids) or
                len(set(channel_ids)) != len(channel_ids)):
            raise EphysReaderError("channel_ids must be a non-empty list without duplicates")
        lookup = {channel["channel_id"]: channel for channel in stream["channels"]}
        if any(channel_id not in lookup for channel_id in channel_ids):
            raise EphysReaderError("Channel is not part of the selected stream")
        channels = [lookup[channel_id] for channel_id in channel_ids]
        if len(channels) * (stop_sample - start_sample) * 8 > 256 * 1024 * 1024:
            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
        output = np.empty((len(channels), stop_sample - start_sample), dtype=np.float64)
        samples = stream["samples_per_record"]
        with self.path.open("rb") as handle:
            self._assert_unchanged(handle)
            for row, channel in enumerate(channels):
                for record in range(start_sample // samples, (stop_sample - 1) // samples + 1):
                    first = max(start_sample, record * samples)
                    last = min(stop_sample, (record + 1) * samples)
                    byte_offset = (self._header_bytes + record * self._record_bytes +
                                   self._channel_offsets[channel["source_index"]] + (first - record * samples) * self._width)
                    handle.seek(byte_offset)
                    raw = handle.read((last - first) * self._width)
                    if len(raw) != (last - first) * self._width:
                        raise EphysReaderError("Source truncated during window read")
                    if self._width == 2:
                        values = np.frombuffer(raw, dtype="<i2")
                    else:
                        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
                        values = (values ^ 0x800000) - 0x800000
                    output[row, first - start_sample:last - start_sample] = values
                if representation == "physical":
                    output[row] *= channel["digital_to_physical_scale"]
                    output[row] += channel["digital_to_physical_offset"]
                    output[row] *= channel["scale_to_si"] if channel["scale_to_si"] is not None else 1.0
            self._assert_unchanged(handle)
        return output
