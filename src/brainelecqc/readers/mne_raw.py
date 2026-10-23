"""Lazy optional-format adapter with explicit calibration and clock limits.

This adapter never calls filtering, referencing, resampling, projection or
channel picking at construction. Raw EDF/BDF uses the separate format reader
because mixed rates and EDF+D clocks must not be coerced by another library.
"""
from copy import deepcopy
import logging
import math
from numbers import Integral
from pathlib import Path
import warnings

import numpy as np

from .edf import EphysReaderError, _identity
from .sidecars import eeglab_metadata, nicolet_metadata

_LOG = logging.getLogger(__name__)


class MneRawReader:
    def __init__(self, path, source_id, format_name):
        if not isinstance(source_id, str) or not source_id:
            raise EphysReaderError("source_id must be a stable non-empty string")
        self.path, self.source_id, self.format_name = Path(path), source_id, format_name
        if self.path.is_symlink() or not self.path.is_file():
            raise EphysReaderError("Input must be a regular source file")
        self._warnings = []
        preflight = {"eeglab": eeglab_metadata,
                     "nicolet_data": nicolet_metadata}
        try:
            metadata = preflight[format_name](self.path) if format_name in preflight else {"files": [self.path]}
        except (KeyError, TypeError, OverflowError) as exc:
            raise EphysReaderError("Incomplete {} metadata: {}".format(format_name, exc)) from exc
        except ImportError as exc:
            raise EphysReaderError("This format requires the optional 'readers' dependencies: pip install 'brainelecqc[readers]'") from exc
        files = metadata["files"]
        if format_name == "nihon":
            files = files + [p for p in self.path.parent.iterdir() if p.stem.lower() == self.path.stem.lower()
                             and p.suffix.lower() in (".pnt", ".log", ".21e")]
        self._signatures = {p: _identity(p.stat()) for p in files}
        try:
            import mne
        except ImportError as exc:
            raise EphysReaderError("This format requires the optional 'readers' dependencies: pip install 'brainelecqc[readers]'") from exc
        self._mne_version = mne.__version__
        factories = {"eeglab": (mne.io.read_raw_eeglab, {"eog": ()}),
                     "fif": (mne.io.read_raw_fif, {"on_split_missing": "raise"}),
                     "nicolet_data": (mne.io.read_raw_nicolet, {"ch_type": "eeg", "eog": (), "ecg": (), "emg": ()}),
                     "neuroscan_cnt": (mne.io.read_raw_cnt, {"eog": (), "ecg": (), "emg": ()}),
                     "nihon": (mne.io.read_raw_nihon, {})}
        try:
            factory, options = factories[format_name]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self._raw = factory(self.path, preload=False, verbose="ERROR", **options)
            for warning in caught:
                if "Channel names are not unique" in str(warning.message) and "names" not in metadata:
                    raise EphysReaderError("Duplicate labels need independent source-name metadata; decoder renaming is not accepted")
                self._warn("decoder_warning", str(warning.message))
        except (OSError, ValueError, TypeError, NotImplementedError, IndexError, RuntimeError) as exc:
            raise EphysReaderError("{} decoder rejected input: {}".format(format_name, exc)) from exc
        if self._raw.preload:
            self._raw.close()
            raise EphysReaderError("Format decoder attempted whole-record preloading; this input is not supported")
        for filename in self._raw.filenames:
            if filename is None:
                raise EphysReaderError("Decoded signal lacks a physical input filename")
            file = Path(filename)
            if file.resolve().parent != self.path.resolve().parent or file.is_symlink():
                raise EphysReaderError("Decoder referenced a file outside the imported input directory")
            self._signatures.setdefault(file, _identity(file.stat()))
        if metadata.get("expected_samples", self._raw.n_times) != self._raw.n_times:
            raise EphysReaderError("Decoder sample count differs from exact source payload")
        if format_name == "nihon":
            header = self._raw._raw_extras[0].get("header", {})
            blocks = [b for c in header.get("controlblocks", []) for b in c.get("datablocks", [])]
            if len(blocks) != 1:
                raise EphysReaderError("Multi-block Nihon files need source-timestamp validation; concatenation is not permitted")
        if format_name in ("nihon", "neuroscan_cnt"):
            self._warn("vendor_sample_not_validated", "The optional decoder is wired, but this vendor format has no independent numeric sample acceptance yet.")
        self._build_description(metadata)
        self._assert_unchanged()

    def _warn(self, code, reason):
        self._warnings.append({"code": code, "severity": "degraded", "reason": reason})
        _LOG.warning("reader_degraded code=%s reason=%s", code, reason)

    def _build_description(self, metadata):
        raw = self._raw
        frequency = float(raw.info["sfreq"])
        if not math.isfinite(frequency) or frequency <= 0 or raw.n_times <= 0:
            raise EphysReaderError("Invalid decoded sampling frequency or sample count")
        origin = int(raw.first_samp)
        if origin < 0:
            raise EphysReaderError("Negative source sample origins need an explicit signed-clock adapter")
        self._stream_id = self.source_id + ":samples"
        names = metadata.get("names", raw.ch_names)
        if len(names) != len(raw.ch_names):
            raise EphysReaderError("Source channel names and decoded rows do not agree")
        types = raw.get_channel_types()
        channels = []
        for index, (name, decoder_name, channel, kind) in enumerate(zip(names, raw.ch_names, raw.info["chs"], types)):
            # FIFF's unit code 107 denotes volts; get_data returns calibrated SI.
            # Other quantities retain their code and cannot enter voltage metrics.
            voltage = int(channel["unit"]) == 107 and kind != "stim"
            channels.append({"channel_id": "{}:ch:{}".format(self.source_id, index), "source_index": index,
                             "raw_name": name, "decoder_name": decoder_name,
                             "name_source": "source_header" if "names" in metadata else "decoder_metadata",
                             "channel_type": kind if self.format_name == "fif" else "unknown",
                             "decoder_channel_type": kind, "classification_status": "unreviewed",
                             "unit": "V" if voltage else "unknown", "quantity": "voltage" if voltage else "unknown",
                             "physical_unit": raw._orig_units.get(decoder_name, "") if raw._orig_units else "",
                             "decoder_unit_code": int(channel["unit"]), "unit_status": "decoder_si" if voltage else "unknown_quantity"})
        self._segments = [{"segment_id": self._stream_id + ":segment:0", "start_sample": origin,
                           "stop_sample": origin + int(raw.n_times), "recording_start_seconds": origin / frequency}]
        events = []
        for index, annotation in enumerate(raw.annotations):
            onset, duration, description = float(annotation["onset"]), float(annotation["duration"]), str(annotation["description"])
            if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                raise EphysReaderError("Non-finite or negative annotation timing")
            if description.lower() in ("boundary", "bad boundary", "edge boundary", "bad_acq_skip"):
                raise EphysReaderError("Discontinuity annotation needs an explicit source-clock adapter; no continuous timeline was inferred")
            position = onset * frequency
            sample = int(round(position)) if abs(position - round(position)) <= 1e-7 else None
            span = next((s for s in self._segments if s["start_sample"] <= position < s["stop_sample"]), None)
            mappings = [] if span is None else [{"stream_id": self._stream_id, "segment_id": span["segment_id"],
                                               "sample_position": position, "sample_index": sample,
                                               "alignment": "exact_sample" if sample is not None else "between_samples"}]
            # Decoder annotations are relative to the stored-sample clock.
            # BrainVision segment offsets can change the elapsed recording time.
            recording_onset = (span["recording_start_seconds"] + (position - span["start_sample"]) / frequency
                               if span is not None else onset)
            events.append({"event_id": "{}:annotation:{}".format(self.source_id, index), "source_id": self.source_id,
                           "clock_id": self.source_id, "source_annotation_index": index, "text": description,
                           "onset_seconds": recording_onset, "duration_seconds": duration,
                           "source_decoder_onset_seconds": onset, "sample_mappings": mappings,
                           "interpretation_status": "decoder_annotation", "decoder": "mne", "decoder_version": self._mne_version})
        self._stream = {"stream_id": self._stream_id, "source_id": self.source_id, "clock_id": self.source_id,
                        "sampling_frequency_hz": frequency, "sample_index_origin": origin,
                        "sample_count": int(raw.n_times), "channels": channels, "segments": self._segments,
                        "time_source": "source_sample_clock",
                        "reference": "as_stored", "processing_applied": [], "channel_order": "source_header"}
        self._description = {"reader_version": "mne-lazy-1", "decoder_version": self._mne_version,
                             "source_id": self.source_id, "format": self.format_name,
                             "clock": {"clock_id": self.source_id, "origin": "source_sample_clock", "timezone": "unknown",
                                       "cross_source_alignment": "unverified", "sample_index_origin": origin},
                             "streams": [self._stream], "events": events, "warnings": self._warnings,
                             "processing_applied": [], "scientific_validity": "pending",
                             "representations": ["physical"], "spatial_metadata": "outside_qc_input_contract"}

    def _assert_unchanged(self):
        for path, signature in self._signatures.items():
            if path.is_symlink() or _identity(path.stat()) != signature:
                raise EphysReaderError("Source bundle changed; reverify the recording input fingerprint")

    def describe(self):
        self._assert_unchanged()
        return deepcopy(self._description)

    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
        self._assert_unchanged()
        if representation != "physical":
            raise EphysReaderError("This optional adapter exposes calibrated physical samples only; raw ADC codes are unavailable")
        if (isinstance(start_sample, bool) or isinstance(stop_sample, bool) or
                not isinstance(start_sample, Integral) or not isinstance(stop_sample, Integral)):
            raise EphysReaderError("Sample bounds must be integers")
        segment = next((s for s in self._segments if s["segment_id"] == segment_id), None)
        if stream_id != self._stream_id or segment is None or not segment["start_sample"] <= start_sample < stop_sample <= segment["stop_sample"]:
            raise EphysReaderError("Window must be a non-empty interval inside one known stream segment")
        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or any(not isinstance(c, str) for c in channel_ids)
                or len(set(channel_ids)) != len(channel_ids)):
            raise EphysReaderError("Select distinct ordered channel IDs")
        lookup = {c["channel_id"]: i for i, c in enumerate(self._stream["channels"])}
        if any(c not in lookup for c in channel_ids):
            raise EphysReaderError("Unknown channel ID")
        if len(channel_ids) * (stop_sample - start_sample) * 8 > 256 * 1024 * 1024:
            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
        origin = self._stream["sample_index_origin"]
        with warnings.catch_warnings(record=True) as caught:
            result = self._raw.get_data(picks=[lookup[c] for c in channel_ids], start=start_sample - origin,
                                        stop=stop_sample - origin, reject_by_annotation=None, verbose="ERROR")
        if caught:
            raise EphysReaderError("Decoder warned during sample read: " + "; ".join(str(w.message) for w in caught))
        result = np.asarray(result, dtype="<f8", order="C")
        if result.shape != (len(channel_ids), stop_sample - start_sample) or not np.isfinite(result).all():
            raise EphysReaderError("Decoded window has wrong shape or non-finite values")
        self._assert_unchanged()
        return result

    def close(self):
        self._raw.close()
