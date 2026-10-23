"""Bounded metadata preflight before an optional third-party decoder runs."""
from configparser import ConfigParser, Error as ConfigurationError
import io
from pathlib import Path
import struct
import re

from .edf import EphysReaderError


def local_file(parent, name):
    """A referenced file must be present beside its entry point, not outside it."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise EphysReaderError("Invalid sidecar filename")
    candidate = parent / name
    if candidate.is_symlink() or candidate.resolve().parent != parent.resolve() or not candidate.is_file():
        raise EphysReaderError("Required sidecar must be a regular file in the input directory: " + name)
    return candidate


def text_header(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise EphysReaderError("Text metadata exceeds the 16 MiB supported limit")
    data = path.read_bytes()
    # BrainVision specifies Codepage explicitly; absent means ANSI/Windows.
    if re.search(rb"(?im)^Codepage\s*=\s*UTF-8\s*$", data) or data.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "cp1252"
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise EphysReaderError("Unsupported metadata encoding; export with explicit UTF-8") from exc


def brainvision_metadata(path):
    config = ConfigParser(interpolation=None)
    text = text_header(path)
    try:
        config.read_string(text[text.index("[Common Infos]"):])
    except ConfigurationError as exc:
        raise EphysReaderError("Invalid BrainVision header configuration") from exc
    common = config["Common Infos"]
    if common.get("DataFormat", "").upper() != "BINARY":
        raise EphysReaderError("BrainVision ASCII input is not supported by the bounded binary adapter")
    data = local_file(path.parent, common["DataFile"])
    marker = local_file(path.parent, common["MarkerFile"])
    count = int(common["NumberOfChannels"])
    if not 1 <= count <= 4096:
        raise EphysReaderError("Invalid channel count")
    channel_fields = [config["Channel Infos"]["Ch{}".format(i + 1)].split(",") for i in range(count)]
    names = [fields[0].replace("\\1", ",") for fields in channel_fields]
    sizes = {"INT_16": 2, "UINT_16": 2, "IEEE_FLOAT_32": 4}
    width = sizes.get(config["Binary Infos"]["BinaryFormat"])
    if width is None or data.stat().st_size % (width * count):
        raise EphysReaderError("BrainVision payload is unsupported or ends with a partial sample")
    return {"files": [path, data, marker], "names": names, "markers": marker, "config": config,
            "data": data, "channel_fields": channel_fields,
            "expected_samples": data.stat().st_size // (width * count)}


def eeglab_metadata(path):
    """Accept bounded uncompressed MAT-v5 headers referencing external FDT.

    Embedded matrices, compressed MAT elements and MAT-v7.3 need a separate
    streaming metadata adapter. Reject before a decoder can preload a signal.
    """
    if path.stat().st_size > 16 * 1024 * 1024:
        raise EphysReaderError("EEGLAB metadata exceeds 16 MiB; use a small external-FDT header")
    payload = path.read_bytes()
    if len(payload) < 128 or payload[126:128] not in (b"IM", b"MI"):
        raise EphysReaderError("Only uncompressed MAT-v5 EEGLAB headers are supported")
    endian = "<" if payload[126:128] == b"IM" else ">"
    offset = 128
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise EphysReaderError("Truncated EEGLAB MAT element")
        kind, size = struct.unpack_from(endian + "II", payload, offset)
        if kind == 15:
            raise EphysReaderError("Compressed MAT metadata needs a bounded decoder; save an uncompressed external-FDT header")
        if kind != 14 or offset + 8 + size > len(payload):
            raise EphysReaderError("Unsupported EEGLAB MAT element layout")
        offset += 8 + ((size + 7) // 8) * 8
    from scipy.io import loadmat
    mat = loadmat(io.BytesIO(payload), squeeze_me=True, struct_as_record=False)
    eeg = mat.get("EEG")
    if eeg is None:
        raise EphysReaderError("EEGLAB header requires an EEG structure")
    if int(eeg.trials) != 1 or not isinstance(eeg.data, str):
        raise EphysReaderError("Only continuous EEGLAB with an external .fdt is supported; embedded/epoched arrays are not preloaded")
    data = local_file(path.parent, eeg.data)
    if data.suffix.lower() != ".fdt" or data.stat().st_size != int(eeg.nbchan) * int(eeg.pnts) * 4:
        raise EphysReaderError("EEGLAB FDT payload length disagrees with channel/sample counts")
    import numpy as np
    channel_metadata = np.atleast_1d(eeg.chanlocs)
    names = [str(channel.labels) for channel in channel_metadata]
    return {"files": [path, data], "names": names, "expected_samples": int(eeg.pnts)}


def nicolet_metadata(path):
    header = local_file(path.parent, path.stem + ".head")
    values = dict(line.split("=", 1) for line in text_header(header).splitlines() if line)
    names = values["elec_names"].strip("[]").split(",")
    count, samples = int(values["num_channels"]), int(values["num_samples"])
    if len(names) != count or samples <= 0 or path.stat().st_size != count * samples * 2:
        raise EphysReaderError("Nicolet .data/.head channel/sample counts disagree with exact payload length")
    return {"files": [path, header], "names": names, "expected_samples": samples}
