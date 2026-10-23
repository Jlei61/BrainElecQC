"""Small byte-authored EDF/BDF fixtures; no scientific reader writes the oracle."""
from pathlib import Path

import numpy as np


def write_edf_bdf(path, channels, *, bdf=False, reserved="", duration="1",
                  annotation_records=None, record_count=None):
    """channels: dictionaries with label, records (digital lists), optional calibration."""
    channels = [dict(channel) for channel in channels]
    width = 3 if bdf else 2
    if annotation_records is not None:
        size = max(128, max(map(len, annotation_records)))
        samples = (size + width - 1) // width
        channels.append({"label": ("BDF" if bdf else "EDF") + " Annotations",
                         "records": annotation_records, "annotation": True,
                         "samples": samples, "unit": ""})
    n_records = len(channels[0]["records"])
    n_signals = len(channels)
    limit = 2 ** (width * 8 - 1)

    def field(value, size):
        raw = value if isinstance(value, bytes) else str(value).encode("ascii")
        assert len(raw) <= size
        return raw.ljust(size, b" ")

    header = (b"\xffBIOSEMI" if bdf else b"0       ")
    header += field("X", 80) + field("Startdate X X X X", 80)
    header += b"01.01.2412.00.00" + field((n_signals + 1) * 256, 8)
    header += field(reserved, 44) + field(n_records if record_count is None else record_count, 8)
    header += field(duration, 8) + field(n_signals, 4)
    fields = [("label", 16, ""), ("transducer", 80, ""), ("unit", 8, "uV"),
              ("physical_min", 8, -limit), ("physical_max", 8, limit - 1),
              ("digital_min", 8, -limit), ("digital_max", 8, limit - 1),
              ("prefilter", 80, ""), ("samples", 8, None), ("reserved", 32, "")]
    for key, size, default in fields:
        for channel in channels:
            value = channel.get(key, default)
            if key == "samples" and value is None:
                value = len(channel["records"][0])
            header += field(value, size)
    payload = bytearray()
    for record in range(n_records):
        for channel in channels:
            values = channel["records"][record]
            if channel.get("annotation"):
                payload.extend(values.ljust(channel["samples"] * width, b"\0"))
            elif bdf:
                for value in values:
                    payload.extend((int(value) & 0xffffff).to_bytes(3, "little"))
            else:
                payload.extend(np.asarray(values, dtype="<i2").tobytes())
    Path(path).write_bytes(header + payload)
    return Path(path)


def tal(onset, text=None, duration=None):
    onset = str(onset)
    if not onset.startswith(("+", "-")):
        onset = "+" + onset
    stamp = onset.encode("ascii")
    if duration is not None:
        stamp += b"\x15" + str(duration).encode("ascii")
    text = b"" if text is None else (text if isinstance(text, bytes) else text.encode("utf-8"))
    return stamp + b"\x14" + text + b"\x14\x00"
