"""Explicit entry points; ambiguous suffixes never choose a vendor silently."""
from pathlib import Path

from .edf import EdfBdfReader, EphysReaderError


FORMAT_CAPABILITIES = {
    "edf": {"dependency": None, "status": "real_sample_verified"},
    "bdf": {"dependency": None, "status": "real_sample_verified"},
    "brainvision": {"dependency": None, "status": "generated_fixture_verified"},
    "fif": {"dependency": "mne", "status": "generated_fixture_verified"},
    "eeglab": {"dependency": "mne", "status": "generated_fixture_verified_external_fdt_only"},
    "nicolet_data": {"dependency": "mne", "status": "generated_fixture_verified"},
    "neuroscan_cnt": {"dependency": "mne", "status": "adapter_pending_vendor_sample"},
    "nihon": {"dependency": "mne", "status": "adapter_pending_vendor_sample"},
    "nicolet_e": {"dependency": None, "status": "unsupported_pending_validation"},
}
_SUFFIXES = {".edf": "edf", ".bdf": "bdf", ".vhdr": "brainvision", ".set": "eeglab",
             ".fif": "fif", ".data": "nicolet_data", ".e": "nicolet_e"}


def identify_format(path, format_hint=None):
    path = Path(path)
    if format_hint is not None:
        if format_hint not in FORMAT_CAPABILITIES:
            raise EphysReaderError("Unknown format_hint; use one of " + ", ".join(FORMAT_CAPABILITIES))
        return format_hint
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".fif.gz"):
        return "fif"
    if suffix in (".eeg", ".cnt"):
        raise EphysReaderError("Ambiguous extension {}; specify the device format explicitly. BrainVision must use its .vhdr entry point.".format(suffix))
    if suffix not in _SUFFIXES:
        raise EphysReaderError("Unsupported input extension {}; no decoder was guessed".format(suffix))
    return _SUFFIXES[suffix]


def open_reader(path, source_id, format_hint=None):
    """Return a bounded reader; NPY derivatives need their explicit descriptor."""
    kind = identify_format(path, format_hint)
    if kind in ("edf", "bdf"):
        reader = EdfBdfReader(path, source_id)
        if not reader.describe()["format"].lower().startswith(kind):
            raise EphysReaderError("The source signature disagrees with the declared format")
        return reader
    if kind == "nicolet_e":
        raise EphysReaderError("Nicolet/Nervus .e requires a separately validated decoder; .data/.head is a different export. Retain the original .e and obtain a validated EDF/BDF export for this release.")
    if kind == "brainvision":
        from .brainvision import BrainVisionReader
        return BrainVisionReader(path, source_id)
    from .mne_raw import MneRawReader
    return MneRawReader(path, source_id, kind)


def discover_sources(root):
    """List entry points without decoding, executing files or loading samples.

    Sidecars listed here are a discovery hint. The reader validates the exact
    header references, and the storage layer fingerprints the entire bundle.
    """
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    brainvision_data = set()
    for path in files:
        if path.suffix.lower() == ".vhdr":
            from .sidecars import brainvision_metadata
            try:
                brainvision_data.add(brainvision_metadata(path)["data"])
            except (ValueError, OSError, KeyError):
                # Still list the invalid entry point so catalog construction
                # records its actual decoder error rather than hiding it.
                pass
    result = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".eeg" and path in brainvision_data:
            continue
        if suffix in (".cnt", ".eeg"):
            kind, status = "ambiguous", "requires_format_hint"
        elif suffix in _SUFFIXES or path.name.lower().endswith(".fif.gz"):
            kind = identify_format(path)
            status = ("unsupported_pending_validation" if kind == "nicolet_e" else
                      "requires_optional_dependency" if FORMAT_CAPABILITIES[kind]["dependency"] else "supported")
        else:
            continue
        sidecar_suffixes = {"brainvision": (".vmrk", ".eeg"), "eeglab": (".fdt",),
                            "nicolet_data": (".head",), "nihon": (".pnt", ".log", ".21e")}.get(kind, ())
        sidecars = [p.relative_to(root).as_posix() for p in files
                    if p.parent == path.parent and p.suffix.lower() in sidecar_suffixes]
        result.append({"relative_path": path.relative_to(root).as_posix(), "format": kind,
                       "sidecars": sidecars, "status": status})
    return result
