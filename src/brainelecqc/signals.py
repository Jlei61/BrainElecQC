"""Verified identity-bound signal catalogs; no waveform arrays are cached."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
from threading import RLock

from .db import get_db
from .readers import discover_sources, open_reader
from .recordings import _artifact_root, get_recording
from .storage import ImportConflict, confined, inspect_snapshot, inventory, stamp

READER_VERSION = "brainelecqc-window-v1"
_CACHE = OrderedDict()
_LOCK = RLock()
_MAX_CATALOGS = 8
logger = logging.getLogger(__name__)


def _fingerprint(root):
    """Stat guard is accepted only after a full SHA256 verification in this process.

    Including inode/ctime detects same-length edits with restored mtime. No input
    sample arrays are cached. A changed guard always triggers full verification.
    """
    root = confined(root, root.parent)
    manifest = confined(root / "manifest.json", root)
    entries = [("manifest.json", stamp(manifest))]
    entries.extend((p.relative_to(root).as_posix(), stamp(p)) for p in inventory(root / "files"))
    entries.append(("root_entries", tuple(sorted(p.name for p in root.iterdir()))))
    return tuple(entries)


def _neuracle_evidence(root, manifest):
    evidence = []
    for item in manifest["files"]:
        relative = Path(item["relative_path"])
        if relative.name != "recordInformation.json":
            continue
        if item["size_bytes"] > 4 * 1024 * 1024:
            evidence.append({"source": relative.as_posix(), "status": "unread",
                             "reason": "metadata exceeds 4 MiB; explicit review required"})
            continue
        try:
            raw = json.loads((root / "files" / relative).read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("recordInformation must be an object")
            # Only acquisition fields, never patient demographics or clinical notes.
            info = raw.get("DataFileInformations", [])
            selected = [{key: block.get(key) for key in
                         ("FileName", "SampleRate", "BeginTimeStamp", "DataLength")}
                        for block in info if isinstance(block, dict)] if isinstance(info, list) else []
            evidence.append({"source": relative.as_posix(), "status": "declared_unverified",
                             "sample_rate": raw.get("SampleRate"), "data_files": selected,
                             "clock_alignment": "unverified",
                             "policy": "Header calibration and TAL are retained; sidecar never overrides them"})
        except (ValueError, OSError) as exc:
            evidence.append({"source": relative.as_posix(), "status": "failed", "reason": str(exc)})
    return evidence


def _build(recording, root):
    readers, streams, sources, events, errors = {}, [], [], [], []
    members = {m["relative_path"]: m for m in recording["manifest"]["files"]}
    discovered = discover_sources(root / "files")
    hints = recording["metadata"].get("format_hints", {})
    if not isinstance(hints, dict):
        raise ValueError("metadata.format_hints must map relative paths to explicit format names")
    for candidate in discovered:
        relative = candidate["relative_path"]
        member = members[relative]
        source_key = recording["input_artifact_id"] + "\0" + relative
        source_id = "file-" + hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
        path = confined(root / "files" / relative, root)
        try:
            reader = open_reader(path, source_id=source_id, format_hint=hints.get(relative))
            description = reader.describe()
            source = {"source_id": source_id, "relative_path": relative,
                      "sha256": member["sha256"], "format": description["format"],
                      "clock": description.get("clock"),
                      "warnings": description.get("warnings", []), "status": "readable"}
            sources.append(source)
            for stream in description["streams"]:
                stream = deepcopy(stream)
                stream["channel_order_policy"] = stream["channel_order"]
                stream["channel_order"] = [c["channel_id"] for c in stream["channels"]]
                stream.update(source_id=source_id, clock_id=source_id,
                              source_relative_path=relative, source_sha256=member["sha256"],
                              clock_alignment="source_local_unaligned")
                readers[stream["stream_id"]] = reader
                streams.append(stream)
            for event in description.get("events", []):
                events.append({**event, "source_id": source_id, "clock_id": source_id,
                               "clock_alignment": "source_local_unaligned"})
        except ValueError as exc:
            error = {"source_id": source_id, "relative_path": relative, "status": "failed",
                     "reason": str(exc)}
            errors.append(error)
            sources.append(error)
            logger.warning("ephys source unreadable artifact=%s source=%s reason=%s",
                           recording["input_artifact_id"], source_id, exc)
    catalog = {"schema_version": 1, "reader_version": READER_VERSION,
               "recording_id": recording["recording_id"], "artifact_id": recording["input_artifact_id"],
               "content_sha256": recording["input_artifact"]["content_sha256"],
               "manifest_sha256": recording["input_artifact"]["manifest_sha256"],
               "data_kind": "raw_samples", "processing": [], "scientific_validity": "pending",
               "streams": streams, "sources": sources, "event_count": len(events),
               "status": "readable" if streams and not errors else "partial" if streams else "unreadable",
               "errors": errors, "declared_metadata": recording["metadata"],
               "acquisition_evidence": _neuracle_evidence(root, recording["manifest"]),
               "clock_policy": "Timestamps are in each source file clock; no cross-file synchronization assumed",
               "unsupported_members": sorted(set(members) - {d["relative_path"] for d in discovered}
                                             - {s for d in discovered for s in d.get("sidecars", [])})}
    return {"catalog": catalog, "readers": readers, "events": events, "root": root}


def verified_catalog(artifact_id):
    with get_db() as conn:
        artifact = conn.execute("SELECT * FROM ephys_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if artifact is None:
        raise LookupError("Input artifact not found")
    if artifact["kind"] not in ("raw_input", "derived_voltage"):
        raise ValueError("Unsupported signal artifact kind")
    recording = get_recording(artifact["recording_id"])
    if artifact["kind"] == "raw_input" and recording["input_artifact_id"] != artifact_id:
        raise ImportConflict("Artifact and recording identity disagree; restore the database relation")
    root = _artifact_root(artifact)
    try:
        manifest, parent = None, None
        if artifact["kind"] == "derived_voltage":
            from .derived import read_manifest, build_catalog
            manifest = read_manifest(artifact, root, recording)
            parent = verified_catalog(recording["input_artifact_id"])
            if manifest["parent_content_sha256"] != parent["catalog"]["content_sha256"]:
                raise ImportConflict("Processed voltage belongs to different parent bytes")
        with _LOCK:
            signature = _fingerprint(root)
            key = (str(root), artifact_id, artifact["content_sha256"], artifact["manifest_sha256"], READER_VERSION)
            cached = _CACHE.get(key)
            if cached is not None and cached["signature"] == signature:
                if parent is not None:
                    # Identical restored parent bytes can have a new inode/ctime.
                    # Keep the freshly SHA-verified guard, not an obsolete parent entry.
                    cached["parent_entry"] = parent
                _CACHE.move_to_end(key)
                return cached
            issues = inspect_snapshot(root, artifact["manifest_sha256"], artifact["content_sha256"])
            if issues:
                raise ImportConflict(f"Input fingerprint mismatch; restore snapshot before reading: {issues}")
            logger.info("ephys catalog verified artifact=%s route=%s sha256=%s", artifact_id,
                        "cold_sha256_verification" if cached is None else "changed_stat_reverification",
                        artifact["content_sha256"])
            entry = (_build(recording, root) if parent is None else
                     build_catalog(artifact, root, recording, manifest, parent, READER_VERSION))
            if _fingerprint(root) != signature:
                raise ImportConflict("Input changed while building the signal catalog; retry after restoring it")
            entry["signature"] = signature
            _CACHE[key] = entry
            _CACHE.move_to_end(key)
            while len(_CACHE) > _MAX_CATALOGS:
                _CACHE.popitem(last=False)
            return entry
    except (OSError, ValueError) as exc:
        if isinstance(exc, ImportConflict):
            raise
        raise ImportConflict(f"Cannot verify input snapshot: {exc}") from exc


def assert_unchanged(entry):
    try:
        if _fingerprint(entry["root"]) != entry["signature"]:
            raise ImportConflict("Input changed during window read; result discarded")
        if "parent_entry" in entry:
            assert_unchanged(entry["parent_entry"])
    except (OSError, ValueError) as exc:
        raise ImportConflict(f"Input changed during window read; result discarded: {exc}") from exc


def describe_artifact(artifact_id):
    return deepcopy(verified_catalog(artifact_id)["catalog"])


def list_events(artifact_id, offset=0, limit=100):
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 1000:
        raise ValueError("Events need offset>=0 and 1<=limit<=1000")
    entry = verified_catalog(artifact_id)
    result = {"artifact_id": artifact_id, "recording_id": entry["catalog"]["recording_id"],
              "content_sha256": entry["catalog"]["content_sha256"], "offset": offset, "limit": limit,
              "total": len(entry["events"]), "events": deepcopy(entry["events"][offset:offset + limit]),
              "event_scope": entry["catalog"].get("event_scope", "input_original_samples")}
    for key in ("parent_artifact_id", "parent_content_sha256"):
        if key in entry["catalog"]:
            result[key] = entry["catalog"][key]
    return result
