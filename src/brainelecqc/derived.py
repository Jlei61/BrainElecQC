"""Publish processed voltage with its parent samples, clocks and parameters.

Workers provide output files and processing provenance through Python calls.
Artifacts retain the original channel and segment mapping for later review.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from .config import settings
from .db import current_ts, get_db
from .readers.npy import NpyVoltageReader
from .recordings import get_recording
from .storage import SnapshotWriter, ImportConflict, canonical_bytes, confined, digest, hash_file, stamp
from .derived_contracts import normalize_stream, validate_provenance


def publish_voltage(parent_artifact_id, outputs, processing, software_versions):
    """Copy worker-owned NPYs to a new immutable artifact; never replace earlier output.

    outputs = [{path: internal worker path, stream: explicit clock/channel descriptor}].
    Values are validated in bounded windows before publication, with no whole-array load.
    """
    from .signals import verified_catalog, assert_unchanged
    parent = verified_catalog(parent_artifact_id)
    if parent["catalog"]["data_kind"] != "raw_samples":
        raise ValueError("Derived v1 binds to the raw input; record all processing steps together")
    processing, software_versions = validate_provenance(processing, software_versions)
    if not isinstance(outputs, list) or not outputs or len(outputs) > 512:
        raise ValueError("Provide 1..512 explicit processed streams")
    recording = get_recording(parent["catalog"]["recording_id"])
    artifact_id = str(uuid4())
    streams = [normalize_stream(o["stream"], parent["catalog"], artifact_id, i) for i, o in enumerate(outputs)]
    if len({s["stream_key"] for s in streams}) != len(streams):
        raise ValueError("Derived stream keys must be unique within an artifact")
    folder = confined(settings.derivatives_root /
                      recording["recording_id"] / "artifacts", settings.storage_root)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".publish-", dir=folder))
    final, published, committed = folder / artifact_id, False, False
    try:
        writer = SnapshotWriter(stage)
        for output, stream in zip(outputs, streams):
            source = Path(output["path"])
            if source.is_symlink():
                raise ValueError("Derived source cannot be a symlink")
            before = stamp(source)
            fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                member = writer.add_stream(stream["source_relative_path"], handle)
            if stamp(source) != before or hash_file(source) != {k: member[k] for k in ("size_bytes", "sha256")}:
                raise ImportConflict("Processed source changed while copying; retry the completed worker output")
            stream["source_sha256"] = member["sha256"]
            reader = NpyVoltageReader(stage / "files" / member["relative_path"], stream)
            # Validation is O(N) once at publication, but bounded in memory.
            step = max(1, 1024 * 1024 // (8 * len(stream["channels"])))
            for segment in stream["segments"]:
                for start in range(segment["start_sample"], segment["stop_sample"], step):
                    reader.read_window(stream["stream_id"], segment["segment_id"], stream["channel_order"],
                                       start, min(start + step, segment["stop_sample"]), representation="physical")
        entries = writer.finish()
        created_at = current_ts()
        manifest = {"schema_version": 1, "kind": "derived_voltage", "artifact_id": artifact_id,
                    "recording_id": recording["recording_id"], "parent_artifact_id": parent_artifact_id,
                    "parent_content_sha256": parent["catalog"]["content_sha256"], "streams": streams,
                    "processing": processing, "software_versions": software_versions,
                    "files": entries, "content_sha256": digest(entries), "created_at": created_at,
                    "scientific_validity": "pending"}
        raw_manifest = canonical_bytes(manifest)
        (stage / "manifest.json").write_bytes(raw_manifest)
        (stage / "manifest.json").chmod(0o444)
        assert_unchanged(parent)
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT content_sha256 FROM ephys_artifacts WHERE artifact_id=?",
                                   (parent_artifact_id,)).fetchone()
            if current is None or current["content_sha256"] != manifest["parent_content_sha256"]:
                raise ImportConflict("Parent identity changed during processed publication")
            final.mkdir(exist_ok=False)
            published = True
            (stage / "files").rename(final / "files")
            (stage / "manifest.json").rename(final / "manifest.json")
            storage_path = final.relative_to(settings.storage_root.resolve()).as_posix()
            conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
                         (artifact_id, recording["recording_id"], "derived_voltage", storage_path,
                          manifest["content_sha256"], hashlib.sha256(raw_manifest).hexdigest(),
                          len(entries), writer.size, created_at))
        committed = True
        return {"artifact_id": artifact_id, "recording_id": recording["recording_id"],
                "storage_path": storage_path, "content_sha256": manifest["content_sha256"],
                "scientific_validity": "pending"}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if published and not committed:
            shutil.rmtree(final)


def read_manifest(artifact, root, recording):
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["manifest_sha256"]:
        raise ImportConflict("Processed manifest fingerprint mismatch")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "derived_voltage"
            or manifest.get("artifact_id") != artifact["artifact_id"]
            or manifest.get("recording_id") != recording["recording_id"]
            or manifest.get("parent_artifact_id") != recording["input_artifact_id"]):
        raise ImportConflict("Processed artifact/recording/parent identities disagree")
    return manifest


def build_catalog(artifact, root, recording, manifest, parent, reader_version):
    if manifest["parent_content_sha256"] != parent["catalog"]["content_sha256"]:
        raise ImportConflict("Processed artifact was built from different parent bytes")
    processing, versions = validate_provenance(manifest["processing"], manifest["software_versions"])
    readers, streams, sources = {}, [], []
    members = {m["relative_path"]: m for m in manifest["files"]}
    for i, declared in enumerate(manifest["streams"]):
        stream = normalize_stream(declared, parent["catalog"], artifact["artifact_id"], i)
        member = members[stream["source_relative_path"]]
        stream["source_sha256"] = member["sha256"]
        if stream != declared:
            raise ImportConflict("Processed stream metadata disagrees with its parent mapping")
        readers[stream["stream_id"]] = NpyVoltageReader(confined(root / "files" / member["relative_path"], root), stream)
        streams.append(stream)
        sources.append({"source_id": stream["source_id"], "relative_path": stream["source_relative_path"],
                        "sha256": member["sha256"], "format": "npy_voltage_v1", "status": "readable"})
    catalog = {"schema_version": 1, "reader_version": reader_version, "recording_id": recording["recording_id"],
               "artifact_id": artifact["artifact_id"], "content_sha256": artifact["content_sha256"],
               "manifest_sha256": artifact["manifest_sha256"], "data_kind": "derived_samples",
               "processing": processing, "software_versions": versions, "scientific_validity": "pending",
               "parent_artifact_id": manifest["parent_artifact_id"], "parent_content_sha256": manifest["parent_content_sha256"],
               "streams": streams, "sources": sources, "status": "readable", "errors": [],
               "event_count": len(parent["events"]), "event_scope": "parent_input_original_samples",
               "parent_sources": deepcopy(parent["catalog"]["sources"]),
               "clock_policy": "Derived segments retain verified parent clocks; no cross-file synchronization"}
    return {"catalog": catalog, "readers": readers, "events": deepcopy(parent["events"]), "root": root,
            "parent_entry": parent}
