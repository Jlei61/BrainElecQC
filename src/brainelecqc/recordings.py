"""Stable recording identity and immutable input snapshots."""

import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from .config import settings
from .db import current_ts, get_db
from .contracts import SCHEMA_VERSION, descriptor
from .storage import (ImportConflict, SnapshotWriter, canonical_bytes,
                            confined, digest, inspect_snapshot)

logger = logging.getLogger(__name__)


def _row(recording_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM ephys_recordings WHERE recording_id=?",
                           (recording_id,)).fetchone()
        if row is None:
            raise LookupError("Recording not found")
        artifact = conn.execute("SELECT * FROM ephys_artifacts WHERE artifact_id=?",
                                (row["input_artifact_id"],)).fetchone()
    if artifact is None:
        raise ImportConflict("Recording input artifact is missing; restore its database entry")
    return row, artifact


def _artifact_root(artifact):
    return confined(settings.storage_root / artifact["storage_path"], settings.storage_root)


def _present(row, artifact, include_manifest=True):
    result = json.loads(row["descriptor_json"])
    result.update(recording_id=row["recording_id"], input_artifact_id=row["input_artifact_id"],
                  created_at=row["created_at"], schema_version=SCHEMA_VERSION,
                  status="stored_unreviewed", scientific_validity="pending",
                  metadata_status="declared_unverified", input_artifact=artifact)
    if include_manifest:
        try:
            root = _artifact_root(artifact)
            if (root / "manifest.json").is_symlink():
                raise ValueError("Manifest is a symlink")
            raw = (root / "manifest.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact["manifest_sha256"]:
                raise ValueError("Manifest fingerprint mismatch")
            result["manifest"] = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ImportConflict(f"Input manifest is unavailable or changed: {exc}; run verify") from exc
    return result


def get_recording(recording_id):
    return _present(*_row(recording_id))


def list_recordings(participant_id=None):
    with get_db() as conn:
        sql, values = "SELECT * FROM ephys_recordings", ()
        if participant_id is not None:
            sql += " WHERE participant_id=?"
            values = (participant_id,)
        rows = conn.execute(sql + " ORDER BY created_at, recording_id", values).fetchall()
        artifacts = {r["artifact_id"]: r for r in conn.execute("SELECT * FROM ephys_artifacts").fetchall()}
    return [_present(row, artifacts[row["input_artifact_id"]], False) for row in rows]


def verify_recording(recording_id):
    row, artifact = _row(recording_id)
    try:
        issues = inspect_snapshot(_artifact_root(artifact), artifact["manifest_sha256"],
                                  artifact["content_sha256"])
    except ValueError as exc:
        issues = [{"code": "unsafe_storage", "reason": str(exc)}]
    logger.log(logging.WARNING if issues else logging.INFO,
               "ephys input verification recording=%s valid=%s issues=%s", recording_id, not issues, issues)
    return {"recording_id": row["recording_id"], "input_artifact_id": artifact["artifact_id"],
            "valid": not issues, "issues": issues, "checked_at": current_ts(),
            "scientific_validity": "pending"}


def import_recording(payload, uploads=None, *, paths=None):
    """Copy explicit local paths or uploads, then publish a new immutable identity.

    Python and CLI callers supply explicit local paths or named binary streams.
    An import_key is global to this store; exact repeated bytes/metadata reuse it.
    """
    declared = descriptor(payload)
    uploads, paths = list(uploads or []), list(paths or [])
    if bool(uploads) == bool(paths):
        raise ValueError("Choose exactly one source: uploaded files or local paths")
    # Provenance records relative names, never absolute host paths.
    sources = ([{"kind": "upload", "original_filename": f.filename} for f in uploads] if uploads
               else [{"kind": "local_copy", "original_filename": Path(p).name} for p in paths])
    parent = confined(settings.inputs_root, settings.storage_root)
    parent.mkdir(parents=True, exist_ok=True)
    recording_id, artifact_id = str(uuid4()), str(uuid4())
    stage = Path(tempfile.mkdtemp(prefix=".import-", dir=parent))
    final = parent / recording_id
    published = False
    committed = False
    try:
        writer = SnapshotWriter(stage)
        for value in paths:
            path = Path(value).expanduser().absolute()
            # Only caller-selected files are read; links are refused, never followed.
            writer.add_path(path, path.parent)
        for upload in uploads:
            writer.add_upload(upload)
        entries = writer.finish()
        writer.verify_sources()
        content_sha256 = digest(entries)
        request_sha256 = digest({"descriptor": declared, "sources": sources,
                                 "content_sha256": content_sha256})
        created_at = current_ts()
        manifest = {"schema_version": SCHEMA_VERSION, "recording_id": recording_id,
                    "input_artifact_id": artifact_id, **declared, "sources": sources,
                    "metadata_status": "declared_unverified", "scientific_validity": "pending",
                    "files": entries, "content_sha256": content_sha256, "created_at": created_at}
        raw_manifest = canonical_bytes(manifest)
        (stage / "manifest.json").write_bytes(raw_manifest)
        (stage / "manifest.json").chmod(0o444)
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT recording_id, request_sha256 FROM ephys_recordings "
                               "WHERE import_key=?",
                               (payload.get("import_key"),)).fetchone()
            if old is not None:
                if old["request_sha256"] != request_sha256:
                    raise ImportConflict("import_key already identifies different input or metadata")
                if not verify_recording(old["recording_id"])["valid"]:
                    raise ImportConflict("Existing input fingerprint mismatch; restore it or use a new import_key")
                return get_recording(old["recording_id"]), True
            # Exclusive reservation: even an unexpected empty destination is never replaced.
            final.mkdir(exist_ok=False)
            published = True
            (stage / "files").rename(final / "files")
            (stage / "manifest.json").rename(final / "manifest.json")
            conn.execute("INSERT INTO ephys_recordings VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (recording_id, declared["participant_id"], declared["label"], declared["modality"],
                          declared["session_id"], artifact_id, payload.get("import_key"),
                          request_sha256, canonical_bytes(declared).decode("utf-8"), created_at))
            conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
                         (artifact_id, recording_id, "raw_input", final.relative_to(settings.storage_root.resolve()).as_posix(),
                          content_sha256, hashlib.sha256(raw_manifest).hexdigest(),
                          len(entries), writer.size, created_at))
        committed = True
        logger.info("ephys recording imported recording=%s artifact=%s files=%d sha256=%s",
                    recording_id, artifact_id, len(entries), content_sha256)
        return get_recording(recording_id), False
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if published and not committed:
            shutil.rmtree(final)
