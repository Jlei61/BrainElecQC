# Cortrix SEEG QC 接入

这份接入补丁保留 Cortrix 的 SEEG QC 浏览器界面、Flask 接口、任务、数据表和契约测试。BrainElecQC 的独立入口见 [SEEG QC 使用说明](seeg-qc.md)。

补丁基于 Cortrix 提交 15d250579caac3abb0d07f6e54807cc99070d065。在这个提交对应的干净工作区提取补丁并检查。

```bash
python3 - /path/to/BrainElecQC/docs/cortrix-seeg-qc.md <<'PY' | git apply --check
from pathlib import Path
import sys

document = Path(sys.argv[1]).read_text(encoding="utf-8")
patch = document.split("\n````diff\n", 1)[1].split("\n````", 1)[0]
sys.stdout.write(patch + "\n")
PY
```

检查通过后，把命令末尾的 git apply --check 改为 git apply。代码沿用 [Apache 2.0 许可](licenses/Apache-2.0.md)。

````diff
diff --git a/.gitignore b/.gitignore
index 2a1fd388a0f07c4f779d5c89f13ae21d7836f764..04657ffbae0a89f9ec4ce3673e162f4c42b1c5c1 100644
--- a/.gitignore
+++ b/.gitignore
@@ -341 +341,19 @@ backend/tests/**
 !backend/tests/contract/test_backend_storage_isolation_guard.py
+
+# QC S0/S1 fixtures and immutable-input regression contracts.
+!backend/tests/ephys_fixtures.py
+!backend/tests/contract/test_ephys_fixture_baseline.py
+!backend/tests/contract/test_ephys_recordings.py
+!backend/tests/contract/test_ephys_migration.py
+
+# QC S2 format truth, binary window and timeline contracts.
+!backend/tests/ephys_edf_fixtures.py
+!backend/tests/contract/test_ephys_edf_reader.py
+!backend/tests/contract/test_ephys_samples_api.py
+!backend/tests/contract/test_ephys_npy_reader.py
+!backend/tests/contract/test_ephys_derived.py
+
+# SEEG basic QC evidence, review and frozen-export contracts.
+!backend/tests/contract/test_seeg_qc_metrics.py
+!backend/tests/contract/test_seeg_qc_waveform.py
+!backend/tests/contract/test_ephys_qc_review.py
diff --git a/backend/app/__init__.py b/backend/app/__init__.py
index 61ca2ff80c3e6ddebac895049635775f58b359df..c8e39a369b5abeb3a949615a883c085aa0e69636 100644
--- a/backend/app/__init__.py
+++ b/backend/app/__init__.py
@@ -100,2 +100,5 @@ def create_app() -> Flask:
     from .api.v1.source import v1_source_bp
+    from .api.v1.ephys import ephys_bp
+    from .api.v1.ephys_samples import ephys_samples_bp
+    from .api.v1.ephys_qc import ephys_qc_bp
     from .api.upload_sessions import upload_sessions_bp
@@ -125,2 +128,5 @@ def create_app() -> Flask:
     app.register_blueprint(v1_source_bp)
+    app.register_blueprint(ephys_bp)
+    app.register_blueprint(ephys_samples_bp)
+    app.register_blueprint(ephys_qc_bp)
     app.register_blueprint(upload_sessions_bp)
diff --git a/backend/app/api/v1/ephys.py b/backend/app/api/v1/ephys.py
new file mode 100644
index 0000000000000000000000000000000000000000..a1832b7a55670e31f1b883235d2e42235d080bfc
--- /dev/null
+++ b/backend/app/api/v1/ephys.py
@@ -0,0 +1,66 @@
+"""Independent S1 import/control plane. Does not run or replace legacy processing."""
+
+import json
+import logging
+from flask import Blueprint, request
+from werkzeug.exceptions import BadRequest
+
+from ..response import fail, ok
+from ...domain import ephys_recordings as recordings
+from ...domain.ephys_storage import ImportConflict
+
+ephys_bp = Blueprint("v1_ephys", __name__, url_prefix="/api/v1/ephys")
+logger = logging.getLogger(__name__)
+
+
+@ephys_bp.errorhandler(ImportConflict)
+def conflict(error):
+    return fail("EPHYS_IMPORT_CONFLICT", str(error), status=409)
+
+
+@ephys_bp.errorhandler(ValueError)
+@ephys_bp.errorhandler(BadRequest)
+def invalid(error):
+    return fail("EPHYS_INVALID_INPUT", str(error), status=400)
+
+
+@ephys_bp.errorhandler(LookupError)
+def missing(error):
+    return fail("EPHYS_NOT_FOUND", str(error), status=404)
+
+
+@ephys_bp.errorhandler(OSError)
+def storage_error(error):
+    logger.exception("QC input storage failed")
+    return fail("EPHYS_STORAGE_ERROR", "Input storage failed; inspect server logs and retry", status=500)
+
+
+@ephys_bp.post("/recordings")
+def create_recording():
+    if request.mimetype == "multipart/form-data":
+        payload = json.loads(request.form.get("metadata", "{}"))
+        uploads = request.files.getlist("files")
+    else:
+        if not request.is_json:
+            raise ValueError("Use application/json or multipart/form-data")
+        payload, uploads = request.get_json(), None
+    result, reused = recordings.import_recording(payload, uploads)
+    return ok(result, {"reused": reused}), 200 if reused else 201
+
+
+@ephys_bp.get("/recordings")
+def list_recordings():
+    subject_id = request.args.get("subject_id")
+    if not subject_id:
+        raise ValueError("subject_id is required")
+    return ok(recordings.list_recordings(subject_id))
+
+
+@ephys_bp.get("/recordings/<recording_id>")
+def get_recording(recording_id):
+    return ok(recordings.get_recording(recording_id))
+
+
+@ephys_bp.post("/recordings/<recording_id>/verify")
+def verify_recording(recording_id):
+    return ok(recordings.verify_recording(recording_id))
diff --git a/backend/app/api/v1/ephys_qc.py b/backend/app/api/v1/ephys_qc.py
new file mode 100644
index 0000000000000000000000000000000000000000..c42665dd2dde3073f0e9b22576ca30fa65ee79d0
--- /dev/null
+++ b/backend/app/api/v1/ephys_qc.py
@@ -0,0 +1,89 @@
+"""Basic SEEG QC controls; evidence samples remain on the binary window API."""
+import logging
+
+from flask import Blueprint, Response, current_app, request
+from werkzeug.exceptions import BadRequest
+
+from ..response import ok, fail
+from ...domain import ephys_qc as qc
+from ...domain.ephys_recordings import get_recording
+from ...domain.ephys_signals import describe_artifact
+from ...domain.ephys_storage import ImportConflict
+
+ephys_qc_bp = Blueprint("v1_ephys_qc", __name__, url_prefix="/api/v1/ephys")
+logger = logging.getLogger(__name__)
+
+
+@ephys_qc_bp.errorhandler(ImportConflict)
+def conflict(error):
+    return fail("EPHYS_QC_CONFLICT", str(error), status=409)
+
+
+@ephys_qc_bp.errorhandler(ValueError)
+@ephys_qc_bp.errorhandler(BadRequest)
+def invalid(error):
+    return fail("EPHYS_QC_INVALID", str(error), status=400)
+
+
+@ephys_qc_bp.errorhandler(LookupError)
+def missing(error):
+    return fail("EPHYS_QC_NOT_FOUND", str(error), status=404)
+
+
+@ephys_qc_bp.errorhandler(OSError)
+def inaccessible(error):
+    logger.exception("QC storage unavailable")
+    return fail("EPHYS_QC_STORAGE_ERROR", "QC storage unavailable; inspect server logs", status=500)
+
+
+def _payload():
+    if not request.is_json or not isinstance(request.get_json(), dict):
+        raise ValueError("Use a JSON object")
+    return request.get_json()
+
+
+@ephys_qc_bp.get("/recordings/<recording_id>/qc")
+def history_route(recording_id):
+    return ok(qc.history(recording_id))
+
+
+@ephys_qc_bp.get("/recordings/<recording_id>/qc/channel-map")
+def map_route(recording_id):
+    from ...processing.ephys_qc.seeg_channels import resolve_seeg_channels
+    record = get_recording(recording_id)
+    catalog = describe_artifact(record["input_artifact_id"])
+    return ok({"channel_map": resolve_seeg_channels(catalog)})
+
+
+@ephys_qc_bp.post("/recordings/<recording_id>/qc/runs")
+def create_route(recording_id):
+    runner = getattr(current_app, "job_runner", None)
+    if runner is None:
+        return fail("EPHYS_QC_WORKER_UNAVAILABLE", "Job runner unavailable; restart the backend", status=503)
+    result = qc.create_run(recording_id, _payload())
+    runner.enqueue(result["job_id"])
+    return ok(result), 202
+
+
+@ephys_qc_bp.get("/qc/runs/<run_id>")
+def run_route(run_id):
+    return ok(qc.get_run(run_id))
+
+
+@ephys_qc_bp.patch("/qc/runs/<run_id>/review")
+def review_route(run_id):
+    return ok(qc.save_review(run_id, _payload()))
+
+
+@ephys_qc_bp.post("/qc/runs/<run_id>/freeze")
+def freeze_route(run_id):
+    return ok(qc.freeze_run(run_id, _payload()))
+
+
+@ephys_qc_bp.get("/qc/runs/<run_id>/export")
+def export_route(run_id):
+    response = Response(qc.export_run(run_id), content_type="application/zip")
+    response.headers["Content-Disposition"] = f'attachment; filename="seeg-qc-{run_id}.zip"'
+    response.headers["Cache-Control"] = "no-store"
+    response.headers["X-Content-Type-Options"] = "nosniff"
+    return response
diff --git a/backend/app/api/v1/ephys_samples.py b/backend/app/api/v1/ephys_samples.py
new file mode 100644
index 0000000000000000000000000000000000000000..3a9477ec730eac200c97b99883740d54bfc289b6
--- /dev/null
+++ b/backend/app/api/v1/ephys_samples.py
@@ -0,0 +1,77 @@
+"""S2 control metadata + framed binary samples. Never reuse legacy envelopes."""
+
+import logging
+from flask import Blueprint, Response, request
+
+from ..response import ok, fail
+from ...domain.ephys_signals import describe_artifact, list_events
+from ...domain.ephys_windows import samples, overview, encode_window
+from ...domain.ephys_storage import ImportConflict
+
+ephys_samples_bp = Blueprint("v1_ephys_samples", __name__, url_prefix="/api/v1/ephys")
+logger = logging.getLogger(__name__)
+
+
+@ephys_samples_bp.errorhandler(ImportConflict)
+def conflict(error):
+    return fail("EPHYS_INPUT_CHANGED", str(error), status=409)
+
+
+@ephys_samples_bp.errorhandler(ValueError)
+def invalid(error):
+    return fail("EPHYS_INVALID_WINDOW", str(error), status=400)
+
+
+@ephys_samples_bp.errorhandler(LookupError)
+def missing(error):
+    return fail("EPHYS_NOT_FOUND", str(error), status=404)
+
+
+@ephys_samples_bp.errorhandler(OSError)
+def inaccessible(error):
+    logger.exception("Signal input unavailable")
+    return fail("EPHYS_READ_FAILED", "Signal input unavailable; inspect logs", status=500)
+
+
+def _integer(name, default=None):
+    value = request.args.get(name, default)
+    if value is None or isinstance(value, str) and (not value.isascii() or not value.isdecimal()):
+        raise ValueError(f"{name} must be an explicit nonnegative integer")
+    return int(value)
+
+
+def _parameters():
+    for key in ("stream_id", "segment_id", "representation"):
+        if not request.args.get(key):
+            raise ValueError(f"{key} is required")
+    return {"stream_id": request.args["stream_id"], "segment_id": request.args["segment_id"],
+            "channel_ids": request.args.getlist("channel_id"), "start_sample": _integer("start_sample"),
+            "stop_sample": _integer("stop_sample"), "representation": request.args["representation"]}
+
+
+def _binary(result):
+    metadata, array = result
+    response = Response(encode_window(metadata, array), content_type="application/octet-stream")
+    response.headers["Cache-Control"] = "no-store"
+    response.headers["X-Content-Type-Options"] = "nosniff"
+    return response
+
+
+@ephys_samples_bp.get("/artifacts/<artifact_id>/signals")
+def signals_route(artifact_id):
+    return ok(describe_artifact(artifact_id))
+
+
+@ephys_samples_bp.get("/artifacts/<artifact_id>/events")
+def events_route(artifact_id):
+    return ok(list_events(artifact_id, _integer("offset", 0), _integer("limit", 100)))
+
+
+@ephys_samples_bp.get("/artifacts/<artifact_id>/samples")
+def samples_route(artifact_id):
+    return _binary(samples(artifact_id, **_parameters()))
+
+
+@ephys_samples_bp.get("/artifacts/<artifact_id>/overview")
+def overview_route(artifact_id):
+    return _binary(overview(artifact_id, **_parameters(), bin_samples=_integer("bin_samples")))
diff --git a/backend/app/db.py b/backend/app/db.py
index 66cea765ed2fc2ec25d625d88a0f4342571b3d47..45bde123f97a392bf174b6794f8da69ad5a62052 100644
--- a/backend/app/db.py
+++ b/backend/app/db.py
@@ -8,3 +8,3 @@ from typing import Any, Dict, Iterable, List, Optional
 from .config import settings
-from .migrations import seed_anchors_and_seeg_contacts_v2
+from .migrations import ephys_artifacts_v2, ephys_recordings_v1, ephys_qc_v1, seed_anchors_and_seeg_contacts_v2

@@ -70,2 +70,5 @@ def _run_migrations():
         seed_anchors_and_seeg_contacts_v2.upgrade(conn)
+        ephys_recordings_v1.upgrade(conn)
+        ephys_artifacts_v2.upgrade(conn)
+        ephys_qc_v1.upgrade(conn)

diff --git a/backend/app/domain/assets.py b/backend/app/domain/assets.py
index eeeb06ae38c0163c91cb1464915d50322b351b28..e372e798282aa0d7c6e3f8af0af9f041e1d115be 100644
--- a/backend/app/domain/assets.py
+++ b/backend/app/domain/assets.py
@@ -184,2 +184,4 @@ def _resolve_relative_dir(asset_type: str) -> Path:
     safe = secure_filename(asset_type or "custom")
+    if safe.lower() == "ephys-recordings":
+        raise AssetError("ephys-recordings is reserved for immutable QC inputs")
     return Path(safe or "custom")
@@ -218,2 +220,3 @@ def reset_asset_group(subject_id: str, asset_type: str):
     normalized = (asset_type or "").lower()
+    _resolve_relative_dir(normalized)  # Reject reserved namespaces before any DB deletion.
     _reset_asset_records(subject_id, normalized)
@@ -448,2 +451,4 @@ def register_asset(

+    _resolve_relative_dir((asset_type or "").strip().lower())
+
     staging_path = _save_to_staging(file_storage)
diff --git a/backend/app/domain/ephys_contracts.py b/backend/app/domain/ephys_contracts.py
new file mode 100644
index 0000000000000000000000000000000000000000..45e308239c4e1a0fef8f1279f90f03f8b22387c6
--- /dev/null
+++ b/backend/app/domain/ephys_contracts.py
@@ -0,0 +1,57 @@
+"""S0 ingestion contract. Declared acquisition metadata is not a QC verdict."""
+
+import json
+import math
+
+SCHEMA_VERSION = 1
+METRIC_STATES = ("computed", "insufficient_data", "disabled", "failed", "not_applicable")
+SCIENTIFIC_VALIDITY = ("pending", "valid", "degraded", "blocking")
+MODALITIES = ("eeg", "seeg", "ecog", "neuropixels", "mixed", "unknown")
+
+
+def descriptor(payload):
+    if not isinstance(payload, dict):
+        raise ValueError("Import body must be a JSON object")
+    allowed = {"subject_id", "label", "modality", "session_id", "metadata",
+               "asset_ids", "legacy_eeg_id", "import_key"}
+    if set(payload) - allowed:
+        raise ValueError(f"Unknown import fields: {sorted(set(payload) - allowed)}")
+    subject_id = payload.get("subject_id")
+    if (not isinstance(subject_id, str) or not subject_id or
+            subject_id in (".", "..") or any(c in subject_id for c in "/\\\0:")):
+        raise ValueError("subject_id must be a single safe path component")
+    label = payload.get("label")
+    if not isinstance(label, str) or not label.strip() or len(label) > 512:
+        raise ValueError("A nonempty label of at most 512 characters is required")
+    modality = payload.get("modality", "unknown")
+    if modality not in MODALITIES:
+        raise ValueError(f"modality must be one of {MODALITIES}")
+    for key in ("session_id", "import_key"):
+        value = payload.get(key)
+        if value is not None and (not isinstance(value, str) or not value or len(value) > 256):
+            raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
+    metadata = payload.get("metadata", {})
+    if not isinstance(metadata, dict):
+        raise ValueError("metadata must be an object; unknown units/timing must remain null")
+    # Reject NaN/Infinity before storage; keep declarations and provenance verbatim.
+    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
+    if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
+        raise ValueError("metadata exceeds the 4 MiB limit")
+    channels = metadata.get("channels", [])
+    if not isinstance(channels, list) or any(not isinstance(c, dict) for c in channels):
+        raise ValueError("metadata.channels must be a list of objects")
+    ids = [c.get("channel_id") for c in channels]
+    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
+        raise ValueError("Each declared channel needs a unique channel_id, independent of raw_name")
+    streams = metadata.get("streams", [])
+    if not isinstance(streams, list):
+        raise ValueError("metadata.streams must be a list")
+    for stream in streams:
+        if not isinstance(stream, dict):
+            raise ValueError("metadata.streams entries must be objects")
+        rate = stream.get("sample_rate_hz")
+        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
+                                 or not math.isfinite(rate) or rate <= 0):
+            raise ValueError("sample_rate_hz must be positive or null when unknown")
+    return {"subject_id": subject_id, "label": label, "modality": modality,
+            "session_id": payload.get("session_id"), "metadata": json.loads(encoded)}
diff --git a/backend/app/domain/ephys_derived.py b/backend/app/domain/ephys_derived.py
new file mode 100644
index 0000000000000000000000000000000000000000..26523af2cc112276b8b51cdc6f24c906dc63390f
--- /dev/null
+++ b/backend/app/domain/ephys_derived.py
@@ -0,0 +1,143 @@
+"""Internal processed-voltage publication. No API accepts arbitrary server paths.
+
+S3 executors can publish verified outputs here; publication alone is never QC approval.
+The first format preserves parent sample clocks (no resampling or implicit gap filling).
+"""
+from copy import deepcopy
+import hashlib
+import json
+import os
+from pathlib import Path
+import shutil
+import tempfile
+from uuid import uuid4
+
+from ..config import settings
+from ..db import current_ts, get_db
+from ..processing.ephys_qc.readers.npy import NpyVoltageReader
+from .ephys_recordings import get_recording
+from .ephys_storage import SnapshotWriter, ImportConflict, canonical_bytes, confined, digest, hash_file, stamp
+from .ephys_derived_contracts import normalize_stream, validate_provenance
+
+
+def publish_voltage(parent_artifact_id, outputs, processing, software_versions):
+    """Copy worker-owned NPYs to a new immutable artifact; never replace earlier output.
+
+    outputs = [{path: internal worker path, stream: explicit clock/channel descriptor}].
+    Values are validated in bounded windows before publication, with no whole-array load.
+    """
+    from .ephys_signals import verified_catalog, assert_unchanged
+    parent = verified_catalog(parent_artifact_id)
+    if parent["catalog"]["data_kind"] != "raw_samples":
+        raise ValueError("Derived v1 binds to the raw input; record all processing steps together")
+    processing, software_versions = validate_provenance(processing, software_versions)
+    if not isinstance(outputs, list) or not outputs or len(outputs) > 512:
+        raise ValueError("Provide 1..512 explicit processed streams")
+    recording = get_recording(parent["catalog"]["recording_id"])
+    artifact_id = str(uuid4())
+    streams = [normalize_stream(o["stream"], parent["catalog"], artifact_id, i) for i, o in enumerate(outputs)]
+    if len({s["stream_key"] for s in streams}) != len(streams):
+        raise ValueError("Derived stream keys must be unique within an artifact")
+    folder = confined(settings.output_root / recording["subject_id"] / "electrophysiology" / "qc" /
+                      recording["recording_id"] / "artifacts", settings.dataset_root)
+    folder.mkdir(parents=True, exist_ok=True)
+    stage = Path(tempfile.mkdtemp(prefix=".publish-", dir=folder))
+    final, published, committed = folder / artifact_id, False, False
+    try:
+        writer = SnapshotWriter(stage)
+        for output, stream in zip(outputs, streams):
+            source = Path(output["path"])
+            if source.is_symlink():
+                raise ValueError("Derived source cannot be a symlink")
+            before = stamp(source)
+            fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
+            with os.fdopen(fd, "rb") as handle:
+                member = writer.add_stream(stream["source_relative_path"], handle)
+            if stamp(source) != before or hash_file(source) != {k: member[k] for k in ("size_bytes", "sha256")}:
+                raise ImportConflict("Processed source changed while copying; retry the completed worker output")
+            stream["source_sha256"] = member["sha256"]
+            reader = NpyVoltageReader(stage / "files" / member["relative_path"], stream)
+            # Validation is O(N) once at publication, but bounded in memory.
+            step = max(1, 1024 * 1024 // (8 * len(stream["channels"])))
+            for segment in stream["segments"]:
+                for start in range(segment["start_sample"], segment["stop_sample"], step):
+                    reader.read_window(stream["stream_id"], segment["segment_id"], stream["channel_order"],
+                                       start, min(start + step, segment["stop_sample"]), representation="physical")
+        entries = writer.finish()
+        created_at = current_ts()
+        manifest = {"schema_version": 1, "kind": "derived_voltage", "artifact_id": artifact_id,
+                    "recording_id": recording["recording_id"], "parent_artifact_id": parent_artifact_id,
+                    "parent_content_sha256": parent["catalog"]["content_sha256"], "streams": streams,
+                    "processing": processing, "software_versions": software_versions,
+                    "files": entries, "content_sha256": digest(entries), "created_at": created_at,
+                    "scientific_validity": "pending"}
+        raw_manifest = canonical_bytes(manifest)
+        (stage / "manifest.json").write_bytes(raw_manifest)
+        (stage / "manifest.json").chmod(0o444)
+        assert_unchanged(parent)
+        with get_db() as conn:
+            conn.execute("BEGIN IMMEDIATE")
+            current = conn.execute("SELECT content_sha256 FROM ephys_artifacts WHERE artifact_id=?",
+                                   (parent_artifact_id,)).fetchone()
+            if current is None or current["content_sha256"] != manifest["parent_content_sha256"]:
+                raise ImportConflict("Parent identity changed during processed publication")
+            final.mkdir(exist_ok=False)
+            published = True
+            (stage / "files").rename(final / "files")
+            (stage / "manifest.json").rename(final / "manifest.json")
+            storage_path = final.relative_to(settings.dataset_root.resolve()).as_posix()
+            conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                         (artifact_id, recording["recording_id"], "derived_voltage", storage_path,
+                          manifest["content_sha256"], hashlib.sha256(raw_manifest).hexdigest(),
+                          len(entries), writer.size, created_at))
+        committed = True
+        return {"artifact_id": artifact_id, "recording_id": recording["recording_id"],
+                "storage_path": storage_path, "content_sha256": manifest["content_sha256"],
+                "scientific_validity": "pending"}
+    finally:
+        if stage.exists():
+            shutil.rmtree(stage)
+        if published and not committed:
+            shutil.rmtree(final)
+
+
+def read_manifest(artifact, root, recording):
+    raw = (root / "manifest.json").read_bytes()
+    if hashlib.sha256(raw).hexdigest() != artifact["manifest_sha256"]:
+        raise ImportConflict("Processed manifest fingerprint mismatch")
+    manifest = json.loads(raw)
+    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "derived_voltage"
+            or manifest.get("artifact_id") != artifact["artifact_id"]
+            or manifest.get("recording_id") != recording["recording_id"]
+            or manifest.get("parent_artifact_id") != recording["input_artifact_id"]):
+        raise ImportConflict("Processed artifact/recording/parent identities disagree")
+    return manifest
+
+
+def build_catalog(artifact, root, recording, manifest, parent, reader_version):
+    if manifest["parent_content_sha256"] != parent["catalog"]["content_sha256"]:
+        raise ImportConflict("Processed artifact was built from different parent bytes")
+    processing, versions = validate_provenance(manifest["processing"], manifest["software_versions"])
+    readers, streams, sources = {}, [], []
+    members = {m["relative_path"]: m for m in manifest["files"]}
+    for i, declared in enumerate(manifest["streams"]):
+        stream = normalize_stream(declared, parent["catalog"], artifact["artifact_id"], i)
+        member = members[stream["source_relative_path"]]
+        stream["source_sha256"] = member["sha256"]
+        if stream != declared:
+            raise ImportConflict("Processed stream metadata disagrees with its parent mapping")
+        readers[stream["stream_id"]] = NpyVoltageReader(confined(root / "files" / member["relative_path"], root), stream)
+        streams.append(stream)
+        sources.append({"source_id": stream["source_id"], "relative_path": stream["source_relative_path"],
+                        "sha256": member["sha256"], "format": "npy_voltage_v1", "status": "readable"})
+    catalog = {"schema_version": 1, "reader_version": reader_version, "recording_id": recording["recording_id"],
+               "artifact_id": artifact["artifact_id"], "content_sha256": artifact["content_sha256"],
+               "manifest_sha256": artifact["manifest_sha256"], "data_kind": "derived_samples",
+               "processing": processing, "software_versions": versions, "scientific_validity": "pending",
+               "parent_artifact_id": manifest["parent_artifact_id"], "parent_content_sha256": manifest["parent_content_sha256"],
+               "streams": streams, "sources": sources, "status": "readable", "errors": [],
+               "event_count": len(parent["events"]), "event_scope": "parent_input_original_samples",
+               "parent_sources": deepcopy(parent["catalog"]["sources"]),
+               "clock_policy": "Derived segments retain verified parent clocks; no cross-file synchronization"}
+    return {"catalog": catalog, "readers": readers, "events": deepcopy(parent["events"]), "root": root,
+            "parent_entry": parent}
diff --git a/backend/app/domain/ephys_derived_contracts.py b/backend/app/domain/ephys_derived_contracts.py
new file mode 100644
index 0000000000000000000000000000000000000000..f9c63c7873298d8ea4470c92dac06ec963b0406d
--- /dev/null
+++ b/backend/app/domain/ephys_derived_contracts.py
@@ -0,0 +1,92 @@
+"""First processed-voltage format: explicit linear channel lineage, unchanged sample clock."""
+from copy import deepcopy
+import math
+
+
+def _integer(value):
+    return type(value) is int and value >= 0
+
+
+def _number(value):
+    return type(value) in (int, float) and math.isfinite(value)
+
+
+def normalize_stream(spec, parent, artifact_id, index):
+    """Validate actual clock/sample lineage; resampling needs a later explicit mapping."""
+    if not isinstance(spec, dict):
+        raise ValueError("Derived stream must be an object")
+    stream_key = spec.get("stream_key")
+    if not isinstance(stream_key, str) or not stream_key.strip():
+        raise ValueError("Derived stream_key is required")
+    original = next((s for s in parent["streams"] if s["stream_id"] == spec.get("parent_stream_id")), None)
+    if original is None:
+        raise ValueError("Derived stream must name a stream of its verified parent artifact")
+    fs = spec.get("sampling_frequency_hz")
+    if not _number(fs) or fs != original["sampling_frequency_hz"]:
+        raise ValueError("Derived v1 keeps the parent sampling rate; resampling needs an explicit new format")
+    origin, count = spec.get("sample_index_origin"), spec.get("sample_count")
+    if not _integer(origin) or not _integer(count) or count == 0:
+        raise ValueError("Derived sample origin/count must be explicit nonnegative/positive integers")
+    reference = spec.get("reference")
+    if not isinstance(reference, str) or not reference.strip():
+        raise ValueError("Derived reference must be explicit")
+    channel_specs, segment_specs = spec.get("channels"), spec.get("segments")
+    if not isinstance(channel_specs, list) or not 1 <= len(channel_specs) <= 512:
+        raise ValueError("Derived stream must declare 1..512 channels")
+    if not isinstance(segment_specs, list) or not segment_specs:
+        raise ValueError("Derived stream must declare its segments")
+    source_id, stream_id = f"{artifact_id}:npy:{index}", f"{artifact_id}:stream:{index}"
+    by_id = {c["channel_id"]: c for c in original["channels"]}
+    channels = []
+    for i, channel in enumerate(channel_specs):
+        if not isinstance(channel, dict) or channel.get("unit") != "V" or not isinstance(channel.get("raw_name"), str):
+            raise ValueError("Derived voltage needs explicit names and SI V units")
+        members, weights = channel.get("source_channel_ids"), channel.get("weights")
+        if (not isinstance(members, list) or not members or any(not isinstance(c, str) for c in members)
+                or len(set(members)) != len(members) or any(c not in by_id or by_id[c].get("unit") != "V" for c in members)):
+            raise ValueError("Derived channel lineage must refer to known voltage channels in the parent stream")
+        if not isinstance(weights, list) or len(weights) != len(members) or not all(_number(w) for w in weights):
+            raise ValueError("Derived linear channel weights must be finite and match ordered source channels")
+        channels.append({"channel_id": f"{stream_id}:ch:{i}", "raw_name": channel["raw_name"],
+                         "unit": "V", "quantity": "voltage", "source_channel_ids": list(members),
+                         "weights": list(weights), "channel_type": "unknown"})
+    segments, cursor, previous_parent_stop = [], origin, None
+    for i, segment in enumerate(segment_specs):
+        if not isinstance(segment, dict):
+            raise ValueError("Derived segment must be an object")
+        start, stop = segment.get("start_sample"), segment.get("stop_sample")
+        pstart, pstop = segment.get("parent_start_sample"), segment.get("parent_stop_sample")
+        pseg = next((s for s in original["segments"] if s["segment_id"] == segment.get("parent_segment_id")), None)
+        if (not all(_integer(n) for n in (start, stop, pstart, pstop)) or start != cursor or stop <= start
+                or pseg is None or pstop - pstart != stop - start or pstart < pseg["start_sample"]
+                or pstop > pseg["stop_sample"] or previous_parent_stop is not None and pstart < previous_parent_stop):
+            raise ValueError("Derived ranges must cover stored samples once and map within ordered parent segments")
+        onset = pseg["recording_start_seconds"] + (pstart - pseg["start_sample"]) / fs
+        if (segment.get("clock_mapping") != "identity" or not _number(segment.get("recording_start_seconds"))
+                or not math.isclose(segment["recording_start_seconds"], onset, rel_tol=0, abs_tol=1e-9)):
+            raise ValueError("Derived segment time must retain the exact parent clock and gaps")
+        segments.append({"segment_id": f"{stream_id}:segment:{i}", "start_sample": start, "stop_sample": stop,
+                         "recording_start_seconds": onset, "parent_stream_id": original["stream_id"],
+                         "parent_segment_id": pseg["segment_id"], "parent_start_sample": pstart,
+                         "parent_stop_sample": pstop, "clock_mapping": "identity"})
+        cursor, previous_parent_stop = stop, pstop
+    if cursor != origin + count:
+        raise ValueError("Derived segments do not cover the declared sample count")
+    return {"stream_key": stream_key, "stream_id": stream_id, "source_id": source_id,
+            "parent_stream_id": original["stream_id"], "clock_id": original["clock_id"],
+            "clock_alignment": original["clock_alignment"], "sampling_frequency_hz": fs,
+            "sample_index_origin": origin, "sample_count": count, "channels": channels,
+            "channel_order": [c["channel_id"] for c in channels], "segments": segments,
+            "reference": reference, "source_relative_path": f"voltage-{index}.npy"}
+
+
+def validate_provenance(processing, software_versions):
+    if (not isinstance(processing, list) or not processing or
+            any(not isinstance(p, dict) or not isinstance(p.get("operation"), str) or not p["operation"].strip()
+                or not isinstance(p.get("parameters"), dict) for p in processing)):
+        raise ValueError("Processed voltage must record operations and their parameters")
+    if (not isinstance(software_versions, dict) or not software_versions or
+            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
+                for k, v in software_versions.items())):
+        raise ValueError("Processed voltage must record software versions")
+    return deepcopy(processing), deepcopy(software_versions)
diff --git a/backend/app/domain/ephys_qc.py b/backend/app/domain/ephys_qc.py
new file mode 100644
index 0000000000000000000000000000000000000000..51629dbe999d469df08678bedb340ea8b98700cc
--- /dev/null
+++ b/backend/app/domain/ephys_qc.py
@@ -0,0 +1,251 @@
+"""SEEG scan orchestration, durable optimistic reviews and frozen exports."""
+from copy import deepcopy
+import hashlib
+import json
+import logging
+import platform
+from uuid import uuid4
+
+from ..db import current_ts, get_db
+from .ephys_recordings import get_recording
+from .ephys_signals import verified_catalog, assert_unchanged
+from .ephys_storage import ImportConflict, canonical_bytes, digest
+from .ephys_qc_review import SCOPE, validate_review, summarize
+from .ephys_qc_export import build_package
+
+logger = logging.getLogger(__name__)
+
+
+def _json(value):
+    return canonical_bytes(value).decode("utf-8")
+
+
+def _row(run_id, conn=None):
+    if conn is None:
+        with get_db() as connection:
+            return _row(run_id, connection)
+    row = conn.execute("SELECT * FROM ephys_qc_runs WHERE run_id=?", (run_id,)).fetchone()
+    if row is None:
+        raise LookupError("QC run not found")
+    return row
+
+
+def _audit(conn, run_id, revision, action, snapshot, reviewer=None):
+    conn.execute("INSERT INTO ephys_qc_audit VALUES(?,?,?,?,?,?,?)",
+                 (str(uuid4()), run_id, revision, action, reviewer, _json(snapshot), current_ts()))
+
+
+def _verify_input(row):
+    entry = verified_catalog(row["artifact_id"])
+    provenance = json.loads(row["provenance_json"])
+    for key in ("recording_id", "artifact_id", "content_sha256", "manifest_sha256"):
+        if entry["catalog"][key] != provenance[key]:
+            raise ImportConflict("QC input fingerprint changed; restore input or start a new scan")
+    return entry
+
+
+def create_run(recording_id, payload):
+    if not isinstance(payload, dict):
+        raise ValueError("Use a JSON object")
+    record = get_recording(recording_id)
+    if record["modality"] not in ("seeg", "mixed"):
+        raise ValueError("This scan profile is for SEEG or explicitly mixed recordings")
+    artifact_id = payload.get("artifact_id") or record["input_artifact_id"]
+    entry = verified_catalog(artifact_id)
+    catalog = entry["catalog"]
+    if catalog["recording_id"] != recording_id:
+        raise ValueError("Artifact does not belong to this recording")
+    if not catalog["streams"]:
+        raise ValueError("No readable signal stream; inspect the recording input")
+    config, channel_map = payload.get("config", {}), payload.get("channel_map")
+    if not isinstance(config, dict) or channel_map is not None and not isinstance(channel_map, list):
+        raise ValueError("config must be an object and channel_map a list")
+    from ..processing.ephys_qc.seeg import SEEG_ALGORITHM_VERSION
+    import numpy
+    import scipy
+    request = {"config": config, "channel_map": channel_map}
+    provenance = {key: catalog[key] for key in ("recording_id", "artifact_id", "content_sha256", "manifest_sha256", "reader_version")}
+    provenance.update(algorithm_version=SEEG_ALGORITHM_VERSION, request_sha256=digest(request),
+                      software={"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__})
+    run_id, now = str(uuid4()), current_ts()
+    with get_db() as conn:
+        conn.execute("BEGIN IMMEDIATE")
+        subject = conn.execute("SELECT subject_id FROM ephys_recordings WHERE recording_id=?", (recording_id,)).fetchone()
+        cursor = conn.execute("INSERT INTO jobs(subject_id,job_type,status,payload,created_at) VALUES(?,?,?,?,?)",
+                              (subject["subject_id"], "ephys_qc_scan", "pending",
+                               _json({"run_id": run_id, **provenance}), now))
+        conn.execute("INSERT INTO ephys_qc_runs(run_id,recording_id,artifact_id,job_id,status,provenance_json,request_json,review_json,created_at,updated_at) "
+                     "VALUES(?,?,?,?,?,?,?,?,?,?)", (run_id, recording_id, artifact_id, cursor.lastrowid, "pending",
+                                                    _json(provenance), _json(request), _json({"decisions": [], "summary": None}), now, now))
+        _audit(conn, run_id, 0, "created", provenance)
+    logger.info("SEEG QC created run=%s artifact=%s sha256=%s", run_id, artifact_id, catalog["content_sha256"])
+    return get_run(run_id)
+
+
+def execute_scan(run_id):
+    row = _row(run_id)
+    with get_db() as conn:
+        changed = conn.execute("UPDATE ephys_qc_runs SET status='running',updated_at=? "
+                               "WHERE run_id=? AND status='pending' AND frozen_json IS NULL", (current_ts(), run_id)).rowcount
+        if changed != 1:
+            raise ImportConflict("QC run already executed; create a new scan for a rerun")
+    try:
+        from ..processing.ephys_qc.seeg import scan_seeg, SEEG_ALGORITHM_VERSION
+        provenance = json.loads(row["provenance_json"])
+        if provenance["algorithm_version"] != SEEG_ALGORITHM_VERSION:
+            raise ImportConflict("QC algorithm changed since submission; start a new scan")
+        entry = _verify_input(row)
+        request = json.loads(row["request_json"])
+        if digest(request) != provenance["request_sha256"]:
+            raise ImportConflict("QC request fingerprint changed; create a new scan")
+        def read_window(stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
+            result = entry["readers"][stream_id].read_window(stream_id, segment_id, channel_ids,
+                                                            start_sample, stop_sample, representation=representation)
+            assert_unchanged(entry)
+            return result
+        report = scan_seeg(deepcopy(entry["catalog"]), read_window, **request)
+        assert_unchanged(entry)
+        report["catalog"] = deepcopy(entry["catalog"])
+        review = {"decisions": [], "summary": summarize(report["catalog"], [], report["windows"])}
+        serialized = _json(report)
+        with get_db() as conn:
+            conn.execute("UPDATE ephys_qc_runs SET status='succeeded',report_json=?,report_sha256=?,review_json=?,updated_at=? WHERE run_id=?",
+                         (serialized, hashlib.sha256(serialized.encode()).hexdigest(), _json(review), current_ts(), run_id))
+            _audit(conn, run_id, 0, "scan_completed", {"report_sha256": digest(report)})
+        return {"run_id": run_id, "window_count": len(report["windows"]), "status": "succeeded"}
+    except Exception as exc:
+        with get_db() as conn:
+            conn.execute("UPDATE ephys_qc_runs SET status='failed',error=?,updated_at=? WHERE run_id=?", (str(exc), current_ts(), run_id))
+            _audit(conn, run_id, 0, "scan_failed", {"error": str(exc)})
+        logger.exception("SEEG QC scan failed run=%s", run_id)
+        raise
+
+
+def get_run(run_id):
+    row = _row(run_id)
+    report = json.loads(row["report_json"]) if row["report_json"] else None
+    if report is not None and digest(report) != row["report_sha256"]:
+        raise ImportConflict("QC report fingerprint mismatch; restore the persisted report")
+    result = {key: row[key] for key in ("run_id", "recording_id", "artifact_id", "job_id", "status", "error", "created_at", "updated_at")}
+    result.update(profile="seeg_basic_v1", scope=SCOPE, provenance=json.loads(row["provenance_json"]),
+                  request=json.loads(row["request_json"]), report=report, frozen=bool(row["frozen_json"]))
+    result["review"] = {**json.loads(row["review_json"]), "revision": row["revision"]}
+    # A terminated/crashed worker cannot write its own terminal state. Preserve
+    # the durable job result as the authority for such incomplete runs.
+    if row["status"] in ("pending", "running") and row["job_id"]:
+        with get_db() as conn:
+            job = conn.execute("SELECT status,result FROM jobs WHERE id=?", (row["job_id"],)).fetchone()
+        if job and job["status"] in ("failed", "cancelled", "terminated", "succeeded"):
+            result["status"] = "cancelled" if job["status"] in ("cancelled", "terminated") else "failed"
+            result["error"] = (json.loads(job["result"] or "{}").get("error") or "Worker ended without a completed QC report; start a new scan")
+    if row["frozen_json"]:
+        frozen = json.loads(row["frozen_json"])
+        result.update(frozen_at=frozen["frozen_at"], frozen_by=frozen["frozen_by"], assessment=frozen["assessment"])
+    return result
+
+
+def history(recording_id):
+    get_recording(recording_id)
+    with get_db() as conn:
+        rows = conn.execute("SELECT run_id,status,created_at,frozen_json IS NOT NULL AS frozen,job_id,error "
+                            "FROM ephys_qc_runs WHERE recording_id=? ORDER BY created_at DESC,rowid DESC", (recording_id,)).fetchall()
+    for row in rows:
+        row["frozen"] = bool(row["frozen"])
+        if row["status"] in ("pending", "running"):
+            current = get_run(row["run_id"])
+            row.update(status=current["status"], error=current["error"])
+    return {"recording_id": recording_id, "runs": rows,
+            "current": get_run(rows[0]["run_id"]) if rows else None}
+
+
+def _editable(row, revision):
+    if row["frozen_json"]:
+        raise ImportConflict("QC run is frozen; create a new scan to revise it")
+    if row["revision"] != revision:
+        raise ImportConflict("QC revision conflict; reload the current review before saving")
+    if row["status"] != "succeeded":
+        raise ImportConflict("Wait for a successful scan before review or freeze")
+
+
+def save_review(run_id, payload):
+    run = get_run(run_id)
+    if run["report"] is None:
+        raise ImportConflict("Wait for a successful scan before review")
+    decisions = validate_review(payload, run["report"]["catalog"], run["report"]["windows"])
+    previous = {d["decision_id"]: d for d in run["review"]["decisions"]}
+    for decision in decisions:
+        old = previous.get(decision["decision_id"])
+        content = lambda d: {k: v for k, v in d.items() if k not in ("reviewer", "reviewed_at")}
+        if old and content(old) == content(decision):
+            decision.update(reviewer=old["reviewer"], reviewed_at=old["reviewed_at"])
+    review = {"decisions": decisions, "summary": summarize(run["report"]["catalog"], decisions, run["report"]["windows"])}
+    entry = _verify_input(_row(run_id))
+    with get_db() as conn:
+        conn.execute("BEGIN IMMEDIATE")
+        row = _row(run_id, conn)
+        _editable(row, payload["base_revision"])
+        assert_unchanged(entry)
+        revision = row["revision"] + 1
+        conn.execute("UPDATE ephys_qc_runs SET revision=?,review_json=?,updated_at=? WHERE run_id=?",
+                     (revision, _json(review), current_ts(), run_id))
+        _audit(conn, run_id, revision, "review_saved", review, payload["reviewer"].strip())
+    return get_run(run_id)
+
+
+def freeze_run(run_id, payload):
+    run = get_run(run_id)
+    if not isinstance(payload, dict) or type(payload.get("base_revision")) is not int:
+        raise ValueError("base_revision must be an integer")
+    reviewer = payload.get("reviewer")
+    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
+        raise ValueError("A reviewer name is required")
+    row = _row(run_id)
+    _editable(row, payload["base_revision"])
+    if run["review"]["revision"] != payload["base_revision"]:
+        raise ImportConflict("QC revision changed while preparing the frozen package; reload the review")
+    entry = _verify_input(row)
+    summary = run["review"]["summary"]
+    limitations = ["病理活动、刺激事件与解剖空间质量未评估；用途范围为基础信号技术质量。"]
+    if summary["unknown_seconds"]:
+        limitations.append("未复核或待定区间保留为未知；仅明确保留的区间计入可用时间。")
+    mapping_pending = any(c.get("status") != "confirmed" for c in run["report"]["channel_map"])
+    if mapping_pending:
+        limitations.append("部分通道映射尚未确认。")
+    coverage = run["report"]["coverage"]
+    if coverage["scanned_fraction"] < 1:
+        limitations.append(f"自动特征仅覆盖 {coverage['scanned_fraction']:.1%} 的通道时间；其余区间未自动评估。")
+    integrity = run["report"]["integrity"]
+    if integrity.get("source_errors"):
+        limitations.append("部分输入文件未能读取；当前统计仅包含可读信号流。")
+    if integrity.get("unknown_voltage_unit_channel_ids"):
+        limitations.append("部分通道电压单位未确认，相关幅度指标未评估。")
+    status = "pending" if not summary["reviewed_ratio"] else "degraded" if len(limitations) > 1 else "reviewed"
+    if not summary["kept_seconds"]:
+        availability = "pending" if summary["unknown_seconds"] else "unusable"
+    else:
+        availability = "limited_usable" if summary["excluded_seconds"] or len(limitations) > 1 else "usable"
+    snapshot = deepcopy(run)
+    snapshot.update(frozen=True, frozen_at=current_ts(), frozen_by=reviewer.strip(),
+                    assessment={"status": status, "availability": availability, "scope": SCOPE, "limitations": limitations},
+                    export_validation={"format": "cortrix_internal_qc_zip_v1", "readback": "passed"})
+    payload_bytes = build_package(snapshot)
+    with get_db() as conn:
+        conn.execute("BEGIN IMMEDIATE")
+        _editable(_row(run_id, conn), payload["base_revision"])
+        assert_unchanged(entry)
+        conn.execute("UPDATE ephys_qc_runs SET frozen_json=?,export_sha256=?,updated_at=? WHERE run_id=?",
+                     (_json(snapshot), hashlib.sha256(payload_bytes).hexdigest(), current_ts(), run_id))
+        _audit(conn, run_id, row["revision"], "frozen", {"export_sha256": hashlib.sha256(payload_bytes).hexdigest(),
+                                                        "assessment": snapshot["assessment"]}, reviewer.strip())
+    return get_run(run_id)
+
+
+def export_run(run_id):
+    row = _row(run_id)
+    if not row["frozen_json"]:
+        raise ImportConflict("Freeze the reviewed version before export")
+    _verify_input(row)
+    result = build_package(json.loads(row["frozen_json"]))
+    if hashlib.sha256(result).hexdigest() != row["export_sha256"]:
+        raise ImportConflict("Frozen export fingerprint mismatch; restore the saved version")
+    return result
diff --git a/backend/app/domain/ephys_qc_export.py b/backend/app/domain/ephys_qc_export.py
new file mode 100644
index 0000000000000000000000000000000000000000..e1a20efe9b90951bbe2db25f9a13ef97a4c07ef6
--- /dev/null
+++ b/backend/app/domain/ephys_qc_export.py
@@ -0,0 +1,119 @@
+"""Deterministic internal QC ZIP, with member hashes and independent readback."""
+import csv
+import hashlib
+import html
+import io
+import json
+import zipfile
+
+from .ephys_storage import canonical_bytes
+
+_EVENT_FIELDS = ["event_id", "channel_id", "stream_id", "segment_id", "start_sample", "stop_sample",
+                 "recording_start_seconds", "recording_stop_seconds", "duration_seconds", "recovery_status",
+                 "termination_reason", "score_effect", "reasons", "stages", "features"]
+
+
+def _csv(rows, fields):
+    output = io.StringIO(newline="")
+    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
+    writer.writeheader()
+    for row in rows:
+        values = {}
+        for key, value in row.items():
+            if isinstance(value, (list, dict)):
+                value = json.dumps(value, ensure_ascii=False)
+            # Prevent formula execution when reviewers open CSV in a spreadsheet.
+            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
+                value = "'" + value
+            values[key] = value
+        writer.writerow(values)
+    return output.getvalue().encode("utf-8-sig")
+
+
+def build_package(snapshot):
+    summary = snapshot["review"]["summary"]
+    report = snapshot["report"]
+    files = {"qc_result.json": canonical_bytes(snapshot),
+             "channel_map.json": canonical_bytes(report["channel_map"]),
+             "channels.csv": _csv(summary["channels"], ["channel_id", "raw_name", "stream_id", "total_seconds",
+                                                        "kept_seconds", "excluded_seconds", "unknown_seconds", "reviewed_ratio"]),
+             "decisions.csv": _csv(snapshot["review"]["decisions"], ["decision_id", "channel_id", "stream_id", "scope",
+                                                                       "segment_id", "start_sample", "stop_sample", "action",
+                                                                       "response", "reason", "reviewer", "reviewed_at", "evidence_window_ids"]),
+             "windows.csv": _csv(report["windows"], ["window_id", "channel_id", "stream_id", "segment_id", "start_sample",
+                                                       "stop_sample", "duration_seconds", "score", "flags", "metrics"])}
+    esc = lambda value: html.escape(str(value))
+    rows = "".join("<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key in
+                                    ("raw_name", "kept_seconds", "excluded_seconds", "unknown_seconds")) + "</tr>"
+                   for row in summary["channels"])
+    body = ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>SEEG 基础 QC</title>"
+            "<style>body{font:16px system-ui;margin:40px;max-width:1100px}table{border-collapse:collapse}"
+            "td,th{padding:8px;border:1px solid #bbb}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
+            f"<h1>SEEG 基础技术 QC</h1><p>记录：{esc(snapshot['recording_id'])} · 版本：{esc(snapshot['run_id'])}</p>"
+            f"<p>初评分：{esc(report['initial_score'].get('total'))}；复核后评分：{esc(summary['score'])}；"
+            f"审核覆盖：{summary['reviewed_ratio']:.1%}；待定：{summary['unknown_ratio']:.1%}</p>"
+            f"<p>自动扫描覆盖：{report['coverage']['scanned_fraction']:.1%}；自动规则：{esc(report['profile'])}</p>"
+            f"<p>已复核部分保留率：{esc(summary['retained_percent'])}%；候选证据已裁决：{summary['evidence_reviewed_ratio']:.1%}。</p>"
+            "<p>技术复评分只按明确的候选裁决更新，待定部分沿用初评风险；保留率单独统计。未复核数据不视为合格。</p>"
+            f"<p>评价：{esc(snapshot['assessment']['status'])} · 范围：基础信号技术质量</p>"
+            "<table><thead><tr><th>通道</th><th>确认可用秒数</th><th>排除秒数</th><th>待定秒数</th></tr></thead>"
+            f"<tbody>{rows}</tbody></table><h2>限制</h2><pre>{esc(json.dumps(snapshot['assessment']['limitations'], ensure_ascii=False, indent=2))}</pre>"
+            f"<h2>复核记录</h2><pre>{esc(json.dumps(snapshot['review']['decisions'], ensure_ascii=False, indent=2))}</pre></html>")
+    files["report.html"] = body.encode("utf-8")
+    if "waveform_events" in report:
+        events = report["waveform_events"]
+        files["waveform_events.csv"] = _csv(events, _EVENT_FIELDS)
+        files["windows.csv"] = _csv(report["windows"], ["window_id", "channel_id", "stream_id", "segment_id",
+                                                      "start_sample", "stop_sample", "duration_seconds", "score",
+                                                      "flags", "metrics", "waveform_event_ids"])
+        names = {row["channel_id"]: row["raw_name"] for row in report["channel_map"]}
+        kinds = {"jump": "跳变", "drift": "漂移", "plateau": "平台", "recovery": "恢复确认"}
+        event_rows = []
+        for event in events:
+            stages = "<br>".join(f"{esc(kinds.get(stage['kind'], stage['kind']))}："
+                                  f"[{stage['start_sample']}, {stage['stop_sample']})"
+                                  for stage in event["stages"])
+            status = "已确认恢复" if event["recovery_status"] == "confirmed" else "截至此处未确认恢复"
+            event_rows.append(f"<tr><td style='overflow-wrap:anywhere'>{esc(event['event_id'])}</td>"
+                              f"<td>{esc(names.get(event['channel_id'], event['channel_id']))}</td>"
+                              f"<td>{event['recording_start_seconds']:.4f}</td><td>{event['recording_stop_seconds']:.4f}</td>"
+                              f"<td>{status}</td><td>{stages}</td></tr>")
+        event_html = (f"<h2>波形异常候选（{len(events)}）</h2><p>跳变、漂移、平台及恢复阶段；区间为半开区间，候选供人工复核。</p>"
+                      "<table><thead><tr><th>事件</th><th>通道</th><th>开始秒</th><th>结束秒</th><th>恢复状态</th><th>阶段</th>"
+                      f"</tr></thead><tbody>{''.join(event_rows)}</tbody></table>")
+        files["report.html"] = body.replace("</html>", event_html + "</html>").encode("utf-8")
+    manifest = {"schema_version": 1, "format": "cortrix_internal_qc_zip_v1", "scope": snapshot["scope"],
+                "run_id": snapshot["run_id"], "recording_id": snapshot["recording_id"],
+                "input_fingerprint": snapshot["provenance"]["content_sha256"],
+                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
+    files["manifest.json"] = canonical_bytes(manifest)
+    buffer = io.BytesIO()
+    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
+        for name, data in sorted(files.items()):
+            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
+            info.compress_type = zipfile.ZIP_DEFLATED
+            archive.writestr(info, data)
+    payload = buffer.getvalue()
+    verify_package(payload, snapshot)
+    return payload
+
+
+def verify_package(payload, snapshot):
+    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
+        manifest = json.loads(archive.read("manifest.json"))
+        if set(archive.namelist()) != set(manifest["files"]) | {"manifest.json"}:
+            raise ValueError("QC export member set mismatch")
+        for name, checksum in manifest["files"].items():
+            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
+                raise ValueError(f"QC export checksum mismatch: {name}")
+        if json.loads(archive.read("qc_result.json")) != snapshot:
+            raise ValueError("QC export readback does not match frozen decisions")
+        if len(list(csv.DictReader(io.StringIO(archive.read("decisions.csv").decode("utf-8-sig"))))) != len(snapshot["review"]["decisions"]):
+            raise ValueError("QC export lost decisions")
+        if "waveform_events" in snapshot["report"]:
+            rows = list(csv.DictReader(io.StringIO(archive.read("waveform_events.csv").decode("utf-8-sig"))))
+            events = snapshot["report"]["waveform_events"]
+            if len(rows) != len(events) or any(row["event_id"] != event["event_id"] or
+                                              json.loads(row["stages"]) != event["stages"]
+                                              for row, event in zip(rows, events)):
+                raise ValueError("QC export lost waveform event stages")
diff --git a/backend/app/domain/ephys_qc_review.py b/backend/app/domain/ephys_qc_review.py
new file mode 100644
index 0000000000000000000000000000000000000000..f949ae79febacabca4824138ec51f1337211a90e
--- /dev/null
+++ b/backend/app/domain/ephys_qc_review.py
@@ -0,0 +1,165 @@
+"""Sample-accurate human review: unknown is distinct from accepted signal."""
+from collections import defaultdict
+from uuid import uuid4
+
+from ..db import current_ts
+
+SCOPE = "seeg_basic_signal_only"
+
+
+def validate_review(payload, catalog, windows):
+    if not isinstance(payload, dict) or type(payload.get("base_revision")) is not int:
+        raise ValueError("base_revision must be an integer")
+    reviewer = payload.get("reviewer")
+    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
+        raise ValueError("A reviewer name is required (max 200 characters)")
+    raw = payload.get("decisions")
+    if not isinstance(raw, list) or len(raw) > 20000:
+        raise ValueError("decisions must be a list with at most 20000 entries")
+    channels = {c["channel_id"]: s for s in catalog["streams"] for c in s["channels"]}
+    evidence = {w["window_id"]: w for w in windows}
+    result, ids = [], set()
+    for item in raw:
+        if not isinstance(item, dict) or item.get("channel_id") not in channels:
+            raise ValueError("Each decision needs a channel_id in this input")
+        stream = channels[item["channel_id"]]
+        if item.get("stream_id", stream["stream_id"]) != stream["stream_id"]:
+            raise ValueError("Decision stream and channel disagree")
+        scope = item.get("scope", "interval")
+        action, reason = item.get("action"), item.get("reason")
+        if scope not in ("interval", "channel") or action not in ("keep", "exclude", "pending"):
+            raise ValueError("Use scope channel/interval and action keep/exclude/pending")
+        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
+            raise ValueError("Every decision needs a reason (max 2000 characters)")
+        response = item.get("response")
+        if response not in (None, "confirmed", "rejected", "pending"):
+            raise ValueError("response must be confirmed, rejected or pending")
+        if (response is not None and ((response == "pending") != (action == "pending")) or
+                response == "confirmed" and action != "exclude" or
+                response == "rejected" and action != "keep"):
+            raise ValueError("response must match the technical issue decision")
+        decision_id = item.get("decision_id") or str(uuid4())
+        if not isinstance(decision_id, str) or len(decision_id) > 100 or decision_id in ids:
+            raise ValueError("decision_id must be distinct within a revision")
+        ids.add(decision_id)
+        decision = {"decision_id": decision_id, "channel_id": item["channel_id"],
+                    "stream_id": stream["stream_id"], "scope": scope, "action": action,
+                    "response": response, "reason": reason.strip(), "reviewer": reviewer.strip(),
+                    "reviewed_at": current_ts(), "applicability": SCOPE,
+                    "evidence_window_ids": item.get("evidence_window_ids", [])}
+        if not isinstance(decision["evidence_window_ids"], list) or any(
+                not isinstance(wid, str) or wid not in evidence or
+                evidence[wid]["channel_id"] != item["channel_id"] for wid in decision["evidence_window_ids"]):
+            raise ValueError("Evidence windows must belong to the reviewed channel")
+        if scope == "interval":
+            segment = next((s for s in stream["segments"] if s["segment_id"] == item.get("segment_id")), None)
+            a, b = item.get("start_sample"), item.get("stop_sample")
+            if (segment is None or type(a) is not int or type(b) is not int or
+                    not segment["start_sample"] <= a < b <= segment["stop_sample"]):
+                raise ValueError("Use a half-open integer sample interval inside one segment; split at gaps")
+            decision.update(segment_id=segment["segment_id"], start_sample=a, stop_sample=b)
+            if any(evidence[wid]["segment_id"] != segment["segment_id"] or
+                   evidence[wid]["start_sample"] >= b or evidence[wid]["stop_sample"] <= a
+                   for wid in decision["evidence_window_ids"]):
+                raise ValueError("Evidence must overlap the reviewed interval in the same segment")
+        result.append(decision)
+    # Reject contradictory entries at the same specificity; local intervals may
+    # override whole-channel decisions deliberately, independent of list order.
+    groups = defaultdict(list)
+    for item in result:
+        key = item["channel_id"], item["scope"], item.get("segment_id")
+        for previous in groups[key]:
+            overlaps = item["scope"] == "channel" or (item["start_sample"] < previous["stop_sample"] and
+                                                        previous["start_sample"] < item["stop_sample"])
+            if overlaps and item["action"] != previous["action"]:
+                raise ValueError("Overlapping decisions conflict; edit or remove the existing interval")
+        groups[key].append(item)
+    return result
+
+
+def _pieces(channel_id, segment, decisions):
+    whole = [d for d in decisions if d["channel_id"] == channel_id and d["scope"] == "channel"]
+    local = [d for d in decisions if d["channel_id"] == channel_id and
+             d["scope"] == "interval" and d["segment_id"] == segment["segment_id"]]
+    boundaries = sorted({segment["start_sample"], segment["stop_sample"]} |
+                        {d[k] for d in local for k in ("start_sample", "stop_sample")})
+    base = whole[0]["action"] if whole else "pending"
+    return [(a, b, next((d["action"] for d in local if d["start_sample"] <= a < d["stop_sample"]), base))
+            for a, b in zip(boundaries, boundaries[1:])]
+
+
+def _technical_score(windows, decisions):
+    """Score only explicitly adjudicated evidence, at exact interval boundaries."""
+    from ..processing.ephys_qc.seeg_scoring import compute_quality_score
+    slices, responses = [], {}
+    by_window = defaultdict(list)
+    for decision in decisions:
+        if decision.get("response"):
+            for window_id in decision["evidence_window_ids"]:
+                by_window[window_id].append(decision)
+    for window in windows:
+        relevant = by_window[window["window_id"]]
+        a, b = window["start_sample"], window["stop_sample"]
+        boundaries = {a, b}
+        for decision in relevant:
+            if decision["scope"] == "interval":
+                boundaries.update((max(a, decision["start_sample"]), min(b, decision["stop_sample"])))
+        boundaries = sorted(boundaries)
+        for first, last in zip(boundaries, boundaries[1:]):
+            candidates = [d for d in relevant if d["scope"] == "channel" or d["start_sample"] <= first < d["stop_sample"]]
+            candidates.sort(key=lambda d: d["scope"] == "interval", reverse=True)
+            response = candidates[0]["response"] if candidates else "pending"
+            slice_id = f"{window['window_id']}:slice:{first}:{last}"
+            slices.append({**window, "window_id": slice_id, "start_sample": first, "stop_sample": last,
+                           "duration_seconds": window["duration_seconds"] * (last-first) / (b-a)})
+            responses[slice_id] = "accepted" if response == "confirmed" else response
+    result = compute_quality_score(slices, responses)
+    result["aggregation"] = "duration_weighted_sample_fragments_max_penalty"
+    return result
+
+
+def summarize(catalog, decisions, windows=None):
+    rows, common = [], []
+    for stream in catalog["streams"]:
+        fs = stream["sampling_frequency_hz"]
+        channel_pieces = {}
+        for channel in stream["channels"]:
+            times = {"keep": 0.0, "exclude": 0.0, "pending": 0.0}
+            by_segment = {}
+            for segment in stream["segments"]:
+                parts = _pieces(channel["channel_id"], segment, decisions)
+                by_segment[segment["segment_id"]] = parts
+                for a, b, action in parts:
+                    times[action] += (b - a) / fs
+            channel_pieces[channel["channel_id"]] = by_segment
+            total = sum(times.values())
+            rows.append({"channel_id": channel["channel_id"], "stream_id": stream["stream_id"],
+                         "raw_name": channel.get("raw_name"), "total_seconds": total,
+                         "kept_seconds": times["keep"], "excluded_seconds": times["exclude"],
+                         "unknown_seconds": times["pending"],
+                         "reviewed_ratio": (total - times["pending"]) / total if total else 0})
+        shared = 0.0
+        for segment in stream["segments"]:
+            pieces = [by_segment[segment["segment_id"]] for by_segment in channel_pieces.values()]
+            bounds = sorted({point for parts in pieces for a, b, _ in parts for point in (a, b)})
+            for a, b in zip(bounds, bounds[1:]):
+                if pieces and all(any(x <= a < y and action == "keep" for x, y, action in parts) for parts in pieces):
+                    shared += (b - a) / fs
+        common.append({"stream_id": stream["stream_id"], "channel_ids": list(channel_pieces),
+                       "duration_seconds": shared, "scope": "within_stream_only"})
+    total = sum(r["total_seconds"] for r in rows)
+    kept = sum(r["kept_seconds"] for r in rows)
+    excluded = sum(r["excluded_seconds"] for r in rows)
+    unknown = sum(r["unknown_seconds"] for r in rows)
+    reviewed = kept + excluded
+    technical = _technical_score(windows or [], decisions)
+    return {"channels": rows, "total_channel_seconds": total, "kept_seconds": kept,
+            "excluded_seconds": excluded, "unknown_seconds": unknown,
+            "reviewed_ratio": reviewed / total if total else 0,
+            "unknown_ratio": unknown / total if total else 1,
+            "retained_percent": round(100 * kept / reviewed) if reviewed else None,
+            "reviewed_score": technical["total"], "score": technical["total"], "technical_score": technical,
+            "score_basis": "technical_risk_with_explicit_evidence_responses",
+            "evidence_reviewed_ratio": technical["reviewed_channel_seconds"] / technical["assessed_channel_seconds"]
+                if technical["assessed_channel_seconds"] else 0,
+            "common_usable": common, "interval_convention": "half_open"}
diff --git a/backend/app/domain/ephys_recordings.py b/backend/app/domain/ephys_recordings.py
new file mode 100644
index 0000000000000000000000000000000000000000..b85505af76ea1bd12ef38acfd8645f2ec2313b66
--- /dev/null
+++ b/backend/app/domain/ephys_recordings.py
@@ -0,0 +1,218 @@
+"""Stable recording identity and immutable input snapshots, independent of replay."""
+
+import hashlib
+import json
+import logging
+from pathlib import Path
+import shutil
+import tempfile
+from uuid import uuid4
+
+from ..config import settings
+from ..db import current_ts, get_db
+from .ephys_contracts import SCHEMA_VERSION, descriptor
+from .ephys_storage import (NAMESPACE, ImportConflict, SnapshotWriter, canonical_bytes,
+                            confined, digest, inspect_snapshot)
+
+logger = logging.getLogger(__name__)
+
+
+def _subject(subject_id):
+    with get_db() as conn:
+        subject = conn.execute("SELECT * FROM subjects WHERE subject_id=?", (subject_id,)).fetchone()
+    if subject is None:
+        raise LookupError("Subject not found")
+    return subject
+
+
+def _source_rows(payload, subject, uploads):
+    asset_ids = payload.get("asset_ids")
+    legacy_id = payload.get("legacy_eeg_id")
+    if sum((asset_ids is not None, legacy_id is not None, bool(uploads))) != 1:
+        raise ValueError("Choose exactly one source: asset_ids, legacy_eeg_id, or uploaded files")
+    if uploads:
+        return [], [{"kind": "upload", "original_filename": f.filename} for f in uploads]
+    with get_db() as conn:
+        if asset_ids is not None:
+            if (not isinstance(asset_ids, list) or not asset_ids or
+                    any(type(i) is not int or i <= 0 for i in asset_ids) or
+                    len(asset_ids) != len(set(asset_ids))):
+                raise ValueError("asset_ids must be a nonempty list of distinct positive integers")
+            rows = []
+            for asset_id in sorted(asset_ids):
+                row = conn.execute("SELECT * FROM subject_assets WHERE id=? AND subject_id=?",
+                                   (asset_id, subject["id"])).fetchone()
+                if row is None:
+                    raise LookupError("Asset not found for this subject")
+                rows.append(row)
+            return [r["storage_path"] for r in rows], [
+                {"kind": "asset", "asset_id": r["id"], "asset_type": r["asset_type"],
+                 "original_filename": r["original_filename"], "storage_path": r["storage_path"],
+                 "legacy_checksum": r["checksum"], "metadata_snapshot": r["metadata"],
+                 "row_snapshot": r} for r in rows]
+        if type(legacy_id) is not int or legacy_id <= 0:
+            raise ValueError("legacy_eeg_id must be a positive integer")
+        row = conn.execute("SELECT * FROM eeg_files WHERE id=? AND subject_id=?",
+                           (legacy_id, subject["id"])).fetchone()
+        if row is None:
+            raise LookupError("Legacy EEG record not found for this subject")
+        if not row["raw_path"]:
+            raise ValueError("Legacy record has no raw input; select owned raw assets explicitly")
+        return [row["raw_path"]], [{"kind": "legacy_eeg", "legacy_eeg_id": legacy_id,
+                                    "row_snapshot": row}]
+
+
+def _row(recording_id):
+    with get_db() as conn:
+        row = conn.execute("SELECT * FROM ephys_recordings WHERE recording_id=?",
+                           (recording_id,)).fetchone()
+        if row is None:
+            raise LookupError("Recording not found")
+        artifact = conn.execute("SELECT * FROM ephys_artifacts WHERE artifact_id=?",
+                                (row["input_artifact_id"],)).fetchone()
+    if artifact is None:
+        raise ImportConflict("Recording input artifact is missing; restore its database entry")
+    return row, artifact
+
+
+def _verify_source_rows(conn, sources, subject):
+    for source in sources:
+        if source["kind"] == "upload":
+            continue
+        table = "subject_assets" if source["kind"] == "asset" else "eeg_files"
+        old = source["row_snapshot"]
+        current = conn.execute(f"SELECT * FROM {table} WHERE id=? AND subject_id=?",
+                               (old["id"], subject["id"])).fetchone()
+        if current != old:
+            raise ImportConflict("Source asset/replay entry changed during import; select the current input again")
+
+
+def _artifact_root(artifact):
+    return confined(settings.dataset_root / artifact["storage_path"], settings.dataset_root)
+
+
+def _present(row, artifact, include_manifest=True):
+    result = json.loads(row["descriptor_json"])
+    result.update(recording_id=row["recording_id"], input_artifact_id=row["input_artifact_id"],
+                  created_at=row["created_at"], schema_version=SCHEMA_VERSION,
+                  status="stored_unreviewed", scientific_validity="pending",
+                  metadata_status="declared_unverified", input_artifact=artifact)
+    if include_manifest:
+        try:
+            root = _artifact_root(artifact)
+            if (root / "manifest.json").is_symlink():
+                raise ValueError("Manifest is a symlink")
+            raw = (root / "manifest.json").read_bytes()
+            if hashlib.sha256(raw).hexdigest() != artifact["manifest_sha256"]:
+                raise ValueError("Manifest fingerprint mismatch")
+            result["manifest"] = json.loads(raw)
+        except (OSError, ValueError) as exc:
+            raise ImportConflict(f"Input manifest is unavailable or changed: {exc}; run verify") from exc
+    return result
+
+
+def get_recording(recording_id):
+    return _present(*_row(recording_id))
+
+
+def list_recordings(subject_id):
+    subject = _subject(subject_id)
+    with get_db() as conn:
+        rows = conn.execute("SELECT * FROM ephys_recordings WHERE subject_id=? "
+                            "ORDER BY created_at, recording_id", (subject["id"],)).fetchall()
+        artifacts = {r["artifact_id"]: r for r in conn.execute(
+            "SELECT a.* FROM ephys_artifacts a JOIN ephys_recordings r "
+            "ON a.recording_id=r.recording_id WHERE r.subject_id=?", (subject["id"],)).fetchall()}
+    # Lists are an index, never an integrity verdict. Verification is explicit.
+    return [_present(row, artifacts[row["input_artifact_id"]], False) for row in rows]
+
+
+def verify_recording(recording_id):
+    row, artifact = _row(recording_id)
+    try:
+        issues = inspect_snapshot(_artifact_root(artifact), artifact["manifest_sha256"],
+                                  artifact["content_sha256"])
+    except ValueError as exc:
+        issues = [{"code": "unsafe_storage", "reason": str(exc)}]
+    logger.log(logging.WARNING if issues else logging.INFO,
+               "ephys input verification recording=%s valid=%s issues=%s", recording_id, not issues, issues)
+    return {"recording_id": row["recording_id"], "input_artifact_id": artifact["artifact_id"],
+            "valid": not issues, "issues": issues, "checked_at": current_ts(),
+            "scientific_validity": "pending"}
+
+
+def import_recording(payload, uploads=None):
+    """No key => new identity. Same subject/key + same request/bytes => verified reuse.
+
+    A conflicting key is rejected. Slow copies happen before the DB write lock;
+    publication and idempotency selection are serialized by BEGIN IMMEDIATE.
+    """
+    declared = descriptor(payload)
+    subject = _subject(declared["subject_id"])
+    uploads = list(uploads or [])
+    paths, sources = _source_rows(payload, subject, uploads)
+    raw_root = confined(settings.resourses_root / subject["subject_id"], settings.resourses_root)
+    parent = confined(raw_root / NAMESPACE, settings.resourses_root)
+    parent.mkdir(parents=True, exist_ok=True)
+    recording_id, artifact_id = str(uuid4()), str(uuid4())
+    stage = Path(tempfile.mkdtemp(prefix=".import-", dir=parent))
+    final = parent / recording_id
+    published = False
+    committed = False
+    try:
+        writer = SnapshotWriter(stage)
+        for path in paths:
+            path = Path(path)
+            if not path.is_absolute():
+                path = settings.dataset_root / path
+            path = confined(path, settings.dataset_root)
+            writer.add_path(path, raw_root)
+        for upload in uploads:
+            writer.add_upload(upload)
+        entries = writer.finish()
+        writer.verify_sources()
+        content_sha256 = digest(entries)
+        request_sha256 = digest({"descriptor": declared, "sources": sources,
+                                 "content_sha256": content_sha256})
+        created_at = current_ts()
+        manifest = {"schema_version": SCHEMA_VERSION, "recording_id": recording_id,
+                    "input_artifact_id": artifact_id, **declared, "sources": sources,
+                    "metadata_status": "declared_unverified", "scientific_validity": "pending",
+                    "files": entries, "content_sha256": content_sha256, "created_at": created_at}
+        raw_manifest = canonical_bytes(manifest)
+        (stage / "manifest.json").write_bytes(raw_manifest)
+        (stage / "manifest.json").chmod(0o444)
+        with get_db() as conn:
+            conn.execute("BEGIN IMMEDIATE")
+            _verify_source_rows(conn, sources, subject)
+            old = conn.execute("SELECT recording_id, request_sha256 FROM ephys_recordings "
+                               "WHERE subject_id=? AND import_key=?",
+                               (subject["id"], payload.get("import_key"))).fetchone()
+            if old is not None:
+                if old["request_sha256"] != request_sha256:
+                    raise ImportConflict("import_key already identifies different input or metadata")
+                if not verify_recording(old["recording_id"])["valid"]:
+                    raise ImportConflict("Existing input fingerprint mismatch; restore it or use a new import_key")
+                return get_recording(old["recording_id"]), True
+            # Exclusive reservation: even an unexpected empty destination is never replaced.
+            final.mkdir(exist_ok=False)
+            published = True
+            (stage / "files").rename(final / "files")
+            (stage / "manifest.json").rename(final / "manifest.json")
+            conn.execute("INSERT INTO ephys_recordings VALUES(?,?,?,?,?,?,?,?,?,?)",
+                         (recording_id, subject["id"], declared["label"], declared["modality"],
+                          declared["session_id"], artifact_id, payload.get("import_key"),
+                          request_sha256, canonical_bytes(declared).decode("utf-8"), created_at))
+            conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                         (artifact_id, recording_id, "raw_input", final.relative_to(settings.dataset_root.resolve()).as_posix(),
+                          content_sha256, hashlib.sha256(raw_manifest).hexdigest(),
+                          len(entries), writer.size, created_at))
+        committed = True
+        logger.info("ephys recording imported recording=%s artifact=%s files=%d sha256=%s",
+                    recording_id, artifact_id, len(entries), content_sha256)
+        return get_recording(recording_id), False
+    finally:
+        if stage.exists():
+            shutil.rmtree(stage)
+        if published and not committed:
+            shutil.rmtree(final)
diff --git a/backend/app/domain/ephys_signals.py b/backend/app/domain/ephys_signals.py
new file mode 100644
index 0000000000000000000000000000000000000000..887795d2991d14f53b1313cfac56422c876fc01d
--- /dev/null
+++ b/backend/app/domain/ephys_signals.py
@@ -0,0 +1,193 @@
+"""Verified raw-signal catalogs. Legacy replay and ingestion stay independent."""
+
+from collections import OrderedDict
+from copy import deepcopy
+import hashlib
+import json
+import logging
+from pathlib import Path
+from threading import RLock
+
+from ..db import get_db
+from ..processing.ephys_qc.readers.edf import EdfBdfReader
+from .ephys_recordings import _artifact_root, get_recording
+from .ephys_storage import ImportConflict, confined, inspect_snapshot, inventory, stamp
+
+READER_VERSION = "cortrix-ephys-window-v1"
+_CACHE = OrderedDict()
+_LOCK = RLock()
+_MAX_CATALOGS = 8
+logger = logging.getLogger(__name__)
+
+
+def _fingerprint(root):
+    """Stat guard is accepted only after a full SHA256 verification in this process.
+
+    Including inode/ctime detects same-length edits with restored mtime. No input
+    sample arrays are cached. A changed guard always triggers full verification.
+    """
+    root = confined(root, root.parent)
+    manifest = confined(root / "manifest.json", root)
+    entries = [("manifest.json", stamp(manifest))]
+    entries.extend((p.relative_to(root).as_posix(), stamp(p)) for p in inventory(root / "files"))
+    entries.append(("root_entries", tuple(sorted(p.name for p in root.iterdir()))))
+    return tuple(entries)
+
+
+def _neuracle_evidence(root, manifest):
+    evidence = []
+    for item in manifest["files"]:
+        relative = Path(item["relative_path"])
+        if relative.name != "recordInformation.json":
+            continue
+        if item["size_bytes"] > 4 * 1024 * 1024:
+            evidence.append({"source": relative.as_posix(), "status": "unread",
+                             "reason": "metadata exceeds 4 MiB; explicit review required"})
+            continue
+        try:
+            raw = json.loads((root / "files" / relative).read_text(encoding="utf-8-sig"))
+            if not isinstance(raw, dict):
+                raise ValueError("recordInformation must be an object")
+            # Only acquisition fields, never patient demographics or clinical notes.
+            info = raw.get("DataFileInformations", [])
+            selected = [{key: block.get(key) for key in
+                         ("FileName", "SampleRate", "BeginTimeStamp", "DataLength")}
+                        for block in info if isinstance(block, dict)] if isinstance(info, list) else []
+            evidence.append({"source": relative.as_posix(), "status": "declared_unverified",
+                             "sample_rate": raw.get("SampleRate"), "data_files": selected,
+                             "clock_alignment": "unverified",
+                             "policy": "Header calibration and TAL are retained; sidecar never overrides them"})
+        except (ValueError, OSError) as exc:
+            evidence.append({"source": relative.as_posix(), "status": "failed", "reason": str(exc)})
+    return evidence
+
+
+def _build(recording, root):
+    readers, streams, sources, events, errors = {}, [], [], [], []
+    for member in recording["manifest"]["files"]:
+        relative = member["relative_path"]
+        if Path(relative).suffix.lower() not in (".edf", ".bdf"):
+            continue
+        source_key = recording["input_artifact_id"] + "\0" + relative
+        source_id = "file-" + hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
+        path = confined(root / "files" / relative, root)
+        try:
+            reader = EdfBdfReader(path, source_id=source_id)
+            description = reader.describe()
+            source = {"source_id": source_id, "relative_path": relative,
+                      "sha256": member["sha256"], "format": description["format"],
+                      "clock": description.get("clock"),
+                      "warnings": description.get("warnings", []), "status": "readable"}
+            sources.append(source)
+            for stream in description["streams"]:
+                stream = deepcopy(stream)
+                stream["channel_order_policy"] = stream["channel_order"]
+                stream["channel_order"] = [c["channel_id"] for c in stream["channels"]]
+                stream.update(source_id=source_id, clock_id=source_id,
+                              source_relative_path=relative, source_sha256=member["sha256"],
+                              clock_alignment="source_local_unaligned")
+                readers[stream["stream_id"]] = reader
+                streams.append(stream)
+            for event in description.get("events", []):
+                events.append({**event, "source_id": source_id, "clock_id": source_id,
+                               "clock_alignment": "source_local_unaligned"})
+        except ValueError as exc:
+            error = {"source_id": source_id, "relative_path": relative, "status": "failed",
+                     "reason": str(exc)}
+            errors.append(error)
+            sources.append(error)
+            logger.warning("ephys source unreadable artifact=%s source=%s reason=%s",
+                           recording["input_artifact_id"], source_id, exc)
+    catalog = {"schema_version": 1, "reader_version": READER_VERSION,
+               "recording_id": recording["recording_id"], "artifact_id": recording["input_artifact_id"],
+               "content_sha256": recording["input_artifact"]["content_sha256"],
+               "manifest_sha256": recording["input_artifact"]["manifest_sha256"],
+               "data_kind": "raw_samples", "processing": [], "scientific_validity": "pending",
+               "streams": streams, "sources": sources, "event_count": len(events),
+               "status": "readable" if streams and not errors else "partial" if streams else "unreadable",
+               "errors": errors, "declared_metadata": recording["metadata"],
+               "acquisition_evidence": _neuracle_evidence(root, recording["manifest"]),
+               "clock_policy": "Timestamps are in each source file clock; no cross-file synchronization assumed",
+               "unsupported_members": [m["relative_path"] for m in recording["manifest"]["files"]
+                                       if Path(m["relative_path"]).suffix.lower() not in (".edf", ".bdf")]}
+    return {"catalog": catalog, "readers": readers, "events": events, "root": root}
+
+
+def verified_catalog(artifact_id):
+    with get_db() as conn:
+        artifact = conn.execute("SELECT * FROM ephys_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
+    if artifact is None:
+        raise LookupError("Input artifact not found")
+    if artifact["kind"] not in ("raw_input", "derived_voltage"):
+        raise ValueError("Unsupported signal artifact kind")
+    recording = get_recording(artifact["recording_id"])
+    if artifact["kind"] == "raw_input" and recording["input_artifact_id"] != artifact_id:
+        raise ImportConflict("Artifact and recording identity disagree; restore the database relation")
+    root = _artifact_root(artifact)
+    try:
+        manifest, parent = None, None
+        if artifact["kind"] == "derived_voltage":
+            from .ephys_derived import read_manifest, build_catalog
+            manifest = read_manifest(artifact, root, recording)
+            parent = verified_catalog(recording["input_artifact_id"])
+            if manifest["parent_content_sha256"] != parent["catalog"]["content_sha256"]:
+                raise ImportConflict("Processed voltage belongs to different parent bytes")
+        with _LOCK:
+            signature = _fingerprint(root)
+            key = (str(root), artifact_id, artifact["content_sha256"], artifact["manifest_sha256"], READER_VERSION)
+            cached = _CACHE.get(key)
+            if cached is not None and cached["signature"] == signature:
+                if parent is not None:
+                    # Identical restored parent bytes can have a new inode/ctime.
+                    # Keep the freshly SHA-verified guard, not an obsolete parent entry.
+                    cached["parent_entry"] = parent
+                _CACHE.move_to_end(key)
+                return cached
+            issues = inspect_snapshot(root, artifact["manifest_sha256"], artifact["content_sha256"])
+            if issues:
+                raise ImportConflict(f"Input fingerprint mismatch; restore snapshot before reading: {issues}")
+            logger.info("ephys catalog verified artifact=%s route=%s sha256=%s", artifact_id,
+                        "cold_sha256_verification" if cached is None else "changed_stat_reverification",
+                        artifact["content_sha256"])
+            entry = (_build(recording, root) if parent is None else
+                     build_catalog(artifact, root, recording, manifest, parent, READER_VERSION))
+            if _fingerprint(root) != signature:
+                raise ImportConflict("Input changed while building the signal catalog; retry after restoring it")
+            entry["signature"] = signature
+            _CACHE[key] = entry
+            _CACHE.move_to_end(key)
+            while len(_CACHE) > _MAX_CATALOGS:
+                _CACHE.popitem(last=False)
+            return entry
+    except (OSError, ValueError) as exc:
+        if isinstance(exc, ImportConflict):
+            raise
+        raise ImportConflict(f"Cannot verify input snapshot: {exc}") from exc
+
+
+def assert_unchanged(entry):
+    try:
+        if _fingerprint(entry["root"]) != entry["signature"]:
+            raise ImportConflict("Input changed during window read; result discarded")
+        if "parent_entry" in entry:
+            assert_unchanged(entry["parent_entry"])
+    except (OSError, ValueError) as exc:
+        raise ImportConflict(f"Input changed during window read; result discarded: {exc}") from exc
+
+
+def describe_artifact(artifact_id):
+    return deepcopy(verified_catalog(artifact_id)["catalog"])
+
+
+def list_events(artifact_id, offset=0, limit=100):
+    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 1000:
+        raise ValueError("Events need offset>=0 and 1<=limit<=1000")
+    entry = verified_catalog(artifact_id)
+    result = {"artifact_id": artifact_id, "recording_id": entry["catalog"]["recording_id"],
+              "content_sha256": entry["catalog"]["content_sha256"], "offset": offset, "limit": limit,
+              "total": len(entry["events"]), "events": deepcopy(entry["events"][offset:offset + limit]),
+              "event_scope": entry["catalog"].get("event_scope", "input_original_samples")}
+    for key in ("parent_artifact_id", "parent_content_sha256"):
+        if key in entry["catalog"]:
+            result[key] = entry["catalog"][key]
+    return result
diff --git a/backend/app/domain/ephys_storage.py b/backend/app/domain/ephys_storage.py
new file mode 100644
index 0000000000000000000000000000000000000000..39323a854232b271077471804aa9786332095c92
--- /dev/null
+++ b/backend/app/domain/ephys_storage.py
@@ -0,0 +1,238 @@
+"""Streaming snapshots for QC. No hard links, shared staging names or replacement.
+
+Immutability is enforced by the import API, not a claim of OS-level WORM storage.
+Explicit verification detects later external changes, including ZIP member changes.
+"""
+
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import unicodedata
+import zipfile
+
+NAMESPACE = "ephys-recordings"
+CHUNK_SIZE = 1024 * 1024
+MAX_FILES = 100_000
+MAX_BYTES = 4 * 1024 ** 4  # expanded input; supports large Neuropixels recordings
+
+
+class ImportConflict(ValueError):
+    pass
+
+
+def canonical_bytes(value):
+    return json.dumps(value, ensure_ascii=False, sort_keys=True,
+                      separators=(",", ":"), allow_nan=False).encode("utf-8")
+
+
+def digest(value):
+    return hashlib.sha256(canonical_bytes(value)).hexdigest()
+
+
+def safe_relative(name):
+    if (not isinstance(name, str) or not name or name.startswith("/") or
+            any(c in name for c in "\\\0:") or
+            any(part in ("", ".", "..") for part in name.split("/"))):
+        raise ValueError(f"Unsafe input path: {name!r}")
+    return Path(*name.split("/"))
+
+
+def confined(path, root):
+    """Reject links and escaping paths, including symlinked ancestor directories."""
+    configured_root = Path(os.path.abspath(root))
+    root = configured_root.resolve()
+    path = Path(os.path.abspath(path))
+    try:
+        parts = path.relative_to(configured_root).parts
+    except ValueError:
+        try:
+            parts = path.relative_to(root).parts
+        except ValueError as exc:
+            raise ValueError("Input must belong to the requested subject's raw storage") from exc
+    cursor = root
+    for part in parts:
+        cursor /= part
+        if cursor.is_symlink():
+            raise ValueError(f"Symlink inputs/storage are not supported: {cursor.name}")
+    return root.joinpath(*parts)
+
+
+def hash_file(path):
+    hasher = hashlib.sha256()
+    size = 0
+    # O_NOFOLLOW also protects against a leaf replaced between inventory and open.
+    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
+    with os.fdopen(fd, "rb") as source:
+        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
+            raise ValueError("Only regular files are accepted")
+        for block in iter(lambda: source.read(CHUNK_SIZE), b""):
+            size += len(block)
+            hasher.update(block)
+    return {"size_bytes": size, "sha256": hasher.hexdigest()}
+
+
+def inventory(root):
+    """Inventory regular files, rejecting links/special files without following them."""
+    root = Path(root)
+    if root.is_symlink():
+        raise ValueError("Symlink input is not supported")
+    if root.is_file():
+        return [root]
+    if not root.is_dir():
+        raise ValueError(f"Input does not exist: {root.name}")
+    files = []
+    for folder, dirs, names in os.walk(root, followlinks=False):
+        for name in dirs + names:
+            path = Path(folder) / name
+            mode = path.lstat().st_mode
+            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
+                raise ValueError(f"Non-regular input: {path.relative_to(root)}")
+        files.extend(Path(folder) / name for name in names)
+        if len(files) > MAX_FILES:
+            raise ValueError("Input exceeds the file count limit")
+    return sorted(files)
+
+
+def stamp(path):
+    value = path.stat()
+    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
+
+
+class SnapshotWriter:
+    def __init__(self, stage):
+        self.stage = Path(stage)
+        self.root = self.stage / "files"
+        self.root.mkdir()
+        self.entries = []
+        self.names = {}
+        self.size = 0
+        self.source_checks = []
+
+    def _reserve(self, name):
+        relative = safe_relative(name)
+        # Reject ambiguous case/Unicode names even on case-sensitive hosts.
+        for i in range(1, len(relative.parts) + 1):
+            prefix = "/".join(relative.parts[:i])
+            key = unicodedata.normalize("NFC", prefix).casefold()
+            if key in self.names and (self.names[key] != prefix or i == len(relative.parts)):
+                raise ValueError(f"Duplicate or ambiguous input name: {name}")
+            self.names[key] = prefix
+        if len(self.entries) >= MAX_FILES:
+            raise ValueError("Input exceeds the file count limit")
+        return relative
+
+    def add_stream(self, name, source):
+        relative = self._reserve(name)
+        target = self.root / relative
+        target.parent.mkdir(parents=True, exist_ok=True)
+        hasher = hashlib.sha256()
+        size = 0
+        with target.open("xb") as output:
+            for block in iter(lambda: source.read(CHUNK_SIZE), b""):
+                size += len(block)
+                self.size += len(block)
+                if self.size > MAX_BYTES:
+                    raise ValueError("Expanded input exceeds the storage limit")
+                output.write(block)
+                hasher.update(block)
+        target.chmod(0o444)
+        entry = {"relative_path": relative.as_posix(), "size_bytes": size,
+                 "sha256": hasher.hexdigest()}
+        self.entries.append(entry)
+        return entry
+
+    def add_zip(self, source):
+        try:
+            with zipfile.ZipFile(source) as archive:
+                if len(archive.infolist()) > MAX_FILES:
+                    raise ValueError("Archive exceeds the entry count limit")
+                for member in archive.infolist():
+                    safe_relative(member.filename.rstrip("/"))
+                    kind = stat.S_IFMT(member.external_attr >> 16)
+                    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
+                        raise ValueError("ZIP links and special files are not supported")
+                    if member.flag_bits & 1:
+                        raise ValueError("Encrypted ZIP inputs are not supported")
+                    if member.is_dir():
+                        continue
+                    if self.size + member.file_size > MAX_BYTES:
+                        raise ValueError("Expanded archive exceeds the storage limit")
+                    with archive.open(member) as stream:
+                        self.add_stream(member.filename, stream)
+        except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
+            raise ValueError(f"Cannot read ZIP input: {exc}") from exc
+
+    def add_upload(self, upload):
+        name = upload.filename
+        safe_relative(name)
+        if name.lower().endswith(".zip"):
+            # FileStorage is seekable; ZipFile streams decompression one member at a time.
+            self.add_zip(upload.stream)
+        else:
+            self.add_stream(name, upload.stream)
+
+    def add_path(self, source, subject_root):
+        source = confined(source, subject_root)
+        paths = inventory(source)
+        before = {path: stamp(path) for path in paths}
+        copied = {}
+        for path in paths:
+            confined(path, subject_root)
+            if source.is_file() and path.suffix.lower() == ".zip":
+                copied[path] = hash_file(path)
+                self.add_zip(path)
+            else:
+                name = path.relative_to(source).as_posix() if source.is_dir() else path.name
+                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
+                with os.fdopen(fd, "rb") as stream:
+                    entry = self.add_stream(name, stream)
+                copied[path] = {k: entry[k] for k in ("size_bytes", "sha256")}
+        self.source_checks.append((source, subject_root, paths, before, copied))
+
+    def verify_sources(self):
+        # Verify the complete source set after ALL copies, not just each file locally.
+        for source, subject_root, paths, before, copied in self.source_checks:
+            confined(source, subject_root)
+            if inventory(source) != paths:
+                raise ImportConflict("Source file set changed during import; retry after upload completes")
+            for path in paths:
+                confined(path, subject_root)
+                if stamp(path) != before[path] or hash_file(path) != copied[path]:
+                    raise ImportConflict("Source bytes changed during import; retry after upload completes")
+
+    def finish(self):
+        if not self.entries:
+            raise ValueError("Input contains no regular files")
+        return sorted(self.entries, key=lambda entry: entry["relative_path"])
+
+
+def inspect_snapshot(root, expected_manifest_sha256, expected_content_sha256):
+    """Return differences; never silently accept a present but stale artifact."""
+    issues = []
+    try:
+        root = Path(root)
+        if root.is_symlink() or (root / "manifest.json").is_symlink():
+            raise ValueError("Snapshot or manifest is a symlink")
+        raw = (root / "manifest.json").read_bytes()
+        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
+            issues.append({"code": "manifest_mismatch"})
+        manifest = json.loads(raw)
+        actual = [{"relative_path": p.relative_to(root / "files").as_posix(), **hash_file(p)}
+                  for p in inventory(root / "files")]
+        actual.sort(key=lambda entry: entry["relative_path"])
+        if digest(actual) != expected_content_sha256:
+            issues.append({"code": "content_mismatch"})
+        expected = {entry["relative_path"]: entry for entry in manifest["files"]}
+        observed = {entry["relative_path"]: entry for entry in actual}
+        for name in sorted(expected.keys() | observed.keys()):
+            if expected.get(name) != observed.get(name):
+                kind = "added" if name not in expected else "missing" if name not in observed else "changed"
+                issues.append({"code": kind, "relative_path": name})
+        extras = sorted(p.name for p in root.iterdir() if p.name not in ("files", "manifest.json"))
+        if extras:
+            issues.append({"code": "unexpected_root_entries", "paths": extras})
+    except (OSError, ValueError, KeyError, TypeError) as exc:
+        issues.append({"code": "unreadable_snapshot", "reason": str(exc)})
+    return issues
diff --git a/backend/app/domain/ephys_windows.py b/backend/app/domain/ephys_windows.py
new file mode 100644
index 0000000000000000000000000000000000000000..37e57dbe64d5930af299c517e888c6a9d2fe529a
--- /dev/null
+++ b/backend/app/domain/ephys_windows.py
@@ -0,0 +1,108 @@
+"""Bounded raw windows and explicitly separate min/max overviews."""
+
+import json
+import struct
+import numpy as np
+
+from .ephys_signals import verified_catalog, assert_unchanged
+
+MAX_WINDOW_BYTES = 16 * 1024 * 1024
+MAX_OVERVIEW_BINS = 4096
+MAX_OVERVIEW_INPUT_BYTES = 256 * 1024 * 1024
+
+
+def _select(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation):
+    if (not isinstance(channel_ids, list) or not channel_ids or len(channel_ids) > 512 or
+            any(not isinstance(c, str) for c in channel_ids) or len(set(channel_ids)) != len(channel_ids)):
+        raise ValueError("Select 1..512 distinct ordered channel IDs")
+    if type(start_sample) is not int or type(stop_sample) is not int or not 0 <= start_sample < stop_sample:
+        raise ValueError("Use a nonempty half-open integer sample range [start_sample, stop_sample)")
+    if representation not in ("physical", "digital"):
+        raise ValueError("representation must explicitly be physical or digital")
+    entry = verified_catalog(artifact_id)
+    stream = next((s for s in entry["catalog"]["streams"] if s["stream_id"] == stream_id), None)
+    if stream is None:
+        raise LookupError("Stream not found; inspect signals for unsupported or unreadable sources")
+    segment = next((s for s in stream["segments"] if s["segment_id"] == segment_id), None)
+    if segment is None:
+        raise LookupError("Segment not found in this stream")
+    if start_sample < segment["start_sample"] or stop_sample > segment["stop_sample"]:
+        raise ValueError("Window must lie within one segment; no gap interpolation or tail padding")
+    by_id = {c["channel_id"]: c for c in stream["channels"]}
+    if any(c not in by_id for c in channel_ids):
+        raise ValueError("Requested channel does not belong to this stream")
+    channels = [by_id[c] for c in channel_ids]
+    n_samples = stop_sample - start_sample
+    catalog = entry["catalog"]
+    metadata = {key: catalog[key] for key in ("schema_version", "reader_version", "recording_id",
+                                             "artifact_id", "content_sha256", "data_kind", "processing")}
+    metadata.update(stream_id=stream_id, segment_id=segment_id, channel_ids=channel_ids,
+                    units=["count" if representation == "digital" else c.get("unit") for c in channels],
+                    shape=[len(channel_ids), n_samples], dtype="<f8", order="C",
+                    start_sample=start_sample, stop_sample=stop_sample, representation=representation,
+                    sampling_frequency_hz=stream["sampling_frequency_hz"], clock_id=stream["clock_id"],
+                    clock_alignment=stream["clock_alignment"],
+                    recording_start_seconds=segment["recording_start_seconds"] +
+                    (start_sample - segment["start_sample"]) / stream["sampling_frequency_hz"],
+                    source_relative_path=stream["source_relative_path"],
+                    source_sha256=stream["source_sha256"], interval_convention="half_open",
+                    sample_index_origin=stream["sample_index_origin"])
+    for key in ("parent_artifact_id", "parent_content_sha256", "software_versions"):
+        if key in catalog:
+            metadata[key] = catalog[key]
+    return entry, metadata
+
+
+def samples(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation):
+    if len(channel_ids) * max(0, stop_sample - start_sample) * 8 > MAX_WINDOW_BYTES:
+        raise ValueError("Window exceeds 16 MiB; request smaller sample ranges or fewer channels")
+    entry, metadata = _select(artifact_id, stream_id, segment_id, channel_ids,
+                              start_sample, stop_sample, representation)
+    data = entry["readers"][stream_id].read_window(stream_id, segment_id, channel_ids,
+                                                   start_sample, stop_sample, representation=representation)
+    data = np.asarray(data, dtype="<f8", order="C")
+    if list(data.shape) != metadata["shape"]:
+        raise ValueError("Reader returned a shape inconsistent with requested channel/sample order")
+    assert_unchanged(entry)
+    return metadata, data
+
+
+def overview(artifact_id, stream_id, segment_id, channel_ids, start_sample, stop_sample,
+             representation, bin_samples):
+    if type(bin_samples) is not int or bin_samples <= 0:
+        raise ValueError("bin_samples must be a positive integer")
+    count = (stop_sample - start_sample + bin_samples - 1) // bin_samples
+    if not 1 <= count <= MAX_OVERVIEW_BINS:
+        raise ValueError("Overview must contain 1..4096 bins; increase bin_samples or shorten the range")
+    if count * len(channel_ids) * 16 > MAX_WINDOW_BYTES:
+        raise ValueError("Overview result exceeds 16 MiB; request fewer channels or bins")
+    if (stop_sample - start_sample) * len(channel_ids) * 8 > MAX_OVERVIEW_INPUT_BYTES:
+        raise ValueError("Overview scans at most 256 MiB per request; split into segment subranges")
+    entry, metadata = _select(artifact_id, stream_id, segment_id, channel_ids,
+                              start_sample, stop_sample, representation)
+    output = np.empty((len(channel_ids), count, 2), dtype="<f8")
+    starts, stops = [], []
+    # Scan bounded pieces even when a single overview bin spans a very long record.
+    step = max(1, (1024 * 1024) // (8 * len(channel_ids)))
+    reader = entry["readers"][stream_id]
+    for index, start in enumerate(range(start_sample, stop_sample, bin_samples)):
+        stop = min(start + bin_samples, stop_sample)
+        low, high = np.full(len(channel_ids), np.inf), np.full(len(channel_ids), -np.inf)
+        for part in range(start, stop, step):
+            data = reader.read_window(stream_id, segment_id, channel_ids, part, min(part + step, stop),
+                                      representation=representation)
+            low, high = np.minimum(low, data.min(axis=1)), np.maximum(high, data.max(axis=1))
+        output[:, index, 0], output[:, index, 1] = low, high
+        starts.append(start)
+        stops.append(stop)
+    metadata.update(source_data_kind=metadata["data_kind"], data_kind="minmax_overview", shape=list(output.shape),
+                    axes=["channel", "bin", "extremum"], extrema=["min", "max"],
+                    bin_samples=bin_samples, bin_start_samples=starts, bin_stop_samples=stops)
+    assert_unchanged(entry)
+    return metadata, output
+
+
+def encode_window(metadata, array):
+    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
+    prefix = b"EPHY" + struct.pack("<I", len(header)) + header
+    return prefix + b"\0" * (-len(prefix) % 8) + array.astype("<f8", copy=False).tobytes(order="C")
diff --git a/backend/app/jobs/tasks.py b/backend/app/jobs/tasks.py
index 2804270fb422f5e4e0368fe9df3b29d12baa3c88..1e662e2353d7d62dc0954aba8ea35620f6268d69 100644
--- a/backend/app/jobs/tasks.py
+++ b/backend/app/jobs/tasks.py
@@ -266,2 +266,5 @@ def run_job(job: Dict) -> Tuple[bool, str, Dict]:

+    if job_type == "ephys_qc_scan":
+        from ..domain.ephys_qc import execute_scan
+        return _run_with_logging(job["id"], subject_id, job_type, execute_scan, run_id=payload["run_id"])
     if job_type == "prepare_volume":
diff --git a/backend/app/migrations/ephys_artifacts_v2.py b/backend/app/migrations/ephys_artifacts_v2.py
new file mode 100644
index 0000000000000000000000000000000000000000..8e37b52ce5a34bc5f6857d9f1518fcd6a1ffbc27
--- /dev/null
+++ b/backend/app/migrations/ephys_artifacts_v2.py
@@ -0,0 +1,71 @@
+"""Allow immutable derived voltage artifacts without replacing the raw input."""
+
+import logging
+
+
+logger = logging.getLogger(__name__)
+_COLUMNS = (
+    "artifact_id", "recording_id", "kind", "storage_path", "content_sha256",
+    "manifest_sha256", "file_count", "size_bytes", "created_at",
+)
+
+
+def _value(row, name, index):
+    return row[index] if isinstance(row, tuple) else row[name]
+
+
+def _recording_is_globally_unique(conn):
+    for row in conn.execute("PRAGMA index_list(ephys_artifacts)").fetchall():
+        if not _value(row, "unique", 2) or _value(row, "partial", 4):
+            continue
+        name = _value(row, "name", 1).replace('"', '""')
+        columns = [_value(col, "name", 2) for col in
+                   conn.execute('PRAGMA index_info("{}")'.format(name)).fetchall()]
+        if columns == ["recording_id"]:
+            return True
+    return False
+
+
+def upgrade(conn):
+    """Repeatable table rebuild; preserve all nine positional S1 columns.
+
+    A savepoint keeps DDL and the row copy atomic within the caller's transaction.
+    The migration does not disable foreign keys or commit unrelated writes.
+    """
+    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ephys_artifacts'").fetchone()
+    if row is None:
+        raise RuntimeError("Run ephys_recordings_v1 before ephys_artifacts_v2")
+    columns = tuple(_value(col, "name", 1) for col in
+                    conn.execute("PRAGMA table_info(ephys_artifacts)").fetchall())
+    if columns != _COLUMNS:
+        raise RuntimeError("Unexpected ephys_artifacts columns; inspect schema before migration")
+    rebuild = "'derived_voltage'" not in _value(row, "sql", 0) or _recording_is_globally_unique(conn)
+    conn.execute("SAVEPOINT ephys_artifacts_v2_upgrade")
+    try:
+        if rebuild:
+            conn.execute("""
+                CREATE TABLE ephys_artifacts_v2_new (
+                    artifact_id TEXT PRIMARY KEY,
+                    recording_id TEXT NOT NULL
+                        REFERENCES ephys_recordings(recording_id) ON DELETE CASCADE,
+                    kind TEXT NOT NULL CHECK(kind IN ('raw_input', 'derived_voltage')),
+                    storage_path TEXT NOT NULL UNIQUE,
+                    content_sha256 TEXT NOT NULL,
+                    manifest_sha256 TEXT NOT NULL,
+                    file_count INTEGER NOT NULL,
+                    size_bytes INTEGER NOT NULL,
+                    created_at TEXT NOT NULL
+                )
+            """)
+            names = ", ".join(_COLUMNS)
+            conn.execute("INSERT INTO ephys_artifacts_v2_new ({0}) SELECT {0} FROM ephys_artifacts".format(names))
+            conn.execute("DROP TABLE ephys_artifacts")
+            conn.execute("ALTER TABLE ephys_artifacts_v2_new RENAME TO ephys_artifacts")
+        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ephys_artifacts_raw_input "
+                     "ON ephys_artifacts(recording_id) WHERE kind='raw_input'")
+        conn.execute("RELEASE SAVEPOINT ephys_artifacts_v2_upgrade")
+    except Exception:
+        conn.execute("ROLLBACK TO SAVEPOINT ephys_artifacts_v2_upgrade")
+        conn.execute("RELEASE SAVEPOINT ephys_artifacts_v2_upgrade")
+        raise
+    logger.info("ephys_artifacts_v2 route=%s", "migrated" if rebuild else "schema_verified")
diff --git a/backend/app/migrations/ephys_qc_v1.py b/backend/app/migrations/ephys_qc_v1.py
new file mode 100644
index 0000000000000000000000000000000000000000..4a0a893f9a1f30ff73a772229bf98ee77c8315dd
--- /dev/null
+++ b/backend/app/migrations/ephys_qc_v1.py
@@ -0,0 +1,40 @@
+"""Add durable scan runs and append-only review snapshots alongside raw inputs."""
+import logging
+
+SCHEMA = """
+CREATE TABLE IF NOT EXISTS ephys_qc_runs (
+    run_id TEXT PRIMARY KEY,
+    recording_id TEXT NOT NULL REFERENCES ephys_recordings(recording_id) ON DELETE CASCADE,
+    artifact_id TEXT NOT NULL,
+    job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
+    status TEXT NOT NULL,
+    provenance_json TEXT NOT NULL,
+    request_json TEXT NOT NULL,
+    report_json TEXT,
+    report_sha256 TEXT,
+    revision INTEGER NOT NULL DEFAULT 0,
+    review_json TEXT NOT NULL,
+    frozen_json TEXT,
+    export_sha256 TEXT,
+    error TEXT,
+    created_at TEXT NOT NULL,
+    updated_at TEXT NOT NULL
+);
+CREATE INDEX IF NOT EXISTS idx_ephys_qc_recording ON ephys_qc_runs(recording_id, created_at);
+CREATE TABLE IF NOT EXISTS ephys_qc_audit (
+    audit_id TEXT PRIMARY KEY,
+    run_id TEXT NOT NULL REFERENCES ephys_qc_runs(run_id) ON DELETE CASCADE,
+    revision INTEGER NOT NULL,
+    action TEXT NOT NULL,
+    reviewer TEXT,
+    snapshot_json TEXT NOT NULL,
+    created_at TEXT NOT NULL
+);
+"""
+
+
+def upgrade(conn):
+    for statement in SCHEMA.split(";"):
+        if statement.strip():
+            conn.execute(statement)
+    logging.getLogger(__name__).info("ephys_qc_v1 schema verified")
diff --git a/backend/app/migrations/ephys_recordings_v1.py b/backend/app/migrations/ephys_recordings_v1.py
new file mode 100644
index 0000000000000000000000000000000000000000..8c8778d69aada7e7c442d5de61dc6f2d09289219
--- /dev/null
+++ b/backend/app/migrations/ephys_recordings_v1.py
@@ -0,0 +1,35 @@
+"""Additive, repeatable S1 migration; never rewrites legacy assets or eeg_files."""
+
+
+def upgrade(conn):
+    conn.execute("""
+        CREATE TABLE IF NOT EXISTS ephys_recordings (
+            recording_id TEXT PRIMARY KEY,
+            subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
+            label TEXT NOT NULL,
+            modality TEXT NOT NULL,
+            session_id TEXT,
+            input_artifact_id TEXT NOT NULL UNIQUE,
+            import_key TEXT,
+            request_sha256 TEXT NOT NULL,
+            descriptor_json TEXT NOT NULL,
+            created_at TEXT NOT NULL,
+            UNIQUE(subject_id, import_key)
+        )
+    """)
+    conn.execute("""
+        CREATE TABLE IF NOT EXISTS ephys_artifacts (
+            artifact_id TEXT PRIMARY KEY,
+            recording_id TEXT NOT NULL UNIQUE
+                REFERENCES ephys_recordings(recording_id) ON DELETE CASCADE,
+            kind TEXT NOT NULL CHECK(kind = 'raw_input'),
+            storage_path TEXT NOT NULL UNIQUE,
+            content_sha256 TEXT NOT NULL,
+            manifest_sha256 TEXT NOT NULL,
+            file_count INTEGER NOT NULL,
+            size_bytes INTEGER NOT NULL,
+            created_at TEXT NOT NULL
+        )
+    """)
+    conn.execute("CREATE INDEX IF NOT EXISTS idx_ephys_recordings_subject "
+                 "ON ephys_recordings(subject_id)")
diff --git a/backend/app/processing/ephys_qc/__init__.py b/backend/app/processing/ephys_qc/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..16c325218d0826bfcf3eb2911320242cd1738e70
--- /dev/null
+++ b/backend/app/processing/ephys_qc/__init__.py
@@ -0,0 +1 @@
+"""Electrophysiology QC paths; legacy processing remains separate."""
diff --git a/backend/app/processing/ephys_qc/readers/__init__.py b/backend/app/processing/ephys_qc/readers/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..5bf22dbfe10bbec46f10d2b8055cfbcf7d793246
--- /dev/null
+++ b/backend/app/processing/ephys_qc/readers/__init__.py
@@ -0,0 +1,6 @@
+"""Readers that preserve stored samples and explicit acquisition clocks."""
+
+from .edf import EdfBdfReader, EphysReaderError
+from .npy import NpyVoltageReader
+
+__all__ = ["EdfBdfReader", "EphysReaderError", "NpyVoltageReader"]
diff --git a/backend/app/processing/ephys_qc/readers/edf.py b/backend/app/processing/ephys_qc/readers/edf.py
new file mode 100644
index 0000000000000000000000000000000000000000..5db10709d852eac5311cbf283ecb765fc45aa1df
--- /dev/null
+++ b/backend/app/processing/ephys_qc/readers/edf.py
@@ -0,0 +1,426 @@
+"""Windowed EDF/BDF decoding, without re-reference, filtering or resampling.
+
+Byte layouts and TAL conventions follow EDF+ and BDF+; executable byte-authored
+fixtures cover the conventions, including int24 sign, calibration and gaps.
+Only annotation blocks are scanned during construction. Numerical samples are
+read on demand. Clock storage grows with discontinuities, not signal length.
+"""
+import base64
+import copy
+from decimal import Decimal, InvalidOperation
+import logging
+import math
+from numbers import Integral
+import os
+from pathlib import Path
+import re
+
+import numpy as np
+
+
+class EphysReaderError(ValueError):
+    """Input cannot be interpreted under the explicitly supported contract."""
+
+
+_NUMBER = rb"[+-](?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
+_DURATION = rb"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
+_JOINED_TIMEKEEPING = re.compile(rb"^(" + _NUMBER + rb"\x14\x14)(?=" + _NUMBER + rb"[\x14\x15])")
+_UNIT_SCALES = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}
+_LOGGER = logging.getLogger(__name__)
+
+
+def _decimal(value, label):
+    try:
+        result = Decimal(value)
+    except (InvalidOperation, ValueError) as exc:
+        raise EphysReaderError("Invalid {}: {!r}".format(label, value)) from exc
+    if not result.is_finite():
+        raise EphysReaderError("Non-finite {}".format(label))
+    if not math.isfinite(float(result)):
+        raise EphysReaderError("{} exceeds supported float64 range".format(label))
+    return result
+
+
+def _integer(value, label):
+    number = _decimal(value, label)
+    if number != number.to_integral_value():
+        raise EphysReaderError("{} must be an integer".format(label))
+    return int(number)
+
+
+def _identity(stat):
+    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
+
+
+class EdfBdfReader:
+    """Immutable-source reader. Caller verifies the S1 content fingerprint.
+
+    Sample coordinates are cumulative *stored* samples in each rate stream.
+    Gaps split segments; they never insert samples. Physical output uses volts
+    only for recognized voltage units, retaining unknown physical units as-is.
+    """
+
+    def __init__(self, path, source_id):
+        if not isinstance(source_id, str) or not source_id:
+            raise EphysReaderError("source_id must be a stable non-empty string")
+        self.path = Path(path)
+        self.source_id = source_id
+        self._warnings = []
+        self._warning_index = {}
+        self._events = []
+        self._record_spans = []
+        self._streams = []
+        self._stream_lookup = {}
+        with self.path.open("rb") as handle:
+            self._signature = _identity(os.fstat(handle.fileno()))
+            self._read_header(handle)
+            self._index_time(handle)
+            self._assert_unchanged(handle)
+        self._build_streams()
+        self._map_events()
+        for warning in self._warnings:
+            _LOGGER.warning("ephys_reader_degraded code=%s occurrences=%s trigger=%s",
+                            warning["code"], warning["occurrences"], warning["first_context"])
+
+    def _warn(self, code, reason, **context):
+        # Aggregate repeat deviations, while retaining a concrete first location.
+        if code in self._warning_index:
+            self._warning_index[code]["occurrences"] += 1
+            return
+        warning = {"code": code, "severity": "degraded", "reason": reason,
+                   "occurrences": 1, "first_context": context}
+        self._warning_index[code] = warning
+        self._warnings.append(warning)
+
+    def _text(self, value):
+        try:
+            text = value.decode("ascii")
+        except UnicodeDecodeError:
+            try:
+                text = value.decode("utf-8")
+                encoding = "utf-8"
+            except UnicodeDecodeError:
+                text = value.decode("latin-1")
+                encoding = "latin-1-byte-preserving"
+            self._warn("header_non_ascii", "Header is not standard ASCII; original byte values are preserved.",
+                       decoding=encoding)
+        return text.rstrip(" ")
+
+    def _read_header(self, handle):
+        fixed = handle.read(256)
+        if len(fixed) != 256:
+            raise EphysReaderError("Truncated EDF/BDF fixed header")
+        if fixed[:8] == b"\xffBIOSEMI":
+            self._base_format, self._width = "BDF", 3
+        elif fixed[:8] == b"0       ":
+            self._base_format, self._width = "EDF", 2
+        else:
+            raise EphysReaderError("Unsupported EDF/BDF version signature")
+        self._header_bytes = _integer(self._text(fixed[184:192]), "header length")
+        self._reserved = self._text(fixed[192:236])
+        self._declared_records = _integer(self._text(fixed[236:244]), "record count")
+        self._duration = _decimal(self._text(fixed[244:252]), "record duration")
+        n_signals = _integer(self._text(fixed[252:256]), "signal count")
+        if not 1 <= n_signals <= 4096 or self._header_bytes != 256 * (n_signals + 1):
+            raise EphysReaderError("Invalid signal count or EDF/BDF header length")
+        if self._duration < 0 or self._declared_records < -1 or (self._duration and not float(self._duration)):
+            raise EphysReaderError("Negative record duration or invalid record count")
+        self._format = self._base_format
+        if self._reserved.startswith((self._base_format + "+C", self._base_format + "+D")):
+            self._format = self._reserved[:5]
+        elif self._reserved.startswith(("EDF+", "BDF+")):
+            raise EphysReaderError("Unsupported or contradictory EDF/BDF continuity marker")
+        self._clock = {"clock_id": self.source_id, "origin": "file_header_local_start",
+                       "startdate": self._text(fixed[168:176]), "starttime": self._text(fixed[176:184]),
+                       "timezone": "unknown", "cross_source_alignment": "unverified"}
+        raw = handle.read(self._header_bytes - 256)
+        if len(raw) != self._header_bytes - 256:
+            raise EphysReaderError("Truncated EDF/BDF signal header")
+        fields = [("raw_name", 16), ("transducer", 80), ("physical_unit", 8),
+                  ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
+                  ("digital_max", 8), ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]
+        self._channels = [{} for _ in range(n_signals)]
+        offset = 0
+        for field, width in fields:
+            for index in range(n_signals):
+                value = raw[offset + index * width:offset + (index + 1) * width]
+                self._channels[index][field] = self._text(value)
+            offset += width * n_signals
+        self._annotation_indices = []
+        self._channel_offsets = []
+        offset = 0
+        limit = 2 ** (self._width * 8 - 1)
+        for index, channel in enumerate(self._channels):
+            samples = _integer(channel["samples_per_record"], "samples per record")
+            if not 1 <= samples <= 100000000:
+                raise EphysReaderError("Invalid samples per record in channel {}".format(index))
+            channel["samples_per_record"] = samples
+            self._channel_offsets.append(offset)
+            offset += samples * self._width
+            channel.update(channel_id="{}:ch:{}".format(self.source_id, index),
+                           source_index=index, channel_type="unknown", classification_status="unreviewed")
+            if channel["raw_name"] in ("EDF Annotations", "BDF Annotations"):
+                if samples * self._width > 16 * 1024 * 1024:
+                    raise EphysReaderError("Annotation record exceeds supported 16 MiB limit")
+                self._annotation_indices.append(index)
+                continue
+            for key in ("physical_min", "physical_max", "digital_min", "digital_max"):
+                channel[key] = float(_decimal(channel[key], "channel {} {}".format(index, key)))
+            if (channel["digital_min"] < -limit or channel["digital_max"] >= limit or
+                    channel["digital_min"] >= channel["digital_max"] or
+                    channel["digital_min"] % 1 or channel["digital_max"] % 1 or
+                    channel["physical_min"] == channel["physical_max"]):
+                raise EphysReaderError("Invalid calibration in channel {}".format(index))
+            gain = ((channel["physical_max"] - channel["physical_min"]) /
+                    (channel["digital_max"] - channel["digital_min"]))
+            channel["digital_to_physical_scale"] = gain
+            channel["digital_to_physical_offset"] = channel["physical_min"] - gain * channel["digital_min"]
+            if not math.isfinite(gain) or not math.isfinite(channel["digital_to_physical_offset"]):
+                raise EphysReaderError("Calibration exceeds supported float64 range")
+            scale = _UNIT_SCALES.get(channel["physical_unit"])
+            channel.update(quantity="voltage" if scale is not None else "unknown",
+                           unit="V" if scale is not None else channel["physical_unit"],
+                           scale_to_si=scale, unit_status="known_voltage" if scale is not None else "unknown_quantity")
+            if scale is None:
+                self._warn("unknown_physical_quantity", "Physical values retain their source unit; voltage metrics are unavailable.",
+                           source_index=index, physical_unit=channel["physical_unit"])
+        self._record_bytes = offset
+        payload = self._signature[2] - self._header_bytes
+        if payload < 0 or payload % self._record_bytes:
+            raise EphysReaderError("File length does not contain complete EDF/BDF data records")
+        self._records = payload // self._record_bytes
+        if self._declared_records == -1:
+            self._warn("record_count_from_complete_payload", "Unspecified record count derived from exact complete payload length.",
+                       record_count=self._records)
+        elif self._records != self._declared_records:
+            raise EphysReaderError("File length disagrees with declared EDF/BDF record count")
+        if self._duration == 0 and len(self._annotation_indices) != len(self._channels):
+            raise EphysReaderError("Zero-duration numerical signals have no supported sampling frequency")
+        if "+" in self._format and not self._annotation_indices:
+            raise EphysReaderError("EDF+/BDF+ requires a timekeeping annotation channel")
+
+    def _parse_tals(self, raw, record, channel_index):
+        chunks = raw.split(b"\0")
+        parsed = []
+        byte_offset = 0
+        for chunk_index, original in enumerate(chunks):
+            if not original:
+                byte_offset += 1
+                continue
+            if chunk_index == len(chunks) - 1:
+                raise EphysReaderError("Unterminated TAL in record {}".format(record))
+            pieces = [(original, byte_offset)]
+            match = _JOINED_TIMEKEEPING.match(original) if not parsed else None
+            if match:
+                split = match.end()
+                first_onset = _decimal(original.split(b"\x14", 1)[0].decode("ascii"), "TAL onset")
+                candidate = original[split:].split(b"\x14", 1)[0].split(b"\x15", 1)[0]
+                next_onset = _decimal(candidate.decode("ascii"), "candidate TAL onset")
+                # A text annotation such as '+123' can be valid. Only offer the
+                # known vendor interpretation for an event inside this record;
+                # even then retain both interpretations because bytes alone
+                # cannot prove a NUL was omitted.
+                if not first_onset <= next_onset < first_onset + self._duration:
+                    match = None
+            if match:
+                pieces = [(original[:split], byte_offset), (original[split:], byte_offset + split)]
+                self._warn("timekeeping_missing_nul_separator",
+                           "Ambiguous vendor recovery: split first empty TAL from an in-record timestamp; the original bytes and standard interpretation remain available.",
+                           source_record_index=record, annotation_channel_index=channel_index)
+            for chunk, position in pieces:
+                parts = chunk.split(b"\x14")
+                if len(parts) < 3 or parts[-1] != b"":
+                    raise EphysReaderError("Malformed TAL delimiters in record {}".format(record))
+                stamp = parts[0].split(b"\x15")
+                if len(stamp) > 2 or re.fullmatch(_NUMBER, stamp[0]) is None:
+                    raise EphysReaderError("Invalid TAL onset in record {}".format(record))
+                if len(stamp) == 2 and re.fullmatch(_DURATION, stamp[1]) is None:
+                    raise EphysReaderError("Invalid TAL duration in record {}".format(record))
+                onset_text = stamp[0].decode("ascii")
+                duration_text = stamp[1].decode("ascii") if len(stamp) == 2 else None
+                alternative = None
+                if match:
+                    alternative = {"onset_seconds": float(first_onset), "onset_text": str(first_onset),
+                                   "annotation_texts_base64": [base64.b64encode(text).decode("ascii")
+                                                               for text in original.split(b"\x14")[1:-1] if text]}
+                parsed.append({"onset": _decimal(onset_text, "TAL onset"), "onset_text": onset_text,
+                               "duration_text": duration_text, "texts": parts[1:-1],
+                               "raw": chunk + b"\0", "raw_source": original + b"\0", "byte_offset": position,
+                               "interpretation_status": "vendor_recovered_ambiguous" if match else "standard",
+                               "alternative_standard_interpretation": alternative})
+            byte_offset += len(original) + 1
+        return parsed
+
+    def _index_time(self, handle):
+        if not self._annotation_indices:
+            if self._records:
+                self._record_spans = [{"start": 0, "stop": self._records, "onset": Decimal(0)}]
+            self._time_source = "continuous_file_format"
+            return
+        self._time_source = "per_record_TAL"
+        if "+" not in self._format:
+            self._warn("annotations_without_plus_marker", "TAL timestamps used although header lacks a plus-format marker.")
+        previous_onset = None
+        for record in range(self._records):
+            record_onset = None
+            for channel_index in self._annotation_indices:
+                channel = self._channels[channel_index]
+                offset = self._header_bytes + record * self._record_bytes + self._channel_offsets[channel_index]
+                handle.seek(offset)
+                tals = self._parse_tals(handle.read(channel["samples_per_record"] * self._width), record, channel_index)
+                if channel_index == self._annotation_indices[0]:
+                    if not tals or not tals[0]["texts"] or tals[0]["texts"][0] != b"" or tals[0]["duration_text"] is not None:
+                        raise EphysReaderError("Missing first timekeeping TAL in record {}".format(record))
+                    record_onset = tals[0]["onset"]
+                for tal_index, tal in enumerate(tals):
+                    for annotation_index, raw_text in enumerate(tal["texts"]):
+                        if not raw_text:
+                            continue
+                        try:
+                            text = raw_text.decode("utf-8")
+                            encoding_status = "utf-8"
+                        except UnicodeDecodeError:
+                            text, encoding_status = None, "undecoded_non_utf8"
+                            self._warn("annotation_non_utf8", "Annotation is not UTF-8; original bytes retained without guessing an encoding.",
+                                       source_record_index=record, annotation_channel_index=channel_index)
+                        if len(self._events) >= 1000000:
+                            raise EphysReaderError("Event count exceeds supported metadata limit of 1000000")
+                        self._events.append({"event_id": "{}:ann:{}:{}:{}:{}".format(self.source_id, channel_index, record, tal_index, annotation_index),
+                                             "source_id": self.source_id, "clock_id": self.source_id,
+                                             "annotation_stream_id": "{}:annotations:{}".format(self.source_id, channel_index),
+                                             "source_record_index": record, "annotation_channel_index": channel_index,
+                                             "tal_index": tal_index, "annotation_index": annotation_index,
+                                             "source_byte_offset_in_annotation_record": tal["byte_offset"],
+                                             "onset_text": tal["onset_text"], "onset_seconds": float(tal["onset"]),
+                                             "duration_text": tal["duration_text"],
+                                             "duration_seconds": float(tal["duration_text"]) if tal["duration_text"] is not None else None,
+                                             "text": text, "encoding_status": encoding_status,
+                                             "interpretation_status": tal["interpretation_status"],
+                                             "alternative_standard_interpretation": tal["alternative_standard_interpretation"],
+                                             "raw_text_base64": base64.b64encode(raw_text).decode("ascii"),
+                                             "raw_tal_base64": base64.b64encode(tal["raw"]).decode("ascii"),
+                                             "raw_source_tal_base64": base64.b64encode(tal["raw_source"]).decode("ascii")})
+            if previous_onset is not None and record_onset < previous_onset + self._duration:
+                raise EphysReaderError("Overlapping or backward data record timestamps at record {}".format(record))
+            contiguous = previous_onset is not None and record_onset == previous_onset + self._duration
+            if contiguous:
+                self._record_spans[-1]["stop"] = record + 1
+            else:
+                self._record_spans.append({"start": record, "stop": record + 1, "onset": record_onset})
+                if previous_onset is not None and self._format.endswith("+C"):
+                    self._warn("continuous_flag_has_gaps", "Header says continuous; per-record TAL timestamps contain gaps, which remain separate segments.",
+                               source_record_index=record)
+            previous_onset = record_onset
+
+    def _build_streams(self):
+        for index, channel in enumerate(self._channels):
+            if index in self._annotation_indices:
+                continue
+            samples = channel["samples_per_record"]
+            stream_id = "{}:rate:{}".format(self.source_id, samples)
+            if stream_id not in self._stream_lookup:
+                segments = [{"segment_id": "{}:segment:{}".format(stream_id, span_index),
+                             "start_sample": span["start"] * samples, "stop_sample": span["stop"] * samples,
+                             "recording_start_seconds": float(span["onset"]),
+                             "recording_start_seconds_text": str(span["onset"]),
+                             "source_start_record": span["start"], "source_stop_record": span["stop"]}
+                            for span_index, span in enumerate(self._record_spans)]
+                stream = {"stream_id": stream_id, "source_id": self.source_id, "clock_id": self.source_id,
+                          "sampling_frequency_hz": float(Decimal(samples) / self._duration),
+                          "samples_per_record": samples, "sample_index_origin": 0,
+                          "sample_count": self._records * samples, "channels": [], "segments": segments,
+                          "time_source": self._time_source, "reference": "as_stored",
+                          "processing_applied": [], "channel_order": "source_header_within_rate_group"}
+                if not math.isfinite(stream["sampling_frequency_hz"]) or stream["sampling_frequency_hz"] <= 0:
+                    raise EphysReaderError("Sampling frequency exceeds supported float64 range")
+                self._streams.append(stream)
+                self._stream_lookup[stream_id] = stream
+            self._stream_lookup[stream_id]["channels"].append(channel)
+
+    def _map_events(self):
+        for event in self._events:
+            onset = Decimal(event["onset_text"])
+            mappings = []
+            for stream in self._streams:
+                samples = stream["samples_per_record"]
+                for span_index, span in enumerate(self._record_spans):
+                    duration = (span["stop"] - span["start"]) * self._duration
+                    if span["onset"] <= onset < span["onset"] + duration:
+                        position = span["start"] * samples + (onset - span["onset"]) * samples / self._duration
+                        mappings.append({"stream_id": stream["stream_id"],
+                                         "segment_id": stream["segments"][span_index]["segment_id"],
+                                         "sample_position": float(position),
+                                         "sample_index": int(position) if position == position.to_integral_value() else None,
+                                         "interpretation_status": event["interpretation_status"],
+                                         "alignment": "exact_sample" if position == position.to_integral_value() else "between_samples"})
+                        break
+            event["sample_mappings"] = mappings
+
+    def _assert_unchanged(self, handle=None):
+        current = os.fstat(handle.fileno()) if handle is not None else self.path.stat()
+        if _identity(current) != self._signature or (handle is not None and _identity(self.path.stat()) != self._signature):
+            raise EphysReaderError("Source file changed since reader construction; reverify input fingerprint")
+
+    def describe(self):
+        self._assert_unchanged()
+        return copy.deepcopy({"reader_version": "edf-bdf-1", "source_id": self.source_id,
+                              "format": self._format, "clock": self._clock,
+                              "header_channel_count": len(self._channels), "record_count": self._records,
+                              "record_duration_seconds": float(self._duration),
+                              "annotation_channel_indices": self._annotation_indices,
+                              "streams": self._streams, "events": self._events, "warnings": self._warnings,
+                              "processing_applied": [], "scientific_validity": "pending"})
+
+    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
+        self._assert_unchanged()
+        if representation not in ("physical", "digital"):
+            raise EphysReaderError("representation must be physical or digital")
+        if (isinstance(start_sample, bool) or isinstance(stop_sample, bool) or
+                not isinstance(start_sample, Integral) or not isinstance(stop_sample, Integral)):
+            raise EphysReaderError("Sample bounds must be integers")
+        if not isinstance(stream_id, str) or not isinstance(segment_id, str):
+            raise EphysReaderError("Stream and segment IDs must be strings")
+        stream = self._stream_lookup.get(stream_id)
+        if stream is None:
+            raise EphysReaderError("Unknown stream_id")
+        segment = next((span for span in stream["segments"] if span["segment_id"] == segment_id), None)
+        if (segment is None or start_sample < segment["start_sample"] or stop_sample > segment["stop_sample"] or
+                start_sample >= stop_sample):
+            raise EphysReaderError("Window must be a non-empty half-open interval within the selected segment")
+        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or
+                any(not isinstance(channel_id, str) for channel_id in channel_ids) or
+                len(set(channel_ids)) != len(channel_ids)):
+            raise EphysReaderError("channel_ids must be a non-empty list without duplicates")
+        lookup = {channel["channel_id"]: channel for channel in stream["channels"]}
+        if any(channel_id not in lookup for channel_id in channel_ids):
+            raise EphysReaderError("Channel is not part of the selected stream")
+        channels = [lookup[channel_id] for channel_id in channel_ids]
+        if len(channels) * (stop_sample - start_sample) * 8 > 256 * 1024 * 1024:
+            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
+        output = np.empty((len(channels), stop_sample - start_sample), dtype=np.float64)
+        samples = stream["samples_per_record"]
+        with self.path.open("rb") as handle:
+            self._assert_unchanged(handle)
+            for row, channel in enumerate(channels):
+                for record in range(start_sample // samples, (stop_sample - 1) // samples + 1):
+                    first = max(start_sample, record * samples)
+                    last = min(stop_sample, (record + 1) * samples)
+                    byte_offset = (self._header_bytes + record * self._record_bytes +
+                                   self._channel_offsets[channel["source_index"]] + (first - record * samples) * self._width)
+                    handle.seek(byte_offset)
+                    raw = handle.read((last - first) * self._width)
+                    if len(raw) != (last - first) * self._width:
+                        raise EphysReaderError("Source truncated during window read")
+                    if self._width == 2:
+                        values = np.frombuffer(raw, dtype="<i2")
+                    else:
+                        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
+                        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
+                        values = (values ^ 0x800000) - 0x800000
+                    output[row, first - start_sample:last - start_sample] = values
+                if representation == "physical":
+                    output[row] *= channel["digital_to_physical_scale"]
+                    output[row] += channel["digital_to_physical_offset"]
+                    output[row] *= channel["scale_to_si"] if channel["scale_to_si"] is not None else 1.0
+            self._assert_unchanged(handle)
+        return output
diff --git a/backend/app/processing/ephys_qc/readers/npy.py b/backend/app/processing/ephys_qc/readers/npy.py
new file mode 100644
index 0000000000000000000000000000000000000000..b10568f25ccf002a618b4ff122e282dadb9ee571
--- /dev/null
+++ b/backend/app/processing/ephys_qc/readers/npy.py
@@ -0,0 +1,162 @@
+"""Read versioned derived voltages through bounded, read-only NumPy mappings."""
+from copy import deepcopy
+import math
+from numbers import Integral, Real
+from pathlib import Path
+
+import numpy as np
+
+from .edf import EphysReaderError, _identity
+
+
+def _integer(value, label, minimum=0):
+    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum or value > 2 ** 63 - 1:
+        raise EphysReaderError("{} must be an integer >= {}".format(label, minimum))
+    return int(value)
+
+
+def _text(value, label):
+    if not isinstance(value, str) or not value:
+        raise EphysReaderError("{} must be an explicit non-empty string".format(label))
+    return value
+
+
+def _number(value, label):
+    if isinstance(value, bool) or not isinstance(value, Real):
+        raise EphysReaderError("{} must be a finite number".format(label))
+    try:
+        result = float(value)
+    except (OverflowError, ValueError) as exc:
+        raise EphysReaderError("{} exceeds float64 range".format(label)) from exc
+    if not math.isfinite(result):
+        raise EphysReaderError("{} must be a finite number".format(label))
+    return result
+
+
+class NpyVoltageReader:
+    """Only C-order little-endian float64 volts, with explicit parent mappings.
+
+    Construction validates the descriptor and file layout, without scanning
+    samples. Finite values are checked on the requested window. Publication is
+    responsible for the complete provenance and the artifact's content hash.
+    """
+
+    def __init__(self, path, stream):
+        self.path = Path(path)
+        self._stream = deepcopy(stream)
+        try:
+            self._validate_stream()
+        except (KeyError, TypeError) as exc:
+            raise EphysReaderError("Incomplete derived stream descriptor: {}".format(exc)) from exc
+        self._signature = _identity(self.path.stat())
+        try:
+            self._data = np.load(self.path, mmap_mode="r", allow_pickle=False)
+        except (OSError, ValueError, EOFError) as exc:
+            raise EphysReaderError("Cannot map derived voltage NPY: {}".format(exc)) from exc
+        if not isinstance(self._data, np.memmap):
+            if hasattr(self._data, "close"):
+                self._data.close()
+            raise EphysReaderError("Derived voltage must be a single NPY array")
+        expected = (len(self._stream["channels"]), self._stream["sample_count"])
+        if (self._data.dtype.str != "<f8" or self._data.shape != expected or
+                not self._data.flags.c_contiguous):
+            raise EphysReaderError("Derived voltage requires C-contiguous <f8 with shape channel x stored sample")
+        if self._data.offset + self._data.nbytes != self._signature[2]:
+            raise EphysReaderError("NPY payload length does not exactly match its array shape")
+        self._assert_unchanged()
+
+    def _validate_stream(self):
+        stream = self._stream
+        if not isinstance(stream, dict):
+            raise EphysReaderError("Derived stream descriptor must be an object")
+        for name in ("stream_id", "source_id", "clock_id"):
+            _text(stream[name], name)
+        frequency = _number(stream["sampling_frequency_hz"], "sampling_frequency_hz")
+        if frequency <= 0:
+            raise EphysReaderError("sampling_frequency_hz must be positive")
+        origin = _integer(stream["sample_index_origin"], "sample_index_origin")
+        count = _integer(stream["sample_count"], "sample_count", 1)
+        channels = stream["channels"]
+        if not isinstance(channels, list) or not channels:
+            raise EphysReaderError("Derived channels must be a non-empty ordered list")
+        seen = set()
+        for channel in channels:
+            channel_id = _text(channel["channel_id"], "channel_id")
+            if channel_id in seen or channel["unit"] != "V" or not isinstance(channel["raw_name"], str):
+                raise EphysReaderError("Derived channels require unique IDs, preserved names and explicit SI unit V")
+            seen.add(channel_id)
+            parents = channel["source_channel_ids"]
+            if not isinstance(parents, list) or not parents:
+                raise EphysReaderError("Derived channel requires explicit source_channel_ids")
+            for parent in parents:
+                _text(parent, "source_channel_id")
+            if len(set(parents)) != len(parents):
+                raise EphysReaderError("Duplicate source_channel_ids are ambiguous")
+        segments = stream["segments"]
+        if not isinstance(segments, list) or not segments:
+            raise EphysReaderError("Derived segments must be a non-empty list")
+        next_sample, previous_time_stop, segment_ids = origin, None, set()
+        for segment in segments:
+            segment_id = _text(segment["segment_id"], "segment_id")
+            start = _integer(segment["start_sample"], "start_sample")
+            stop = _integer(segment["stop_sample"], "stop_sample")
+            onset = _number(segment["recording_start_seconds"], "recording_start_seconds")
+            for field in ("parent_stream_id", "parent_segment_id"):
+                _text(segment[field], field)
+            parent_start = _integer(segment["parent_start_sample"], "parent_start_sample")
+            parent_stop = _integer(segment["parent_stop_sample"], "parent_stop_sample")
+            if (segment_id in segment_ids or start != next_sample or stop <= start or
+                    parent_stop - parent_start != stop - start or segment["clock_mapping"] != "identity"):
+                raise EphysReaderError("Segments must exactly cover stored samples with explicit identity parent mappings")
+            if previous_time_stop is not None and onset < previous_time_stop and not math.isclose(onset, previous_time_stop, rel_tol=0, abs_tol=1e-12):
+                raise EphysReaderError("Derived segment clocks overlap or move backward")
+            segment_ids.add(segment_id)
+            next_sample = stop
+            previous_time_stop = onset + (stop - start) / frequency
+            if not math.isfinite(previous_time_stop):
+                raise EphysReaderError("Derived segment clock exceeds float64 range")
+        if next_sample != origin + count:
+            raise EphysReaderError("Segments do not cover the declared stored sample count")
+
+    def _assert_unchanged(self):
+        try:
+            unchanged = _identity(self.path.stat()) == self._signature
+        except OSError as exc:
+            raise EphysReaderError("Derived voltage file changed or disappeared") from exc
+        if not unchanged:
+            raise EphysReaderError("Derived voltage file changed; reverify artifact fingerprint")
+
+    def describe(self):
+        self._assert_unchanged()
+        return {"format": "NPY_FLOAT64_VOLTAGE", "source_id": self._stream["source_id"],
+                "streams": [deepcopy(self._stream)], "events": [], "warnings": [],
+                "value_validation": "finite_per_requested_window", "scientific_validity": "pending"}
+
+    def read_window(self, stream_id, segment_id, channel_ids, start_sample, stop_sample, representation="physical"):
+        self._assert_unchanged()
+        if representation != "physical":
+            raise EphysReaderError("Derived voltages support physical representation only; ADC codes are unavailable")
+        start = _integer(start_sample, "start_sample")
+        stop = _integer(stop_sample, "stop_sample")
+        if stream_id != self._stream["stream_id"]:
+            raise EphysReaderError("Unknown derived stream_id")
+        segment = next((s for s in self._stream["segments"] if s["segment_id"] == segment_id), None)
+        if segment is None or not segment["start_sample"] <= start < stop <= segment["stop_sample"]:
+            raise EphysReaderError("Window must be a non-empty half-open interval within the selected segment")
+        if (not isinstance(channel_ids, (list, tuple)) or not channel_ids or
+                any(not isinstance(c, str) for c in channel_ids) or len(set(channel_ids)) != len(channel_ids)):
+            raise EphysReaderError("Select distinct ordered channel IDs")
+        lookup = {channel["channel_id"]: index for index, channel in enumerate(self._stream["channels"])}
+        if any(c not in lookup for c in channel_ids):
+            raise EphysReaderError("Channel is not part of the selected derived stream")
+        if len(channel_ids) * (stop - start) * 8 > 256 * 1024 * 1024:
+            raise EphysReaderError("Window exceeds 256 MiB; request smaller windows")
+        origin = self._stream["sample_index_origin"]
+        # Advanced row indexing copies only the bounded column slice, never all
+        # stored samples. Return an owned C-order array; callers cannot edit input.
+        result = np.array(self._data[[lookup[c] for c in channel_ids], start - origin:stop - origin],
+                          dtype="<f8", order="C", copy=True)
+        if not np.isfinite(result).all():
+            raise EphysReaderError("Requested derived voltage window contains non-finite values")
+        self._assert_unchanged()
+        return result
diff --git a/backend/app/processing/ephys_qc/seeg.py b/backend/app/processing/ephys_qc/seeg.py
new file mode 100644
index 0000000000000000000000000000000000000000..b26055d3529869bf7d4780f66e56ffa334590740
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg.py
@@ -0,0 +1,261 @@
+"""Bounded, segment-preserving SEEG basic QC scan over the verified catalog."""
+from collections import defaultdict
+from copy import deepcopy
+import hashlib
+import math
+from numbers import Real
+
+import numpy as np
+
+from .seeg_channels import resolve_seeg_channels
+from .seeg_metrics import compute_relations, compute_window_metrics
+from .seeg_scoring import compute_quality_score, score_window
+from .seeg_waveform import (WAVEFORM_DEFAULTS, WAVEFORM_VERSION, WaveformTracker,
+                            attach_waveform_events, preflight_waveform, validate_waveform_config)
+
+
+SEEG_ALGORITHM_VERSION = "seeg-basic-2"
+_DEFAULTS = {**WAVEFORM_DEFAULTS, "window_seconds": 5.0, "line_frequency": 50, "flat_tolerance_v": 0.0,
+             "flat_min_seconds": .5, "clipping_fraction": .01, "clipping_min_seconds": .02,
+             "line_noise_db": 12.0, "line_noise_fraction": .05, "max_windows": 20000,
+             "max_scanned_samples": 100000000, "max_window_bytes": 16777216,
+             "max_relation_evaluations": 1000000, "ranges": None}
+
+
+def _config(config):
+    if config is None:
+        config = {}
+    if not isinstance(config, dict) or set(config) - set(_DEFAULTS):
+        raise ValueError("Unknown SEEG QC configuration fields")
+    result = {**_DEFAULTS, **deepcopy(config)}
+    for key, value in result.items():
+        if key == "ranges":
+            continue
+        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
+            raise ValueError("{} must be a finite number".format(key))
+        if value < 0 or (key != "flat_tolerance_v" and value == 0):
+            raise ValueError("{} must be positive".format(key))
+    if result["line_frequency"] not in (50, 60):
+        raise ValueError("line_frequency must be 50 or 60")
+    for key in ("clipping_fraction", "line_noise_fraction"):
+        if result[key] > 1:
+            raise ValueError("{} must be in (0,1]".format(key))
+    for key in ("max_windows", "max_scanned_samples", "max_window_bytes", "max_relation_evaluations"):
+        if int(result[key]) != result[key] or result[key] > _DEFAULTS[key]:
+            raise ValueError("{} must be an integer within the supported resource limit {}".format(key, _DEFAULTS[key]))
+    if result["window_seconds"] > 60:
+        raise ValueError("window_seconds must be <=60; select ranges for long recordings")
+    validate_waveform_config(result)
+    return result
+
+
+def _integer(value, name):
+    if isinstance(value, bool) or not isinstance(value, int):
+        raise ValueError("{} must be an integer".format(name))
+    return value
+
+
+def _plan(catalog, config):
+    streams = catalog.get("streams")
+    if not isinstance(streams, list) or not streams:
+        raise ValueError("No readable streams in catalog")
+    lookup, spans, gaps = {}, {}, []
+    total = 0.0
+    for stream in streams:
+        sid, fs = stream["stream_id"], stream["sampling_frequency_hz"]
+        if not isinstance(sid, str) or not sid or sid in lookup:
+            raise ValueError("Streams require unique nonempty IDs")
+        if isinstance(fs, bool) or not isinstance(fs, Real) or not math.isfinite(fs) or fs <= 0:
+            raise ValueError("sampling_frequency_hz must be finite and positive")
+        if not isinstance(stream["channels"], list) or not stream["channels"]:
+            raise ValueError("A stream requires channels")
+        if round(config["window_seconds"] * fs) < 2:
+            raise ValueError("window_seconds must contain at least two samples")
+        lookup[sid] = stream
+        previous_stop, previous_time_stop = None, None
+        for segment in stream["segments"]:
+            key = (sid, segment["segment_id"])
+            start, stop = _integer(segment["start_sample"], "start_sample"), _integer(segment["stop_sample"], "stop_sample")
+            onset = segment["recording_start_seconds"]
+            if key in spans or start < 0 or stop <= start or (previous_stop is not None and start < previous_stop):
+                raise ValueError("Segment samples must be distinct, ordered nonempty intervals")
+            if isinstance(onset, bool) or not isinstance(onset, Real) or not math.isfinite(onset):
+                raise ValueError("Segment recording_start_seconds must be finite")
+            if previous_time_stop is not None and onset < previous_time_stop - 1e-9:
+                raise ValueError("Segment clocks overlap")
+            if previous_time_stop is not None and onset > previous_time_stop + 1e-9:
+                gaps.append({"stream_id": sid, "start_seconds": previous_time_stop,
+                             "stop_seconds": onset, "duration_seconds": onset - previous_time_stop})
+            previous_stop, previous_time_stop = stop, onset + (stop - start) / fs
+            spans[key] = segment
+            total += (stop - start) / fs * len(stream["channels"])
+    ranges = config["ranges"]
+    if ranges is None:
+        ranges = [{"stream_id": sid, "segment_id": segment_id, "start_sample": span["start_sample"],
+                   "stop_sample": span["stop_sample"]} for (sid, segment_id), span in spans.items()]
+    if not isinstance(ranges, list) or not ranges:
+        raise ValueError("ranges must be a nonempty list or null")
+    occupied, plan = defaultdict(list), []
+    windows, samples, evaluations = 0, 0, 0
+    for item in ranges:
+        fields = {"stream_id", "segment_id", "start_sample", "stop_sample"}
+        if not isinstance(item, dict) or set(item) != fields:
+            raise ValueError("Each range requires stream_id/segment_id/start_sample/stop_sample")
+        key = (item["stream_id"], item["segment_id"])
+        span = spans.get(key)
+        start, stop = _integer(item["start_sample"], "start_sample"), _integer(item["stop_sample"], "stop_sample")
+        if span is None or not span["start_sample"] <= start < stop <= span["stop_sample"]:
+            raise ValueError("Range must stay inside a known segment")
+        if any(start < b and stop > a for a, b in occupied[key]):
+            raise ValueError("Scan ranges must not overlap")
+        occupied[key].append((start, stop))
+        stream = lookup[key[0]]
+        size = int(round(config["window_seconds"] * stream["sampling_frequency_hz"]))
+        count = len(stream["channels"])
+        nwin = math.ceil((stop - start) / size)
+        windows += nwin * count
+        samples += (stop - start) * count
+        evaluations += nwin * count * (count - 1) // 2
+        if min(size, stop - start) * count * 8 > config["max_window_bytes"]:
+            raise ValueError("Scan window exceeds max_window_bytes; reduce window_seconds")
+        plan.append((stream, span, start, stop, size))
+    if windows > config["max_windows"] or samples > config["max_scanned_samples"] or evaluations > config["max_relation_evaluations"]:
+        raise ValueError("Scan resource limit exceeded; select explicit ranges or a shorter recording")
+    return plan, deepcopy(ranges), gaps, total
+
+
+def scan_seeg(catalog, read_window, channel_map=None, config=None):
+    """Compute JSON-safe evidence; read_window returns channel × physical sample."""
+    config = _config(config)
+    plan, ranges, gaps, total = _plan(catalog, config)
+    preflight_waveform(plan, config)
+    plan.sort(key=lambda p: (p[0]["stream_id"], p[1]["start_sample"], p[2]))
+    mapping = resolve_seeg_channels(catalog, channel_map)
+    by_id = {item["channel_id"]: item for item in mapping}
+    windows, relations, spectra, waveform_events = [], [], {}, []
+    trackers = {}
+    for plan_index, (stream, segment, start, stop, size) in enumerate(plan):
+        channels = stream["channels"]
+        ids = [channel["channel_id"] for channel in channels]
+        mapped = [by_id[cid] for cid in ids]
+        frequency = stream["sampling_frequency_hz"]
+        if not trackers:
+            for row, channel in enumerate(channels):
+                item = mapped[row]
+                kind = item["channel_type"]
+                auxiliary = kind == "unknown" and item["candidate"]["channel_type"] in ("ecg", "eog", "emg", "stim", "misc")
+                if channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage" and kind in ("unknown", "seeg") and not auxiliary:
+                    context = {"channel_id": ids[row], "stream_id": stream["stream_id"], "segment_id": segment["segment_id"],
+                               "recording_start_seconds": segment["recording_start_seconds"], "segment_start_sample": segment["start_sample"]}
+                    trackers[row] = WaveformTracker(frequency, context, config)
+        for first in range(start, stop, size):
+            last = min(stop, first + size)
+            data = np.asarray(read_window(stream["stream_id"], segment["segment_id"], ids, first, last,
+                                          representation="physical"), dtype=np.float64)
+            if data.shape != (len(ids), last - first):
+                raise ValueError("Reader shape must be channel × requested samples")
+            for row, channel in enumerate(channels):
+                try:
+                    with np.errstate(over="raise", invalid="raise", divide="raise"):
+                        metrics, flags, spectrum = compute_window_metrics(data[row], frequency, channel, config)
+                except FloatingPointError as exc:
+                    raise ValueError("Numerical overflow in channel {}; verify physical scaling".format(ids[row])) from exc
+                item = mapped[row]
+                kind = item["channel_type"]
+                # C1/F3-style names occur in both scalp and depth recordings. A
+                # name alone may suggest modality, but cannot settle this gate.
+                if kind == "unknown" and item["candidate"]["channel_type"] in ("ecg", "eog", "emg", "stim", "misc"):
+                    kind = item["candidate"]["channel_type"]
+                eligible = channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage" and kind in ("unknown", "seeg") and last - first >= 2
+                identity = "{}\0{}\0{}\0{}\0{}".format(stream["stream_id"], segment["segment_id"], ids[row], first, last)
+                window = {"window_id": "sw-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
+                          "channel_id": ids[row], "stream_id": stream["stream_id"], "segment_id": segment["segment_id"],
+                          "start_sample": first, "stop_sample": last, "duration_seconds": (last - first) / frequency,
+                          "recording_start_seconds": segment["recording_start_seconds"] + (first - segment["start_sample"]) / frequency,
+                          "metrics": metrics, "flags": flags, "score": score_window(flags, eligible), "score_eligible": eligible,
+                          "status": "computed" if eligible else "not_applicable" if kind not in ("unknown", "seeg") else "insufficient_data"}
+                windows.append(window)
+                if row in trackers:
+                    trackers[row].feed(data[row], first)
+                _accumulate_spectrum(spectra, ids[row], stream["stream_id"], spectrum, window["duration_seconds"])
+            relations.extend(compute_relations(data, mapped, stream["stream_id"], segment["segment_id"], first, last))
+        next_plan = plan[plan_index + 1] if plan_index + 1 < len(plan) else None
+        contiguous = next_plan is not None and next_plan[0]["stream_id"] == stream["stream_id"] and next_plan[1]["segment_id"] == segment["segment_id"] and next_plan[2] == stop
+        if not contiguous:
+            for tracker in trackers.values():
+                tracker.flush("segment_end" if stop == segment["stop_sample"] else "range_end")
+                waveform_events.extend(tracker.events)
+            trackers = {}
+            if len(waveform_events) > config["max_waveform_events"]:
+                raise ValueError("Waveform event limit exceeded; select a shorter scan range")
+    attach_waveform_events(windows, waveform_events)
+    scanned = sum(w["duration_seconds"] for w in windows)
+    channels = _summarize_channels(mapping, windows)
+    for spectrum in spectra.values():
+        weight = spectrum.pop("_weight")
+        power = spectrum.pop("_power")
+        spectrum["power_v2_hz"] = (power / weight).tolist() if weight else []
+    return {"schema_version": 1, "profile": "seeg_basic_v1", "algorithm_version": SEEG_ALGORITHM_VERSION,
+            "detection_version": WAVEFORM_VERSION, "evidence_only_flags": ["boundary_dwell", "waveform_anomaly"],
+            "config": config, "channel_map": mapping, "integrity": _integrity(catalog, mapping, gaps),
+            "windows": windows, "channels": channels, "spectra": list(spectra.values()), "relations": relations,
+            "waveform_events": waveform_events,
+            "initial_score": compute_quality_score(windows),
+            "coverage": {"scope": "full_recording" if config["ranges"] is None else "selected_ranges", "ranges": ranges,
+                         "scanned_channel_seconds": scanned, "total_channel_seconds": total,
+                         "scanned_fraction": scanned / total if total else None,
+                         "window_count": len(windows), "clock_policy": "within_source_segments_only",
+                         "denominator": "readable_stream_channel_seconds",
+                         "complete_input": not bool(catalog.get("errors")),
+                         "unreadable_source_count": len(catalog.get("errors", []))}}
+
+
+def _accumulate_spectrum(spectra, channel_id, stream_id, result, duration):
+    item = spectra.setdefault(channel_id, {"channel_id": channel_id, "stream_id": stream_id,
+                                         "status": "insufficient_data", "frequency_hz": [], "window_count": 0,
+                                         "duration_seconds": 0.0, "_weight": 0.0, "_power": None,
+                                         "method": "duration_weighted_mean_window_welch", "reason": result.get("reason")})
+    if result["status"] != "computed":
+        return
+    power = np.asarray(result["power_v2_hz"])
+    if item["_power"] is None:
+        item["_power"] = np.zeros_like(power)
+        item["frequency_hz"] = result["frequency_hz"]
+    item["_power"] += power * duration
+    item["_weight"] += duration
+    item["duration_seconds"] += duration
+    item["window_count"] += 1
+    item.update(status="computed", reason=None)
+
+
+def _summarize_channels(mapping, windows):
+    grouped = defaultdict(list)
+    for window in windows:
+        grouped[window["channel_id"]].append(window)
+    channels = []
+    for item in mapping:
+        subset = grouped[item["channel_id"]]
+        duration = sum(w["duration_seconds"] for w in subset)
+        flagged = sum(w["duration_seconds"] for w in subset if w["flags"])
+        channels.append({"channel_id": item["channel_id"], "raw_name": item["raw_name"],
+                         "stream_id": item["stream_id"], "scanned_seconds": duration,
+                         "flagged_seconds": flagged, "flagged_fraction": flagged / duration if duration else None,
+                         "line_noise_seconds": sum(w["duration_seconds"] for w in subset if "line_noise" in w["flags"]),
+                         "flags": sorted({flag for w in subset for flag in w["flags"]}),
+                         "score": compute_quality_score(subset)["total"], "status": "candidate_review_required"})
+    return channels
+
+
+def _integrity(catalog, mapping, gaps):
+    unknown = [m["channel_id"] for m in mapping if m["unit"] != "V"]
+    return {"status": "computed", "gaps": gaps, "unknown_voltage_unit_channel_ids": unknown,
+            "mapping_pending_channel_ids": [m["channel_id"] for m in mapping if m["status"] != "confirmed"],
+            "sources": deepcopy(catalog.get("sources", [])), "source_errors": deepcopy(catalog.get("errors", [])),
+            "processing_history": deepcopy(catalog.get("processing", [])),
+            "streams": [{"stream_id": s["stream_id"], "sampling_frequency_hz": s["sampling_frequency_hz"],
+                         "segment_count": len(s["segments"]), "channel_count": len(s["channels"]),
+                         "nyquist_hz": s["sampling_frequency_hz"] / 2,
+                         "prefilter_by_channel": {c["channel_id"]: c.get("prefilter") for c in s["channels"]},
+                         "processing_applied": deepcopy(s.get("processing_applied", [])),
+                         "reference": s.get("reference"), "clock_id": s.get("clock_id")}
+                        for s in catalog["streams"]]}
diff --git a/backend/app/processing/ephys_qc/seeg_channels.py b/backend/app/processing/ephys_qc/seeg_channels.py
new file mode 100644
index 0000000000000000000000000000000000000000..c1829a3d42cbee604bfc023813531cde8883bdb1
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg_channels.py
@@ -0,0 +1,157 @@
+"""Preserved channel identities and reviewable SEEG naming candidates."""
+from collections import defaultdict
+from copy import deepcopy
+import re
+
+
+_CONTACT = re.compile(r"^([^\d\s\-]+)(\d+)$")
+_SCALP = re.compile(r"^(?:FP|AF|F|FT|FC|C|T|TP|CP|P|PO|O)(?:Z|\d+)$", re.I)
+_AUX = re.compile(r"^(ECG|EKG|EOG|EMG|TRIG|TRIGGER|STIM|STATUS)(?:\b|\d|[_-])", re.I)
+_TYPES = {"unknown", "seeg", "eeg", "ecog", "ecg", "eog", "emg", "stim", "misc"}
+_FIELDS = {"channel_id", "channel_type", "status", "shaft", "contact_index", "measurement_kind",
+           "contact_names", "reference", "source_channels"}
+
+
+def _contact(name):
+    match = _CONTACT.fullmatch(name)
+    return (match.group(1), int(match.group(2))) if match else None
+
+
+def _candidate(name):
+    result = {"shaft": None, "contact_index": None, "measurement_kind": "unknown",
+              "contact_names": [], "channel_type": "unknown", "reference": None, "name_adapter": None}
+    text = name.strip()
+    # These vendor labels describe acquisition slots, not biological modality.
+    stripped = re.sub(r"^(?:EEG|POL)\s+", "", text, flags=re.I)
+    if stripped != text:
+        result["name_adapter"] = "vendor_prefix_candidate_only"
+        text = stripped
+    if text.lower().endswith("-ref"):
+        result["reference"] = text[-3:]
+        text = text[:-4]
+    aux = _AUX.match(text)
+    if aux:
+        kind = aux.group(1).lower()
+        result["channel_type"] = {"ekg": "ecg", "trig": "stim", "trigger": "stim", "status": "stim"}.get(kind, kind)
+        return result
+    if _SCALP.fullmatch(text):
+        result["channel_type"] = "eeg"
+        return result
+    pair = text.split("-")
+    if len(pair) == 2 and _contact(pair[0]):
+        first = _contact(pair[0])
+        second_name = first[0] + pair[1] if pair[1].isdigit() else pair[1]
+        second = _contact(second_name)
+        if second:
+            result.update(shaft=first[0] if first[0] == second[0] else None,
+                          measurement_kind="bipolar", contact_names=[pair[0], second_name], channel_type="seeg")
+            return result
+    contact = _contact(text)
+    if contact:
+        result.update(shaft=contact[0], contact_index=contact[1], measurement_kind="monopolar",
+                      contact_names=[text], channel_type="seeg")
+    return result
+
+
+def resolve_seeg_channels(catalog, channel_map=None):
+    """Return candidates; only an explicit confirmed map establishes identities.
+
+    Names are never identifiers. In particular A1/A01 remain separate inputs,
+    and candidate bipolar contacts never cause a numerical re-reference.
+    """
+    if channel_map is None:
+        channel_map = []
+    if not isinstance(channel_map, list):
+        raise ValueError("channel_map must be a list")
+    overrides = {}
+    for item in channel_map:
+        if not isinstance(item, dict) or set(item) - _FIELDS or not isinstance(item.get("channel_id"), str):
+            raise ValueError("channel_map entries require channel_id and supported mapping fields")
+        if item["channel_id"] in overrides:
+            raise ValueError("Duplicate channel_id in channel_map")
+        overrides[item["channel_id"]] = item
+    output, seen = [], set()
+    for stream in catalog.get("streams", []):
+        stream_ids = {c["channel_id"] for c in stream["channels"]}
+        for channel in stream["channels"]:
+            cid = channel["channel_id"]
+            if cid in seen:
+                raise ValueError("Channel IDs must be globally unique")
+            seen.add(cid)
+            raw_name = channel["raw_name"]
+            if not isinstance(raw_name, str):
+                raise ValueError("raw_name must preserve the original string")
+            item = {"channel_id": cid, "raw_name": raw_name, "stream_id": stream["stream_id"],
+                    "source_id": stream.get("source_id"), "unit": channel.get("unit"),
+                    "channel_type": channel.get("channel_type", "unknown").lower(), "status": "unreviewed",
+                    "shaft": None, "contact_index": None, "measurement_kind": "unknown", "contact_names": [],
+                    "reference": None, "source_channels": [], "candidate": _candidate(raw_name), "issues": []}
+            override = overrides.get(cid, {})
+            status = override.get("status", "unreviewed")
+            if status not in ("unreviewed", "confirmed"):
+                raise ValueError("Mapping status must be unreviewed or confirmed")
+            if override and status != "confirmed":
+                raise ValueError("Explicit channel mapping must carry status=confirmed")
+            item.update(deepcopy(override))
+            _validate_mapping(item, stream_ids)
+            if item["status"] != "confirmed":
+                item["issues"].append("mapping_pending")
+            if item["measurement_kind"] == "bipolar":
+                ends = [_contact(name) for name in item["contact_names"]]
+                if all(ends) and (ends[0][0] != ends[1][0] or abs(ends[1][1] - ends[0][1]) != 1):
+                    item["issues"].append("nonadjacent_bipolar_contacts")
+            output.append(item)
+    if set(overrides) - seen:
+        raise ValueError("channel_map contains unknown channel_id")
+    _annotate_names(output)
+    return output
+
+
+def _validate_mapping(item, stream_ids):
+    if item["channel_type"] not in _TYPES:
+        raise ValueError("Unsupported channel_type")
+    kind = item["measurement_kind"]
+    if kind not in ("unknown", "monopolar", "bipolar", "linear"):
+        raise ValueError("Unsupported measurement_kind")
+    names = item["contact_names"]
+    if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names):
+        raise ValueError("contact_names must contain explicit nonempty contact names")
+    if kind in ("monopolar", "bipolar") and len(names) != (1 if kind == "monopolar" else 2):
+        raise ValueError("Monopolar and bipolar maps require one and two contact names respectively")
+    if len(names) != len(set(names)):
+        raise ValueError("A measurement cannot repeat the same contact")
+    sources = item["source_channels"]
+    if (not isinstance(sources, list) or any(not isinstance(c, str) or c not in stream_ids for c in sources)
+            or len(sources) != len(set(sources))):
+        raise ValueError("source_channels must be distinct channel IDs from the same stream")
+    if item["reference"] is not None and not isinstance(item["reference"], str):
+        raise ValueError("reference must be a string or null")
+    index = item["contact_index"]
+    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
+        raise ValueError("contact_index must be a nonnegative integer or null")
+    if item["shaft"] is not None and (not isinstance(item["shaft"], str) or not item["shaft"]):
+        raise ValueError("shaft must be a nonempty string or null")
+
+
+def _annotate_names(channels):
+    aliases, shafts, raw_names = defaultdict(list), defaultdict(list), defaultdict(list)
+    for item in channels:
+        candidate = item["candidate"]
+        raw_names[(item["stream_id"], item["raw_name"])].append(item)
+        if candidate["contact_index"] is not None:
+            key = (item["stream_id"], candidate["shaft"])
+            shafts[key].append(item)
+            aliases[key + (candidate["contact_index"],)].append(item)
+    for group in raw_names.values():
+        if len(group) > 1:
+            for item in group:
+                item["issues"].append("duplicate_raw_name")
+    for group in aliases.values():
+        if len({c["raw_name"] for c in group}) > 1:
+            for item in group:
+                item["issues"].append("possible_alias")
+    for group in shafts.values():
+        numbers = sorted({c["candidate"]["contact_index"] for c in group})
+        if any(b - a > 1 for a, b in zip(numbers, numbers[1:])):
+            for item in group:
+                item["issues"].append("missing_contact_numbers")
diff --git a/backend/app/processing/ephys_qc/seeg_metrics.py b/backend/app/processing/ephys_qc/seeg_metrics.py
new file mode 100644
index 0000000000000000000000000000000000000000..bc7d4c0db6ddb5796665be6f9bd22472c93bbaad
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg_metrics.py
@@ -0,0 +1,172 @@
+"""Window-local evidence in physical volts, without filtering or interpolation."""
+import math
+
+import numpy as np
+from scipy.signal import welch
+
+
+def _longest(mask):
+    padded = np.r_[False, mask, False].astype(np.int8)
+    edges = np.flatnonzero(np.diff(padded))
+    return int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0
+
+
+def _number(value):
+    return float(value) if np.isfinite(value) else None
+
+
+def _rails(channel):
+    """File encoding bounds in volts; header calibration does not confirm ADC rails."""
+    fields = ("physical_min", "physical_max", "scale_to_si", "digital_min", "digital_max")
+    if any(channel.get(field) is None for field in fields):
+        return None
+    if any(isinstance(channel[field], bool) or not isinstance(channel[field], (int, float))
+           or not math.isfinite(channel[field]) for field in fields):
+        raise ValueError("File calibration fields must be finite numbers")
+    low, high = sorted([channel["physical_min"] * channel["scale_to_si"],
+                        channel["physical_max"] * channel["scale_to_si"]])
+    codes = channel["digital_max"] - channel["digital_min"]
+    if low == high or codes <= 0 or channel["scale_to_si"] <= 0:
+        raise ValueError("Invalid file calibration")
+    return low, high, (high - low) / codes
+
+
+def compute_window_metrics(values, frequency, channel, config):
+    values = np.asarray(values, dtype=np.float64)
+    finite = np.isfinite(values)
+    valid = values[finite]
+    voltage = channel.get("unit") == "V" and channel.get("quantity", "voltage") == "voltage"
+    metrics = {"unit": channel.get("unit"), "sample_count": len(values), "finite_sample_count": int(finite.sum()),
+               "finite_fraction": float(finite.mean()), "missing_fraction": float(1 - finite.mean()),
+               "flat_fraction": None, "longest_flat_seconds": None, "clipping_fraction": None,
+               "longest_clipping_seconds": None, "rms_v": None, "mad_v": None, "robust_range_v": None,
+               "offset_v": None, "drift_v_per_second": None, "max_jump_v": None,
+               "kurtosis": None, "line_length_v_per_second": None, "line_noise_db": None,
+               "line_noise_fraction": None, "line_peak_width_hz": None, "high_frequency_ratio": None,
+               "line_harmonics": [],
+               "voltage_status": "computed" if voltage and len(valid) else "insufficient_data",
+               "clipping_status": "insufficient_data", "spectral_status": "insufficient_data",
+               "boundary_dwell_status": "insufficient_data", "boundary_dwell_fraction": None,
+               "longest_boundary_dwell_seconds": None, "boundary_dwell_score_effect": "none",
+               "adc_range_status": "unconfirmed",
+               "background_fit_status": "not_implemented", "alpha_peak_status": "not_scored"}
+    flags = ["nonfinite"] if not finite.all() else []
+    spectrum = {"status": "insufficient_data", "reason": "unknown_voltage_unit" if not voltage else "too_short_or_nonfinite_or_constant",
+                "frequency_hz": [], "power_v2_hz": []}
+    if not voltage:
+        return metrics, flags, spectrum
+    if len(valid):
+        offset = np.median(valid)
+        metrics.update(rms_v=_number(np.sqrt(np.mean(valid ** 2))), mad_v=_number(np.median(np.abs(valid - offset))),
+                       robust_range_v=_number(np.percentile(valid, 99) - np.percentile(valid, 1)), offset_v=_number(offset))
+    if len(values) < 2:
+        return metrics, flags, spectrum
+    differences = np.diff(values)
+    valid_pairs = finite[:-1] & finite[1:]
+    flat = valid_pairs & (np.abs(differences) <= config["flat_tolerance_v"])
+    longest_flat = (_longest(flat) + 1) / frequency if flat.any() else 0.0
+    metrics.update(flat_fraction=float(flat.sum() / max(1, valid_pairs.sum())), longest_flat_seconds=longest_flat)
+    if longest_flat >= config["flat_min_seconds"] or flat.all():
+        flags.append("flat")
+    if valid_pairs.any():
+        metrics.update(max_jump_v=_number(np.max(np.abs(differences[valid_pairs]))),
+                       line_length_v_per_second=_number(np.abs(differences[valid_pairs]).sum() / (len(values) / frequency)))
+    if len(valid) >= 3:
+        centered = valid - valid.mean()
+        variance = np.mean(centered ** 2)
+        metrics["kurtosis"] = _number(np.mean(centered ** 4) / variance ** 2) if variance > 0 else None
+        times = np.flatnonzero(finite) / frequency
+        times -= times.mean()
+        metrics["drift_v_per_second"] = _number(np.dot(times, centered) / np.dot(times, times))
+    rails = _rails(channel)
+    if rails:
+        low, high, step = rails
+        clipped = finite & ((np.abs(values - low) <= step * .25) | (np.abs(values - high) <= step * .25))
+        metrics.update(clipping_status="computed", clipping_fraction=float(clipped.mean()),
+                       longest_clipping_seconds=_longest(clipped) / frequency,
+                       boundary_dwell_status="computed", boundary_dwell_fraction=float(clipped.mean()),
+                       longest_boundary_dwell_seconds=_longest(clipped) / frequency,
+                       boundary_source="file_encoding_limits", boundary_low_v=low, boundary_high_v=high,
+                       encoding_step_v=step)
+        if metrics["clipping_fraction"] >= config["clipping_fraction"] or metrics["longest_clipping_seconds"] >= config["clipping_min_seconds"]:
+            # Retain the old metric names for report readers, but only surface
+            # unconfirmed file bounds as review evidence (no scoring flag).
+            flags.append("boundary_dwell")
+    if finite.all() and len(values) >= 8 and len(values) / frequency >= 1 and np.ptp(values) > 0:
+        spectrum = compute_spectrum(values, frequency, config)
+        metrics.update({key: spectrum[key] for key in ("line_noise_db", "line_noise_fraction", "line_peak_width_hz", "high_frequency_ratio")})
+        metrics["spectral_status"] = "computed"
+        metrics["line_harmonics"] = spectrum["harmonics"]
+        if spectrum["line_candidate"]:
+            flags.append("line_noise")
+    return metrics, flags, spectrum
+
+
+def compute_spectrum(values, frequency, config):
+    nfft = max(8, min(8192, int(round(2 * frequency))))
+    freq, power = welch(values, fs=frequency, window="hann", nperseg=min(len(values), nfft),
+                        nfft=nfft, noverlap=min(len(values), nfft) // 2, detrend="constant", scaling="density")
+    resolution = frequency / nfft
+    total = float(np.sum(power[freq >= 1]) * resolution)
+    harmonics = []
+    for center in np.arange(config["line_frequency"], frequency / 2 - 5, config["line_frequency"]):
+        peak_mask = np.abs(freq - center) <= max(1, resolution)
+        neighbor = (np.abs(freq - center) >= 2) & (np.abs(freq - center) <= 5)
+        if not peak_mask.any() or neighbor.sum() < 2 or resolution > 2:
+            continue
+        indices = np.flatnonzero(peak_mask)
+        index = int(indices[np.argmax(power[indices])])
+        background = float(np.median(power[neighbor]))
+        peak = float(power[index])
+        floor = max(float(np.max(power)) * 1e-15, np.finfo(float).tiny)
+        ratio = 10 * math.log10(max(peak, floor) / max(background, floor))
+        left = right = index
+        while left > 0 and power[left - 1] >= peak * .5:
+            left -= 1
+        while right < len(power) - 1 and power[right + 1] >= peak * .5:
+            right += 1
+        width = (right - left + 1) * resolution
+        fraction = float(power[peak_mask].sum() * resolution / total) if total > 0 else None
+        harmonics.append({"frequency_hz": float(center), "peak_frequency_hz": float(freq[index]),
+                          "prominence_db": ratio, "power_fraction": fraction, "half_power_width_hz": width,
+                          "candidate": bool(ratio >= config["line_noise_db"] and fraction is not None
+                                            and fraction >= config["line_noise_fraction"] and width <= 3)})
+    strongest = max(harmonics, key=lambda item: item["prominence_db"], default=None)
+    return {"status": "computed", "reason": None, "frequency_hz": freq.tolist(), "power_v2_hz": power.tolist(),
+            "method": "welch_hann_constant_detrend", "frequency_resolution_hz": resolution,
+            "nperseg": min(len(values), nfft), "nfft": nfft, "harmonics": harmonics,
+            "line_noise_db": strongest["prominence_db"] if strongest else None,
+            "line_noise_fraction": sum(h["power_fraction"] or 0 for h in harmonics) if harmonics else None,
+            "line_peak_width_hz": strongest["half_power_width_hz"] if strongest else None,
+            "line_candidate": any(h["candidate"] for h in harmonics),
+            "high_frequency_ratio": float(power[freq >= 80].sum() * resolution / total) if total > 0 and freq[-1] >= 80 else None}
+
+
+def compute_relations(data, mappings, stream_id, segment_id, start, stop):
+    """Same-clock pair evidence only; never establishes a damaged physical contact."""
+    relations = []
+    eligible = [i for i, item in enumerate(mappings) if item["unit"] == "V"
+                and (item["channel_type"] == "seeg" or (item["channel_type"] == "unknown"
+                     and item["candidate"]["channel_type"] not in ("ecg", "eog", "emg", "stim")))]
+    for pos, i in enumerate(eligible):
+        a = data[i]
+        if not np.isfinite(a).all() or np.ptp(a) == 0:
+            continue
+        for j in eligible[pos + 1:]:
+            b = data[j]
+            if not np.isfinite(b).all() or np.ptp(b) == 0:
+                continue
+            exact = np.array_equal(a, b)
+            correlation = 1.0 if exact else float(np.corrcoef(a, b)[0, 1])
+            shaft_a = mappings[i]["shaft"] or mappings[i]["candidate"]["shaft"]
+            shaft_b = mappings[j]["shaft"] or mappings[j]["candidate"]["shaft"]
+            if exact or abs(correlation) >= .995:
+                kind = "duplicate_signal_candidate" if exact else "high_correlation_candidate"
+            elif shaft_a and shaft_a == shaft_b:
+                kind = "same_shaft_correlation_evidence"
+            else:
+                continue
+            relations.append({"kind": kind, "channel_ids": [mappings[i]["channel_id"], mappings[j]["channel_id"]],
+                              "stream_id": stream_id, "segment_id": segment_id, "start_sample": start,
+                              "stop_sample": stop, "correlation": correlation, "score_effect": "none"})
+    return relations
diff --git a/backend/app/processing/ephys_qc/seeg_scoring.py b/backend/app/processing/ephys_qc/seeg_scoring.py
new file mode 100644
index 0000000000000000000000000000000000000000..802f6115c533dd011fd5730f11feda16289f5433
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg_scoring.py
@@ -0,0 +1,69 @@
+"""Transparent, uncalibrated SEEG technical-risk rules; no physiology penalty."""
+import math
+
+
+RULE_VERSION = "seeg_basic_v2"
+PENALTIES = {"nonfinite": 100, "flat": 100, "clipping": 80, "line_noise": 35}
+
+
+def score_window(flags, eligible=True):
+    """Related observations share a window: only the strongest deduction applies."""
+    return 100 - max((PENALTIES.get(flag, 0) for flag in flags), default=0) if eligible else None
+
+
+def compute_quality_score(windows, decisions=None):
+    """Duration-weighted channel-window score; missing indicators stay missing.
+
+    Review decisions are optional: accepted confirms technical exclusion (0),
+    rejected dismisses the automatic candidate (100), pending keeps initial risk.
+    This aggregate does not replace the service's exact reviewed interval masks.
+    """
+    decisions = decisions or {}
+    if not isinstance(decisions, dict):
+        raise ValueError("decisions must map window_id to a decision")
+    known = {w["window_id"] for w in windows}
+    if set(decisions) - known:
+        raise ValueError("Decision references unknown window_id")
+    weighted, assessed, unassessed, reviewed = 0.0, 0.0, 0.0, 0.0
+    components = {flag: {"flag": flag, "penalty": penalty, "window_count": 0, "channel_seconds": 0.0,
+                         "evaluated_channel_seconds": 0.0, "unevaluated_channel_seconds": 0.0}
+                  for flag, penalty in PENALTIES.items()}
+    for window in windows:
+        duration = window["duration_seconds"]
+        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
+            raise ValueError("Window duration must be finite and positive")
+        for flag in set(window.get("flags", [])) & set(components):
+            components[flag]["window_count"] += 1
+            components[flag]["channel_seconds"] += duration
+        metrics = window.get("metrics", {})
+        evaluated = {"nonfinite": metrics.get("finite_fraction") is not None,
+                     "flat": metrics.get("flat_fraction") is not None,
+                     "clipping": (metrics.get("clipping_status") == "computed"
+                                  and metrics.get("adc_range_status") == "confirmed"),
+                     "line_noise": metrics.get("line_noise_db") is not None}
+        for flag, component in components.items():
+            key = "evaluated_channel_seconds" if evaluated[flag] else "unevaluated_channel_seconds"
+            component[key] += duration
+        score = window.get("score") if window.get("score_eligible", True) else None
+        decision = decisions.get(window["window_id"], "pending")
+        if isinstance(decision, dict):
+            decision = decision.get("decision", decision.get("status"))
+        if decision not in ("accepted", "rejected", "pending"):
+            raise ValueError("Unsupported review decision")
+        if score is None:
+            unassessed += duration
+            continue
+        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
+            raise ValueError("Score must be in [0,100] or null")
+        if decision != "pending":
+            score = 0 if decision == "accepted" else 100
+            reviewed += duration
+        assessed += duration
+        weighted += score * duration
+    for component in components.values():
+        component["status"] = "computed" if component["evaluated_channel_seconds"] else "insufficient_data"
+    return {"total": int(math.floor(weighted / assessed + .5)) if assessed else None,
+            "components": list(components.values()), "rule_version": RULE_VERSION, "calibrated": False,
+            "aggregation": "duration_weighted_channel_windows_max_penalty",
+            "assessed_channel_seconds": assessed, "unassessed_channel_seconds": unassessed,
+            "reviewed_channel_seconds": reviewed}
diff --git a/backend/app/processing/ephys_qc/seeg_waveform.py b/backend/app/processing/ephys_qc/seeg_waveform.py
new file mode 100644
index 0000000000000000000000000000000000000000..62fe00db6671cb3c12fc78c34a67ae4f4b8a47cd
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg_waveform.py
@@ -0,0 +1,112 @@
+"""Bounded streaming summaries; QC-window boundaries never reset morphology."""
+import math
+
+import numpy as np
+
+from .seeg_waveform_events import WAVEFORM_DEFAULTS, WAVEFORM_VERSION, detect_events
+
+
+def validate_waveform_config(config):
+    if not .002 <= config["waveform_bin_seconds"] <= .1:
+        raise ValueError("waveform_bin_seconds must be within [.002,.1]")
+    if not .05 <= config["waveform_recovery_ratio"] <= .5:
+        raise ValueError("waveform_recovery_ratio must be within [.05,.5]")
+    for key in ("waveform_baseline_seconds", "waveform_min_event_seconds", "waveform_drift_seconds",
+                "waveform_plateau_seconds", "waveform_recovery_seconds"):
+        if not 3 * config["waveform_bin_seconds"] <= config[key] <= 10:
+            raise ValueError("{} must cover >=3 morphology bins and be <=10 seconds".format(key))
+    for key in ("max_waveform_bins", "max_waveform_events"):
+        if int(config[key]) != config[key] or not 1 <= config[key] <= WAVEFORM_DEFAULTS[key]:
+            raise ValueError("{} must be within the supported waveform resource limit".format(key))
+
+
+def preflight_waveform(plan, config):
+    bins = 0
+    for stream, _, start, stop, _ in plan:
+        size = max(1, round(config["waveform_bin_seconds"] * stream["sampling_frequency_hz"]))
+        bins += math.ceil((stop - start) / size) * len(stream["channels"])
+    if bins > config["max_waveform_bins"]:
+        raise ValueError("Waveform summary limit exceeded; select explicit shorter scan ranges")
+
+
+class WaveformTracker:
+    """Retain compact bin features and <one bin of samples per channel."""
+    def __init__(self, frequency, context, config):
+        self.frequency, self.context, self.config = frequency, context, config
+        self.size = max(1, round(frequency * config["waveform_bin_seconds"]))
+        self.buffer = np.empty(0)
+        self.buffer_start = None
+        self.previous = None
+        self.expected = None
+        self.parts = []
+        self.events = []
+
+    def feed(self, values, start):
+        if self.expected is not None and start != self.expected:
+            self.flush("range_end")
+        self.expected = start + len(values)
+        finite = np.isfinite(values)
+        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
+        runs = list(zip(edges[::2], edges[1::2]))
+        cursor = 0
+        for first, last in runs:
+            if first > cursor:
+                self.flush("nonfinite_gap")
+            self._append(values[first:last], start + first)
+            if last < len(values):
+                self.flush("nonfinite_gap")
+            cursor = last
+        if not runs:
+            self.flush("nonfinite_gap")
+
+    def _append(self, values, start):
+        if not len(self.buffer):
+            self.buffer_start = start
+        self.buffer = np.concatenate([self.buffer, values])
+        count = len(self.buffer) // self.size
+        if count:
+            used = count * self.size
+            self._summarize(self.buffer[:used].reshape(count, self.size), self.buffer_start)
+            self.buffer = self.buffer[used:].copy()
+            self.buffer_start += used
+
+    def _summarize(self, rows, start):
+        length = rows.shape[1]
+        centers = np.median(rows, axis=1)
+        mad = np.median(np.abs(rows - centers[:, None]), axis=1)
+        preceding = np.r_[rows[0, 0] if self.previous is None else self.previous, rows[:-1, -1]]
+        differences = np.diff(rows, axis=1, prepend=preceding[:, None])
+        positions = np.argmax(np.abs(differences), axis=1)
+        starts = start + np.arange(len(rows)) * length
+        changed = np.diff(rows, axis=1) != 0
+        prefix = np.where(changed.any(axis=1), np.argmax(changed, axis=1) + 1, length) if length > 1 else np.ones(len(rows))
+        suffix = np.where(changed.any(axis=1), np.argmax(changed[:, ::-1], axis=1) + 1, length) if length > 1 else np.ones(len(rows))
+        result = np.column_stack([starts, starts + length, centers, mad,
+                                  differences[np.arange(len(rows)), positions], starts + positions,
+                                  rows[:, 0], rows[:, -1], prefix, suffix])
+        self.parts.append(result)
+        self.previous = rows[-1, -1]
+
+    def flush(self, termination):
+        if len(self.buffer):
+            self._summarize(self.buffer.reshape(1, -1), self.buffer_start)
+        if self.parts:
+            summary = np.concatenate(self.parts)
+            self.events.extend(detect_events(summary, self.frequency, self.context, self.config, termination))
+        self.buffer = np.empty(0)
+        self.buffer_start, self.previous = None, None
+        self.parts.clear()
+
+
+def attach_waveform_events(windows, events):
+    grouped = {}
+    for event in events:
+        key = (event["stream_id"], event["segment_id"], event["channel_id"])
+        grouped.setdefault(key, []).append(event)
+    for window in windows:
+        key = (window["stream_id"], window["segment_id"], window["channel_id"])
+        selected = [event["event_id"] for event in grouped.get(key, [])
+                    if event["start_sample"] < window["stop_sample"] and event["stop_sample"] > window["start_sample"]]
+        window["waveform_event_ids"] = selected
+        if selected:
+            window["flags"].append("waveform_anomaly")
diff --git a/backend/app/processing/ephys_qc/seeg_waveform_events.py b/backend/app/processing/ephys_qc/seeg_waveform_events.py
new file mode 100644
index 0000000000000000000000000000000000000000..56f39b6e556f49f43452d3890fac27aea10a20f2
--- /dev/null
+++ b/backend/app/processing/ephys_qc/seeg_waveform_events.py
@@ -0,0 +1,249 @@
+"""Joint shape candidates from fixed-time summaries, with explicit recovery."""
+import hashlib
+import math
+
+import numpy as np
+
+
+WAVEFORM_VERSION = "seeg-waveform-1"
+WAVEFORM_DEFAULTS = {
+    "waveform_bin_seconds": .02, "waveform_baseline_seconds": 1.0,
+    "waveform_min_event_seconds": .3, "waveform_jump_min_v": 150e-6,
+    "waveform_relative_multiplier": 8.0, "waveform_scale_floor_v": 5e-6,
+    "waveform_drift_min_v": 300e-6, "waveform_drift_seconds": 1.0,
+    "waveform_long_drift_min_v": .001,
+    "waveform_plateau_seconds": .15, "waveform_recovery_seconds": 1.0,
+    "waveform_recovery_ratio": .2, "waveform_recovery_abs_v": 30e-6,
+    "waveform_recovery_scale_multiplier": 2.0, "max_waveform_bins": 1000000,
+    "max_waveform_events": 10000,
+}
+# Summary columns: half-open sample interval, median, within-bin MAD, strongest
+# signed adjacent-sample jump and its actual later-sample position.
+START, STOP, LEVEL, MAD, JUMP, JUMP_AT, FIRST, LAST, PREFIX, SUFFIX = range(10)
+
+
+def _runs(mask):
+    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
+    return list(zip(edges[::2], edges[1::2]))
+
+
+def _line(values):
+    x = np.arange(len(values), dtype=float)
+    x -= x.mean()
+    y = values - values.mean()
+    denominator = float(np.dot(x, x))
+    slope = float(np.dot(x, y) / denominator) if denominator else 0.0
+    variation = float(np.dot(y, y))
+    r2 = 1 - float(np.sum((y - slope * x) ** 2)) / variation if variation else 0.0
+    return slope, r2
+
+
+def _seed(summary, i, cfg, nb, ne, nd, np_):
+    level = summary[:, LEVEL]
+    before = level[i - nb:i]
+    base = float(np.median(before))
+    scale = max(cfg["waveform_scale_floor_v"], 1.4826 * float(np.median(np.abs(before - base))),
+                1.4826 * float(np.median(summary[i - nb:i, MAD])))
+    threshold = max(cfg["waveform_jump_min_v"], cfg["waveform_relative_multiplier"] * scale)
+    after = level[i:i + ne]
+    shift = float(np.median(after) - base)
+    short_change = float(np.median(level[i:i + 2]) - np.median(level[max(0, i - 2):i]))
+    jump_index = i + int(np.argmax(np.abs(summary[i:min(len(summary), i + 2), JUMP])))
+    actual_jump = float(summary[jump_index, JUMP])
+    sustained = len(after) == ne and np.mean(np.sign(shift) * (after - base) >= abs(shift) * .6) >= .9
+    if (sustained and abs(shift) >= threshold and abs(short_change) >= threshold * .7
+            and abs(actual_jump) >= threshold * .5):
+        return {"i": i, "sample": int(summary[jump_index, JUMP_AT]), "baseline": base, "scale": scale,
+                "jump": actual_jump, "kind": "jump", "reason": "sustained_level_shift", "initial_end": i + ne}
+    if i + nd + np_ > len(level):
+        return None
+    ramp, plateau = level[i:i + nd], level[i + nd:i + nd + np_]
+    excursion = float(np.median(plateau) - base)
+    if abs(excursion) < max(cfg["waveform_drift_min_v"], cfg["waveform_relative_multiplier"] * scale):
+        return None
+    slope, r2 = _line(ramp)
+    tail_slope, _ = _line(ramp[-max(3, nd // 5):])
+    plateau_slope, _ = _line(plateau)
+    stable_plateau = np.ptp(plateau) <= max(4 * scale, .08 * abs(excursion))
+    if (r2 >= .98 and abs(slope) > 0 and np.sign(slope) == np.sign(excursion)
+            and abs(tail_slope) >= .65 * abs(slope) and abs(plateau_slope) <= .1 * abs(slope)
+            and stable_plateau):
+        return {"i": i, "sample": int(summary[i, START]), "baseline": base, "scale": scale, "jump": 0.0,
+                "kind": "drift", "reason": "linear_drift_with_plateau", "initial_end": i + nd + np_}
+    # A long, nearly linear unresolved displacement may end before a plateau.
+    # Smooth oscillatory turning points fail this shape/persistence conjunction.
+    long_stop = i + 2 * nd
+    if long_stop <= len(level):
+        long = level[i:long_stop]
+        long_slope, long_r2 = _line(long)
+        if (long_r2 >= .997 and abs(long[-1] - long[0]) >= max(cfg["waveform_long_drift_min_v"], 2 * cfg["waveform_relative_multiplier"] * scale)
+                and np.sign(long_slope) == np.sign(excursion)):
+            return {"i": i, "sample": int(summary[i, START]), "baseline": base, "scale": scale,
+                    "jump": 0.0, "kind": "drift", "reason": "sustained_linear_drift", "initial_end": long_stop}
+    return None
+
+
+def detect_events(summary, frequency, context, config, termination):
+    if not len(summary):
+        return []
+    dt = max(1, round(frequency * config["waveform_bin_seconds"])) / frequency
+    count = lambda key: max(2, int(math.ceil(config[key] / dt)))
+    nb, ne, nd, np_, nr = [count(k) for k in ("waveform_baseline_seconds", "waveform_min_event_seconds",
+                                            "waveform_drift_seconds", "waveform_plateau_seconds", "waveform_recovery_seconds")]
+    level = summary[:, LEVEL]
+    # Cheap absolute gates keep detailed fits away from normal bins. The final
+    # decision still requires local robust scale, persistence and joint shape.
+    jump_candidates = np.flatnonzero(np.abs(summary[:, JUMP]) >= config["waveform_jump_min_v"] * .5)
+    drift_candidates = np.flatnonzero(np.abs(level[nd:] - level[:-nd]) >= config["waveform_drift_min_v"]) if len(level) > nd else []
+    candidate_indices = sorted(set(int(i) for j in jump_candidates for i in (j - 1, j)) | set(map(int, drift_candidates)))
+    events, cursor = [], nb
+    for i in candidate_indices:
+        if i < cursor or i < nb or i + ne > len(level):
+            continue
+        seed = _seed(summary, i, config, nb, ne, nd, np_)
+        if seed is None:
+            continue
+        if seed["kind"] == "jump":
+            # A sustained event can begin with a smooth displacement before its
+            # largest edge. Trace only an uninterrupted same-direction offset;
+            # a preceding spike that returned to baseline remains separate.
+            original = seed["i"]
+            direction = np.sign(np.median(level[i:seed["initial_end"]]) - seed["baseline"])
+            threshold = max(config["waveform_jump_min_v"] * .5, 3 * seed["scale"])
+            while i > max(cursor, original - nb) and direction * (level[i - 1] - seed["baseline"]) >= threshold:
+                i -= 1
+            if i < original:
+                seed.update(i=i, jump_sample=seed["sample"], sample=int(summary[i, START]))
+        # Recovery is relative to the pre-event baseline; following a shifted
+        # level with a rolling baseline would falsely declare recovery.
+        initial = level[i:seed["initial_end"]]
+        peak = max(abs(float(np.max(initial) - seed["baseline"])), abs(float(np.min(initial) - seed["baseline"])))
+        tolerance = _recovery_tolerance(peak, seed["scale"], config)
+        near = np.abs(level - seed["baseline"]) <= tolerance
+        recovered = None
+        for a, b in _runs(near):
+            if a < seed["initial_end"] or b - a < nr:
+                continue
+            target = summary[a, START] + math.ceil(config["waveform_recovery_seconds"] * frequency)
+            end = int(np.searchsorted(summary[:, STOP], target, side="left")) + 1
+            if end <= b:
+                recovered = (a, end)
+                break
+        end = recovered[1] if recovered else len(level)
+        event = _event(summary, seed, end, recovered, frequency, context, config, termination, np_)
+        events.append(event)
+        cursor = end + nb
+        if len(events) > config["max_waveform_events"]:
+            raise ValueError("Waveform event limit exceeded; select a shorter scan range")
+    return events
+
+
+def _plateaus(summary, first, end, base, scale, peak, frequency, config, np_):
+    exact, pending = [], None
+    for row in summary[first:end]:
+        start, stop = int(row[START]), int(row[STOP])
+        prefix, suffix = int(row[PREFIX]), int(row[SUFFIX])
+        pieces = [(start, start + prefix, row[FIRST])]
+        if prefix < stop - start:
+            pieces.append((stop - suffix, stop, row[LAST]))
+        for a, b, value in pieces:
+            if pending and pending[1] == a and pending[2] == value:
+                pending = (pending[0], b, value)
+            else:
+                if pending:
+                    exact.append(pending)
+                pending = (a, b, value)
+    if pending:
+        exact.append(pending)
+    result = [{"start": a, "stop": b, "level": float(value), "range": 0.0, "method": "exact_constant_run"}
+              for a, b, value in exact if (b - a) / frequency >= config["waveform_plateau_seconds"]
+              and abs(value - base) >= max(3 * scale, peak * .25)]
+    # Approximate plateaus retain one fixed level anchor. Moving-window unions
+    # otherwise turn a long smooth recovery slope into a fictitious plateau.
+    a = first
+    limit = max(2 * scale, peak * .002)
+    while a + np_ <= end:
+        values = summary[a:a + np_, LEVEL]
+        slope, _ = _line(values)
+        if (np.ptp(values) > limit or np.min(np.abs(values - base)) < max(3 * scale, peak * .25)
+                or abs(slope) * np_ > scale * .25):
+            a += 1
+            continue
+        anchor = float(np.median(values))
+        b = a + np_
+        while b < end and abs(summary[b, LEVEL] - anchor) <= limit * .5:
+            b += 1
+        start, stop = int(summary[a, START]), int(summary[b - 1, STOP])
+        overlaps_exact = any(item["start"] < stop and item["stop"] > start for item in result)
+        if not overlaps_exact:
+            result.append({"start": start, "stop": stop, "level": anchor,
+                           "range": float(np.ptp(summary[a:b, LEVEL])), "method": "anchored_low_variability"})
+        a = b
+    return sorted(result, key=lambda item: item["start"])
+
+
+def _event(summary, seed, end, recovered, frequency, context, config, termination, np_):
+    i, base, scale = seed["i"], seed["baseline"], seed["scale"]
+    values = summary[i:end, LEVEL]
+    peak = float(np.max(np.abs(values - base)))
+    plateaus = _plateaus(summary, i, end, base, scale, peak, frequency, config, np_)
+    stages = []
+    if seed["kind"] == "jump":
+        jump_sample = seed.get("jump_sample", seed["sample"])
+        if jump_sample > seed["sample"]:
+            stages.append(_stage("drift", seed["sample"], jump_sample, context, frequency,
+                                 {"direction": "away_from_baseline", "method": "contiguous_offset_backtrace"}))
+        stages.append(_stage("jump", jump_sample, jump_sample + 1, context, frequency,
+                             {"jump_v": seed["jump"]}))
+    else:
+        drift_stop = plateaus[0]["start"] if plateaus else int(summary[end - 1, STOP])
+        stages.append(_stage("drift", seed["sample"], max(seed["sample"] + 1, drift_stop), context, frequency,
+                             {"net_change_v": float(summary[min(seed["initial_end"], end) - 1, LEVEL] - base)}))
+    for plateau in plateaus:
+        stages.append(_stage("plateau", max(seed["sample"], plateau["start"]), plateau["stop"], context, frequency,
+                             {"level_v": plateau["level"], "range_v": plateau["range"], "method": plateau["method"]}))
+    if recovered:
+        recovery_start = int(summary[recovered[0], START])
+        if plateaus and plateaus[-1]["stop"] < recovery_start:
+            stages.append(_stage("drift", plateaus[-1]["stop"], recovery_start, context, frequency,
+                                 {"direction": "toward_baseline"}))
+        stages.append(_stage("recovery", recovery_start,
+                             int(summary[recovered[1] - 1, STOP]), context, frequency,
+                             {"confirmation_start_sample": int(summary[recovered[0], START]),
+                              "confirmed_seconds": config["waveform_recovery_seconds"]}))
+    elif plateaus:
+        drift_start = plateaus[-1]["stop"]
+        stop = int(summary[end - 1, STOP])
+        if stop - drift_start >= max(2, round(config["waveform_plateau_seconds"] * frequency)):
+            stages.append(_stage("drift", drift_start, stop, context, frequency,
+                                 {"direction": "unresolved_baseline_shift", "right_censored": True}))
+    start, stop = seed["sample"], int(summary[end - 1, STOP])
+    identity = "{}:{}:{}:{}:{}".format(context["stream_id"], context["segment_id"], context["channel_id"], start, stop)
+    return {"event_id": "wave-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
+            **{k: context[k] for k in ("channel_id", "stream_id", "segment_id")},
+            "start_sample": start, "stop_sample": stop, "recording_start_seconds": _time(start, context, frequency),
+            "recording_stop_seconds": _time(stop, context, frequency), "duration_seconds": (stop - start) / frequency,
+            "stages": stages, "recovery_status": "confirmed" if recovered else "right_censored",
+            "termination_reason": "recovered" if recovered else termination, "score_effect": "none", "status": "candidate",
+            "features": {"baseline_v": base, "robust_scale_v": scale, "peak_offset_v": peak, "jump_v": seed["jump"],
+                         "plateau_seconds": sum(stage["duration_seconds"] for stage in stages if stage["kind"] == "plateau"),
+                         "recovery_confirmation_seconds": config["waveform_recovery_seconds"],
+                         "recovery_tolerance_v": _recovery_tolerance(peak, scale, config),
+                         "detection_bin_samples": max(1, round(frequency * config["waveform_bin_seconds"]))},
+            "reasons": [seed["reason"]] + (["low_variability_plateau"] if plateaus else [])
+                       + ["baseline_recovery_confirmed" if recovered else "recovery_not_observed"]}
+
+
+def _time(sample, context, frequency):
+    return context["recording_start_seconds"] + (sample - context["segment_start_sample"]) / frequency
+
+
+def _recovery_tolerance(peak, scale, config):
+    baseline = max(config["waveform_recovery_abs_v"], config["waveform_recovery_scale_multiplier"] * scale)
+    return max(config["waveform_scale_floor_v"], min(baseline, config["waveform_recovery_ratio"] * peak))
+
+
+def _stage(kind, start, stop, context, frequency, features):
+    return {"kind": kind, "start_sample": start, "stop_sample": stop,
+            "recording_start_seconds": _time(start, context, frequency), "recording_stop_seconds": _time(stop, context, frequency),
+            "duration_seconds": (stop - start) / frequency, "features": features}
diff --git a/backend/app/schema.sql b/backend/app/schema.sql
index 829da57cbd9a8a85b396634b509ca1ff68b483b0..9595923a2643f9b41e8462d755e8ddca0d37a738 100644
--- a/backend/app/schema.sql
+++ b/backend/app/schema.sql
@@ -159,2 +159,67 @@ CREATE INDEX IF NOT EXISTS idx_seeg_contacts_plan ON seeg_contacts(plan_id);

+-- S1: stable QC identities live alongside, never rewrite, legacy assets/eeg_files.
+-- Keep the additive upgrade in migrations/ephys_recordings_v1.py in sync.
+CREATE TABLE IF NOT EXISTS ephys_recordings (
+    recording_id TEXT PRIMARY KEY,
+    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
+    label TEXT NOT NULL,
+    modality TEXT NOT NULL,
+    session_id TEXT,
+    input_artifact_id TEXT NOT NULL UNIQUE,
+    import_key TEXT,
+    request_sha256 TEXT NOT NULL,
+    descriptor_json TEXT NOT NULL,
+    created_at TEXT NOT NULL,
+    UNIQUE(subject_id, import_key)
+);
+
+CREATE TABLE IF NOT EXISTS ephys_artifacts (
+    artifact_id TEXT PRIMARY KEY,
+    recording_id TEXT NOT NULL
+        REFERENCES ephys_recordings(recording_id) ON DELETE CASCADE,
+    kind TEXT NOT NULL CHECK(kind IN ('raw_input', 'derived_voltage')),
+    storage_path TEXT NOT NULL UNIQUE,
+    content_sha256 TEXT NOT NULL,
+    manifest_sha256 TEXT NOT NULL,
+    file_count INTEGER NOT NULL,
+    size_bytes INTEGER NOT NULL,
+    created_at TEXT NOT NULL
+);
+-- S2: one raw input and multiple immutable derived voltage versions per recording.
+-- Keep migrations/ephys_artifacts_v2.py in sync; the nine column order is stable.
+CREATE UNIQUE INDEX IF NOT EXISTS idx_ephys_artifacts_raw_input
+    ON ephys_artifacts(recording_id) WHERE kind='raw_input';
+CREATE INDEX IF NOT EXISTS idx_ephys_recordings_subject ON ephys_recordings(subject_id);
+
+-- SEEG basic QC: input-bound scans, optimistic review and immutable frozen package.
+-- Keep migrations/ephys_qc_v1.py in sync.
+CREATE TABLE IF NOT EXISTS ephys_qc_runs (
+    run_id TEXT PRIMARY KEY,
+    recording_id TEXT NOT NULL REFERENCES ephys_recordings(recording_id) ON DELETE CASCADE,
+    artifact_id TEXT NOT NULL,
+    job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
+    status TEXT NOT NULL,
+    provenance_json TEXT NOT NULL,
+    request_json TEXT NOT NULL,
+    report_json TEXT,
+    report_sha256 TEXT,
+    revision INTEGER NOT NULL DEFAULT 0,
+    review_json TEXT NOT NULL,
+    frozen_json TEXT,
+    export_sha256 TEXT,
+    error TEXT,
+    created_at TEXT NOT NULL,
+    updated_at TEXT NOT NULL
+);
+CREATE INDEX IF NOT EXISTS idx_ephys_qc_recording ON ephys_qc_runs(recording_id, created_at);
+CREATE TABLE IF NOT EXISTS ephys_qc_audit (
+    audit_id TEXT PRIMARY KEY,
+    run_id TEXT NOT NULL REFERENCES ephys_qc_runs(run_id) ON DELETE CASCADE,
+    revision INTEGER NOT NULL,
+    action TEXT NOT NULL,
+    reviewer TEXT,
+    snapshot_json TEXT NOT NULL,
+    created_at TEXT NOT NULL
+);
+
 COMMIT;
diff --git a/backend/tests/contract/test_ephys_derived.py b/backend/tests/contract/test_ephys_derived.py
new file mode 100644
index 0000000000000000000000000000000000000000..a1739c488fadeeba18151d21a54f25ad88f7068b
--- /dev/null
+++ b/backend/tests/contract/test_ephys_derived.py
@@ -0,0 +1,142 @@
+"""Processed windows retain parent/sample identity without changing raw playback."""
+from copy import deepcopy
+import io
+
+import numpy as np
+import pytest
+from werkzeug.datastructures import FileStorage
+
+from backend.app.db import get_db
+from backend.app.config import settings
+from backend.app.domain.subjects import create_subject
+from backend.app.domain.ephys_recordings import import_recording, get_recording, verify_recording
+from backend.app.domain.ephys_signals import describe_artifact, list_events
+from backend.app.domain.ephys_windows import samples, overview
+from backend.app.domain.ephys_storage import ImportConflict
+from backend.app.domain.ephys_derived import publish_voltage
+from backend.tests.ephys_edf_fixtures import tal, write_edf_bdf
+
+
+@pytest.fixture
+def processed(isolated_storage, tmp_path):
+    create_subject("S2DERIVED", "synthetic", None)
+    source = write_edf_bdf(tmp_path / "raw.edf", [
+        {"label": "same", "records": [[-2, -1, 0, 1], [2, 3, 4, 5]]},
+        {"label": "same", "records": [[3, 2, 1, 0], [-1, -2, -3, -4]]}],
+        reserved="EDF+D", annotation_records=[tal(2.5), tal(8.5)])
+    rec, _ = import_recording({"subject_id": "S2DERIVED", "label": "raw"},
+                             [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="raw.edf")])
+    parent = describe_artifact(rec["input_artifact_id"])
+    raw_stream = parent["streams"][0]
+    data = np.array([[-5, -3, -1, 1, 3, 5, 7, 9]], dtype="<f8") * 1e-6
+    path = tmp_path / "bipolar.npy"
+    np.save(path, data, allow_pickle=False)
+    desc = {"stream_key": "bipolar", "parent_stream_id": raw_stream["stream_id"],
+            "sampling_frequency_hz": 4., "sample_index_origin": 100, "sample_count": 8,
+            "reference": "explicit_linear_montage", "channels": [{"raw_name": "A-B", "unit": "V",
+                "source_channel_ids": [c["channel_id"] for c in raw_stream["channels"]],
+                "weights": [1., -1.]}],
+            "segments": [{"start_sample": 100 + s["start_sample"], "stop_sample": 100 + s["stop_sample"],
+                "recording_start_seconds": s["recording_start_seconds"], "parent_segment_id": s["segment_id"],
+                "parent_start_sample": s["start_sample"], "parent_stop_sample": s["stop_sample"],
+                "clock_mapping": "identity"} for s in raw_stream["segments"]]}
+    return rec, desc, path
+
+
+def publish(example, descriptor=None):
+    rec, desc, path = example
+    return publish_voltage(rec["input_artifact_id"], [{"path": path, "stream": descriptor or desc}],
+                           [{"operation": "linear_montage", "parameters": {"weights": [1., -1.]}}],
+                           {"fixture_generator": "1"})
+
+
+def test_processed_roundtrip_nonzero_origin_gaps_and_raw_are_independent(processed):
+    rec, _, _ = processed
+    result = publish(processed)
+    catalog = describe_artifact(result["artifact_id"])
+    stream = catalog["streams"][0]
+    assert catalog["data_kind"] == "derived_samples" and catalog["scientific_validity"] == "pending"
+    assert catalog["parent_artifact_id"] == rec["input_artifact_id"]
+    assert catalog["parent_content_sha256"] == rec["input_artifact"]["content_sha256"]
+    events = list_events(result["artifact_id"])
+    assert events["event_scope"] == "parent_input_original_samples"
+    assert events["parent_artifact_id"] == rec["input_artifact_id"]
+    assert catalog["processing"][0]["operation"] == "linear_montage"
+    assert stream["segments"][1]["recording_start_seconds"] == 8.5
+    meta, values = samples(result["artifact_id"], stream["stream_id"], stream["segments"][1]["segment_id"],
+                           stream["channel_order"], 105, 108, "physical")
+    assert meta["recording_start_seconds"] == 8.75 and meta["sample_index_origin"] == 100
+    np.testing.assert_array_equal(values, np.array([[5, 7, 9]]) * 1e-6)
+    assert meta["data_kind"] == "derived_samples" and meta["software_versions"] == {"fixture_generator": "1"}
+    over, data = overview(result["artifact_id"], stream["stream_id"], stream["segments"][1]["segment_id"],
+                          stream["channel_order"], 104, 108, "physical", 3)
+    assert over["source_data_kind"] == "derived_samples" and data.shape == (1, 2, 2)
+    assert get_recording(rec["recording_id"])["input_artifact_id"] == rec["input_artifact_id"]
+    assert verify_recording(rec["recording_id"])["valid"]
+    raw = describe_artifact(rec["input_artifact_id"])
+    assert raw["data_kind"] == "raw_samples" and raw["processing"] == []
+
+
+def test_republish_never_overwrites_and_rejects_digital_or_gap(processed):
+    first, second = publish(processed), publish(processed)
+    assert first["artifact_id"] != second["artifact_id"] and first["storage_path"] != second["storage_path"]
+    stream = describe_artifact(first["artifact_id"])["streams"][0]
+    for stop, representation in [(108, "physical"), (104, "digital")]:
+        with pytest.raises(ValueError):
+            samples(first["artifact_id"], stream["stream_id"], stream["segments"][0]["segment_id"],
+                    stream["channel_order"], 100, stop, representation)
+
+
+@pytest.mark.parametrize("change", ["time", "unit", "parent_channel", "shape", "rate", "mapping", "nan"])
+def test_bad_derived_contract_is_not_published(processed, change):
+    rec, desc, path = processed
+    desc = deepcopy(desc)
+    if change == "time": desc["segments"][0]["recording_start_seconds"] = 0
+    if change == "unit": desc["channels"][0]["unit"] = "unknown"
+    if change == "parent_channel": desc["channels"][0]["source_channel_ids"][0] = "other-artifact:ch:0"
+    if change == "shape": np.save(path, np.zeros((2, 8)), allow_pickle=False)
+    if change == "rate": desc["sampling_frequency_hz"] = 1000
+    if change == "mapping": desc["segments"][1]["parent_start_sample"] = 0
+    if change == "nan": np.save(path, np.full((1, 8), np.nan), allow_pickle=False)
+    with pytest.raises(ValueError): publish(processed, desc)
+    with get_db() as conn:
+        assert conn.execute("SELECT COUNT(*) AS n FROM ephys_artifacts WHERE recording_id=?",
+                            (rec["recording_id"],)).fetchone()["n"] == 1
+
+
+def test_changed_parent_or_derived_input_is_never_accepted(processed):
+    rec, _, _ = processed
+    result = publish(processed)
+    describe_artifact(result["artifact_id"])
+    path = settings.dataset_root / rec["input_artifact"]["storage_path"] / "files" / "raw.edf"
+    path.chmod(0o644)
+    with path.open("r+b") as handle:
+        handle.seek(-1, 2)
+        handle.write(b"X")
+    with pytest.raises(ImportConflict): describe_artifact(result["artifact_id"])
+
+
+def test_same_filename_in_different_artifacts_does_not_share_signal_identity(processed):
+    rec, _, path = processed
+    source = path.parent / "raw.edf"
+    again, _ = import_recording({"subject_id": "S2DERIVED", "label": "raw"},
+                               [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="raw.edf")])
+    a, b = [describe_artifact(r["input_artifact_id"])["streams"][0] for r in (rec, again)]
+    for key in ("stream_id", "source_id", "clock_id"):
+        assert a[key] != b[key]
+    assert set(a["channel_order"]).isdisjoint(b["channel_order"])
+
+
+def test_identical_parent_restoration_refreshes_derived_guard(processed):
+    rec, _, _ = processed
+    result = publish(processed)
+    stream = describe_artifact(result["artifact_id"])["streams"][0]
+    path = settings.dataset_root / rec["input_artifact"]["storage_path"] / "files" / "raw.edf"
+    original = path.read_bytes()
+    path.chmod(0o644)
+    path.write_bytes(original)
+    path.chmod(0o444)
+    meta, data = samples(result["artifact_id"], stream["stream_id"], stream["segments"][0]["segment_id"],
+                         stream["channel_order"], 100, 104, "physical")
+    assert meta["data_kind"] == "derived_samples"
+    np.testing.assert_array_equal(data, np.array([[-5, -3, -1, 1]]) * 1e-6)
diff --git a/backend/tests/contract/test_ephys_edf_reader.py b/backend/tests/contract/test_ephys_edf_reader.py
new file mode 100644
index 0000000000000000000000000000000000000000..f579a723bf6d4bc6222c058dd460ecd8a7278afa
--- /dev/null
+++ b/backend/tests/contract/test_ephys_edf_reader.py
@@ -0,0 +1,226 @@
+"""Format truth tests: known bytes, no cleaning and no inferred continuous time."""
+import base64
+
+import numpy as np
+import pytest
+
+from backend.app.processing.ephys_qc.readers.edf import EdfBdfReader, EphysReaderError
+from backend.tests.ephys_edf_fixtures import tal, write_edf_bdf
+
+
+def window(reader, stream=0, segment=0, channels=None, start=None, stop=None, representation="physical"):
+    info = reader.describe()["streams"][stream]
+    span = info["segments"][segment]
+    return reader.read_window(info["stream_id"], span["segment_id"],
+                              channels or [c["channel_id"] for c in info["channels"]],
+                              span["start_sample"] if start is None else start,
+                              span["stop_sample"] if stop is None else stop,
+                              representation=representation)
+
+
+@pytest.mark.parametrize("bdf", [False, True])
+def test_signed_codes_offsets_and_channel_order(tmp_path, bdf):
+    limit = 2 ** (23 if bdf else 15)
+    codes = [-limit, -1, 0, 1, limit - 1]
+    path = write_edf_bdf(tmp_path / "signal.dat", [
+        {"label": "duplicate", "records": [codes]},
+        {"label": "duplicate", "records": [codes], "physical_min": -10,
+         "physical_max": 30, "digital_min": -100, "digital_max": 100, "unit": "mV"},
+        {"label": "A1-A2", "records": [[1] * 5], "unit": "Ohm"}], bdf=bdf)
+    reader = EdfBdfReader(path, "input")
+    channels = reader.describe()["streams"][0]["channels"]
+    assert [c["raw_name"] for c in channels] == ["duplicate", "duplicate", "A1-A2"]
+    assert len({c["channel_id"] for c in channels}) == 3
+    assert [c["source_index"] for c in channels] == [0, 1, 2]
+    np.testing.assert_array_equal(window(reader, representation="digital")[0], codes)
+    np.testing.assert_allclose(window(reader)[0], np.array(codes) * 1e-6, atol=1e-15)
+    np.testing.assert_allclose(window(reader)[1], (np.array(codes) * .2 + 10) * .001)
+    assert channels[2]["quantity"] == "unknown"
+    assert channels[2]["unit"] == "Ohm"
+    np.testing.assert_array_equal(window(reader, channels=[channels[2]["channel_id"], channels[0]["channel_id"]],
+                                        representation="digital"), [[1] * 5, codes])
+
+
+def test_equivalent_volt_and_microvolt_and_negative_gain(tmp_path):
+    path = write_edf_bdf(tmp_path / "units.edf", [
+        {"label": "V", "records": [[-10, 0, 10]], "unit": "V", "physical_min": -.001,
+         "physical_max": .001, "digital_min": -10, "digital_max": 10},
+        {"label": "uV", "records": [[-10, 0, 10]], "unit": "uV", "physical_min": -1000,
+         "physical_max": 1000, "digital_min": -10, "digital_max": 10},
+        {"label": "inverted", "records": [[-10, 0, 10]], "unit": "uV", "physical_min": 1000,
+         "physical_max": -1000, "digital_min": -10, "digital_max": 10}])
+    data = window(EdfBdfReader(path, "units"))
+    np.testing.assert_array_equal(data[0], data[1])
+    np.testing.assert_array_equal(data[0], -data[2])
+
+
+def test_mixed_sampling_rates_auxiliaries_are_not_resampled(tmp_path):
+    path = write_edf_bdf(tmp_path / "mixed.edf", [
+        {"label": "EEG C3", "records": [[0, 1, 2, 3], [4, 5, 6, 7]]},
+        {"label": "ECG", "records": [[90, 91], [92, 93]], "unit": ""},
+        {"label": "TRIG", "records": [[7, 6, 5, 4], [3, 2, 1, 0]], "unit": ""}])
+    reader = EdfBdfReader(path, "mixed")
+    streams = reader.describe()["streams"]
+    assert [x["sampling_frequency_hz"] for x in streams] == [4, 2]
+    assert [[c["source_index"] for c in s["channels"]] for s in streams] == [[0, 2], [1]]
+    np.testing.assert_array_equal(window(reader, start=3, stop=6, representation="digital"), [[3, 4, 5], [4, 3, 2]])
+    np.testing.assert_array_equal(window(reader, stream=1, representation="digital"), [[90, 91, 92, 93]])
+
+
+@pytest.mark.parametrize("bdf", [False, True])
+def test_nonzero_origin_gaps_events_and_utf8(tmp_path, bdf):
+    annotations = [tal("0.25") + tal("0.75", "刺激", "0.5"), tal("1.25"),
+                   tal("4.25") + tal("4.75", "again") + tal("4.75", "again")]
+    path = write_edf_bdf(tmp_path / "gaps.dat", [{"label": "A1-A2", "records": [[1, 2], [3, 4], [5, 6]]}],
+                         bdf=bdf, reserved=("BDF" if bdf else "EDF") + "+D", annotation_records=annotations)
+    reader = EdfBdfReader(path, "gaps")
+    description = reader.describe()
+    spans = description["streams"][0]["segments"]
+    assert [(x["start_sample"], x["stop_sample"], x["recording_start_seconds"]) for x in spans] == [(0, 4, .25), (4, 6, 4.25)]
+    np.testing.assert_array_equal(window(reader, segment=1, representation="digital"), [[5, 6]])
+    with pytest.raises(EphysReaderError):
+        window(reader, start=3, stop=5)
+    events = description["events"]
+    assert [x["text"] for x in events] == ["刺激", "again", "again"]
+    assert events[0]["duration_seconds"] == .5
+    assert len({x["event_id"] for x in events}) == 3
+    assert events[0]["sample_mappings"][0]["sample_index"] == 1
+    assert events[1]["sample_mappings"][0]["sample_index"] == 5
+
+
+def test_continuous_flag_cannot_erase_gap(tmp_path):
+    path = write_edf_bdf(tmp_path / "bad_c.edf", [{"label": "C3", "records": [[1], [2]]}],
+                         reserved="EDF+C", annotation_records=[tal(0), tal(4)])
+    description = EdfBdfReader(path, "bad_c").describe()
+    assert len(description["streams"][0]["segments"]) == 2
+    assert "continuous_flag_has_gaps" in {w["code"] for w in description["warnings"]}
+
+
+def test_real_vendor_missing_nul_and_non_utf8_are_explicit(tmp_path):
+    raw = b"+0.000000\x14\x14+0.000000\x14Segment:\xd6\xd0\xce\xc4\x14\x00"
+    path = write_edf_bdf(tmp_path / "vendor.edf", [{"label": "EEG", "records": [[1, 2]]}],
+                         reserved="EDF+D", annotation_records=[raw])
+    description = EdfBdfReader(path, "vendor").describe()
+    assert {w["code"] for w in description["warnings"]} >= {"timekeeping_missing_nul_separator", "annotation_non_utf8"}
+    event = description["events"][0]
+    assert event["text"] is None
+    assert event["encoding_status"] == "undecoded_non_utf8"
+    assert base64.b64decode(event["raw_text_base64"]) == b"Segment:\xd6\xd0\xce\xc4"
+    assert event["interpretation_status"] == "vendor_recovered_ambiguous"
+    assert base64.b64decode(event["raw_source_tal_base64"]) == raw
+    assert event["alternative_standard_interpretation"]["onset_seconds"] == 0
+
+
+def test_timestamp_like_annotation_outside_record_is_not_auto_repaired(tmp_path):
+    raw = b"+0\x14\x14+123\x14literal text\x14\0"
+    path = write_edf_bdf(tmp_path / "counterexample.edf", [{"label": "EEG", "records": [[1, 2]]}],
+                         reserved="EDF+D", annotation_records=[raw])
+    description = EdfBdfReader(path, "counterexample").describe()
+    assert [event["text"] for event in description["events"]] == ["+123", "literal text"]
+    assert [event["onset_seconds"] for event in description["events"]] == [0, 0]
+    assert all(event["interpretation_status"] == "standard" for event in description["events"])
+    assert "timekeeping_missing_nul_separator" not in {warning["code"] for warning in description["warnings"]}
+
+
+@pytest.mark.parametrize("stop", [4, 100])
+def test_tail_window_never_zero_pads(tmp_path, stop):
+    path = write_edf_bdf(tmp_path / "short.edf", [{"label": "C3", "records": [[1, 2, 3]]}])
+    reader = EdfBdfReader(path, "short")
+    assert window(reader, start=2, stop=3).shape == (1, 1)
+    with pytest.raises(EphysReaderError):
+        window(reader, start=2, stop=stop)
+
+
+@pytest.mark.parametrize("kind", ["truncated", "extra", "zero_calibration", "missing_tal", "backward_time"])
+def test_invalid_input_is_not_repaired_silently(tmp_path, kind):
+    channel = {"label": "C3", "records": [[1, 2], [3, 4]]}
+    options = {}
+    if kind == "zero_calibration":
+        channel.update(physical_min=1, physical_max=1)
+    if kind in ("missing_tal", "backward_time"):
+        options = {"reserved": "EDF+D", "annotation_records": [tal(1), b"" if kind == "missing_tal" else tal(0)]}
+    path = write_edf_bdf(tmp_path / "bad.edf", [channel], **options)
+    if kind == "truncated":
+        path.write_bytes(path.read_bytes()[:-1])
+    if kind == "extra":
+        path.write_bytes(path.read_bytes() + b"extra")
+    with pytest.raises(EphysReaderError):
+        EdfBdfReader(path, "bad")
+
+
+def test_unknown_record_count_is_named_and_file_change_is_detected(tmp_path):
+    path = write_edf_bdf(tmp_path / "unknown.edf", [{"label": "C3", "records": [[1, 2]]}], record_count=-1)
+    reader = EdfBdfReader(path, "unknown")
+    assert "record_count_from_complete_payload" in {w["code"] for w in reader.describe()["warnings"]}
+    path.write_bytes(path.read_bytes()[:-2] + b"\0\0")
+    with pytest.raises(EphysReaderError, match="changed"):
+        window(reader)
+
+
+def test_zero_duration_annotation_only_has_no_fake_frequency(tmp_path):
+    path = write_edf_bdf(tmp_path / "events.edf", [], reserved="EDF+D", duration="0",
+                         annotation_records=[tal(0) + tal(42, "event")])
+    description = EdfBdfReader(path, "events").describe()
+    assert description["streams"] == []
+    assert description["events"][0]["onset_seconds"] == 42
+    assert description["events"][0]["sample_mappings"] == []
+
+
+@pytest.mark.parametrize("physical_unit", [b"\xc2\xb5V", b"\xb5V", b"\xce\xbcV"])
+def test_explicit_microvolt_spellings_are_scaled_and_audited(tmp_path, physical_unit):
+    path = write_edf_bdf(tmp_path / "unit.edf", [{"label": "C3", "records": [[10, -20]], "unit": physical_unit}])
+    reader = EdfBdfReader(path, "unit")
+    np.testing.assert_allclose(window(reader), [[10e-6, -20e-6]])
+    assert "header_non_ascii" in {w["code"] for w in reader.describe()["warnings"]}
+
+
+def test_fractional_and_gap_events_do_not_get_rounded_to_samples(tmp_path):
+    path = write_edf_bdf(tmp_path / "events.edf", [{"label": "C3", "records": [[1, 2], [3, 4]]}],
+                         reserved="EDF+D", annotation_records=[tal(0) + tal("0.1", "fraction") + tal(2, "gap"), tal(4)])
+    events = EdfBdfReader(path, "fraction").describe()["events"]
+    assert events[0]["sample_mappings"][0]["sample_position"] == .2
+    assert events[0]["sample_mappings"][0]["sample_index"] is None
+    assert events[1]["sample_mappings"] == []
+
+
+@pytest.mark.parametrize("selection", [[0], [["not hashable"]], [], ["absent"]])
+def test_invalid_channel_selection_has_contract_error(tmp_path, selection):
+    path = write_edf_bdf(tmp_path / "short.edf", [{"label": "C3", "records": [[1]]}])
+    reader = EdfBdfReader(path, "short")
+    stream = reader.describe()["streams"][0]
+    with pytest.raises(EphysReaderError):
+        reader.read_window(stream["stream_id"], stream["segments"][0]["segment_id"], selection, 0, 1)
+
+
+@pytest.mark.parametrize("calibration", ["NaN", "Infinity", "1e999"])
+def test_nonfinite_calibration_cannot_become_voltage(tmp_path, calibration):
+    path = write_edf_bdf(tmp_path / "bad.edf", [{"label": "C3", "records": [[1]], "physical_max": calibration}])
+    with pytest.raises(EphysReaderError):
+        EdfBdfReader(path, "bad")
+
+
+@pytest.mark.parametrize("options", [
+    {"reserved": "EDF+D"},
+    {"reserved": "EDF+Z"},
+    {"reserved": "BDF+D"},
+    {"duration": "1e-309"},
+    {"duration": "1e-999"},
+])
+def test_invalid_clock_cannot_silently_become_continuous(tmp_path, options):
+    path = write_edf_bdf(tmp_path / "bad_clock.edf", [{"label": "C3", "records": [[1]]}], **options)
+    with pytest.raises(EphysReaderError):
+        EdfBdfReader(path, "bad_clock")
+
+
+def test_multiple_annotation_channels_keep_same_sample_events_distinct(tmp_path):
+    path = write_edf_bdf(tmp_path / "events.edf", [
+        {"label": "C3", "records": [[1, 2]]},
+        {"label": "EDF Annotations", "annotation": True, "samples": 64, "records": [tal(0) + tal("0.5", "event")]},
+        {"label": "EDF Annotations", "annotation": True, "samples": 64, "records": [tal("0.5", "event")]}],
+        reserved="EDF+C")
+    reader = EdfBdfReader(path, "multi-ann")
+    events = reader.describe()["events"]
+    assert len(events) == 2
+    assert events[0]["event_id"] != events[1]["event_id"]
+    assert [event["annotation_channel_index"] for event in events] == [1, 2]
+    assert [event["sample_mappings"][0]["sample_index"] for event in events] == [1, 1]
diff --git a/backend/tests/contract/test_ephys_fixture_baseline.py b/backend/tests/contract/test_ephys_fixture_baseline.py
new file mode 100644
index 0000000000000000000000000000000000000000..7a27e44e491fd1aa8b03f1934344dd07bb16134e
--- /dev/null
+++ b/backend/tests/contract/test_ephys_fixture_baseline.py
@@ -0,0 +1,126 @@
+"""S0 ground-truth examples: no format reader or QC algorithm is exercised."""
+import hashlib
+import json
+
+import numpy as np
+import pytest
+
+from backend.tests.ephys_fixtures import build_ephys_fixture
+
+
+@pytest.fixture
+def example(tmp_path, isolated_storage):
+    return build_ephys_fixture(tmp_path / "fixture")
+
+
+def test_byte_rebuild_and_file_fingerprints(example, tmp_path):
+    rebuilt = build_ephys_fixture(tmp_path / "rebuilt")
+    assert example["manifest_path"].read_bytes() == rebuilt["manifest_path"].read_bytes()
+    for item in example["manifest"]["files"]:
+        original = example["root"] / item["path"]
+        assert original.read_bytes() == (rebuilt["root"] / item["path"]).read_bytes()
+        assert original.stat().st_size == item["size_bytes"]
+        assert hashlib.sha256(original.read_bytes()).hexdigest() == item["sha256"]
+    assert json.loads(example["manifest_path"].read_text()) == example["manifest"]
+    assert example["manifest"]["synthetic"] is True
+    with pytest.raises(FileExistsError):
+        build_ephys_fixture(example["root"])
+
+
+def test_voltage_equivalence_does_not_rescale_digital_channel(example):
+    stream = example["manifest"]["streams"][0]
+    assert stream["sampling_frequency_hz"] == 256
+    assert {channel["type"] for channel in stream["channels"]} == {"EEG", "SEEG", "ECOG", "ECG", "TRIG"}
+    for segment in stream["segments"]:
+        uv = np.load(example["root"] / segment["arrays"]["uV"], allow_pickle=False)
+        volts = np.load(example["root"] / segment["arrays"]["V"], allow_pickle=False)
+        assert uv.shape == (11, segment["n_samples"])
+        np.testing.assert_allclose(volts[:-1], uv[:-1] * 1e-6, rtol=0, atol=1e-12)
+        np.testing.assert_array_equal(volts[-1], uv[-1])
+    volts = np.load(example["root"] / "seg-0-V.npy", allow_pickle=False)
+    assert volts[0, 16] == pytest.approx(-25e-6, abs=1e-12)
+    assert volts[0, 17] == pytest.approx(75e-6, abs=1e-12)
+    assert volts[-1, 101] == 1.0
+    assert volts[-1, 101] != pytest.approx(1e-6)  # not a voltage row
+
+
+def test_nonzero_origin_gap_and_channel_local_interval(example):
+    stream = example["manifest"]["streams"][0]
+    first, second = stream["segments"]
+    assert first["start_sample"] == 1000
+    assert first["stop_sample"] == 1256
+    assert second["start_sample"] == 1500
+    assert second["recording_start_seconds"] == 500 / 256
+    assert sum(segment["n_samples"] for segment in stream["segments"]) == 448
+    assert second["stop_sample"] - first["start_sample"] == 692
+    # Concatenating into a fictitious continuous timeline erases 244 samples.
+    assert second["recording_start_seconds"] != first["n_samples"] / 256
+    uv = np.load(example["root"] / first["arrays"]["uV"], allow_pickle=False)
+    np.testing.assert_array_equal(uv[0, 99:117],
+                                  np.array([-5] + list((np.arange(100, 116) % 16 - 8) + 500) + [-4]))
+    np.testing.assert_array_equal(uv[1, 100:116], ((np.arange(100, 116) + 3) % 16 - 8) * 2)
+    interval = example["expected"]["bad_intervals"][0]
+    assert interval["channel_ids"] == ["C3"]
+    assert (interval["start_sample"], interval["stop_sample"]) == (1100, 1116)
+    assert uv[-1, 101] == 1  # a C3 mask must not erase a valid trigger at the same time
+
+
+def test_missing_sites_and_recorded_bipolar_are_not_inferred_channels(example):
+    channels = example["manifest"]["streams"][0]["channels"]
+    by_id = {channel["channel_id"]: channel for channel in channels}
+    assert "Pz" not in by_id and "A2" not in by_id
+    assert "Pz" in example["manifest"]["streams"][0]["expected_acquisition_channel_ids"]
+    assert by_id["A1-A2"]["measurement"] == "recorded_bipolar"
+    assert by_id["A1-A2"]["electrode_ids"] == ["A1", "A2"]
+    assert by_id["A1-A2"]["electrode_weights"] == {"A1": 1.0, "A2": -1.0}
+    assert by_id["A1-A2"]["monopolar_reconstruction_available"] is False
+    assert [c["channel_id"] for c in channels if c["type"] == "SEEG"] == ["A1", "A3", "A1-A2"]
+    assert ["A1", "A3"] in example["expected"]["not_inferred_bipolar_pairs"]
+    physical_ids = {site["electrode_id"] for site in example["manifest"]["streams"][0]["physical_electrodes"]}
+    assert "A2" in physical_ids and "A1-A2" not in physical_ids
+
+
+def test_grid_adjacency_is_not_storage_row_adjacency(example):
+    grid_channels = [c for c in example["manifest"]["streams"][0]["channels"] if c["type"] == "ECOG"]
+    assert [c["channel_id"] for c in grid_channels] == ["G3", "G1", "G4", "G2"]
+    edges = {frozenset(edge) for edge in example["expected"]["ecog_adjacency"]}
+    assert frozenset(("G1", "G4")) not in edges  # adjacent rows, diagonal sites
+    assert frozenset(("G1", "G2")) in edges  # non-adjacent rows, adjacent sites
+    positions = example["expected"]["ecog_grid_positions"]
+    for a, b in example["expected"]["ecog_adjacency"]:
+        assert sum(abs(x - y) for x, y in zip(positions[a], positions[b])) == 1
+
+
+def test_bipolar_only_input_does_not_claim_recoverable_monopolar_voltage(example):
+    stream = example["manifest"]["streams"][2]
+    assert [channel["channel_id"] for channel in stream["channels"]] == ["A1-A2"]
+    assert stream["monopolar_channels_available"] is False
+    assert stream["physical_electrode_ids"] == ["A1", "A2"]
+    data = np.load(example["root"] / stream["segments"][0]["arrays"]["V"], allow_pickle=False)
+    assert data.shape == (1, 256)
+    # Two distinct pairs of site voltages yield the same recorded difference.
+    first_a, first_b = data[0] + 10e-6, np.full(256, 10e-6)
+    second_a, second_b = first_a + 20e-6, first_b + 20e-6
+    np.testing.assert_allclose(first_a - first_b, second_a - second_b, rtol=0, atol=1e-12)
+    assert not np.array_equal(first_a, second_a)
+
+
+def test_spike_identity_preserves_coincident_events_and_unknown_clock_alignment(example):
+    stream = example["manifest"]["streams"][1]
+    arrays = {name: np.load(example["root"] / path, allow_pickle=False)
+              for name, path in stream["spike_arrays"].items()}
+    assert stream["sampling_frequency_hz"] == 30000
+    assert stream["clock_alignment_to_mixed"] == "unknown"
+    assert stream["time_origin_seconds"] is None
+    assert len(set(arrays["spike_id"])) == 6
+    coincidence = arrays["sample"] == 31032
+    assert arrays["spike_id"][coincidence].tolist() == ["spike-a0", "spike-b0"]
+    assert arrays["unit_id"][coincidence].tolist() == [1, 2]
+    assert len(np.unique(arrays["sample"])) == 5  # dedup by time alone destroys a distinct event
+    assert example["expected"]["no_dedup_retained_spike_ids"] == arrays["spike_id"].tolist()
+    decision = example["expected"]["explicit_exclusion_example"]
+    assert decision["removed_spike_ids"] == [entry["spike_id"] for entry in decision["audit"]]
+    assert set(decision["retained_spike_ids"]) | set(decision["removed_spike_ids"]) == set(arrays["spike_id"])
+    assert "spike-a0" in decision["retained_spike_ids"]
+    for sample in arrays["sample"]:
+        assert sum(segment["start_sample"] <= sample < segment["stop_sample"] for segment in stream["segments"]) == 1
diff --git a/backend/tests/contract/test_ephys_migration.py b/backend/tests/contract/test_ephys_migration.py
new file mode 100644
index 0000000000000000000000000000000000000000..9ce3ba5c797a5aca65cd3ed00591558a799db684
--- /dev/null
+++ b/backend/tests/contract/test_ephys_migration.py
@@ -0,0 +1,147 @@
+"""S1/S2 upgrade compatibility, independent of a production database."""
+
+from pathlib import Path
+import sqlite3
+
+import pytest
+
+from backend.app.migrations.ephys_recordings_v1 import upgrade as upgrade_v1
+from backend.app.migrations.ephys_artifacts_v2 import upgrade as upgrade_v2
+
+SCHEMA = Path(__file__).parents[2] / "app" / "schema.sql"
+
+
+def schema_contract(connection, table):
+    return {
+        "columns": connection.execute(f"PRAGMA table_info({table})").fetchall(),
+        "foreign_keys": connection.execute(f"PRAGMA foreign_key_list({table})").fetchall(),
+        "indexes": connection.execute(f"PRAGMA index_list({table})").fetchall(),
+        "index_columns": {
+            row[1]: connection.execute(f'PRAGMA index_info("{row[1]}")').fetchall()
+            for row in connection.execute(f"PRAGMA index_list({table})").fetchall()
+        },
+    }
+
+
+def test_repeatable_upgrade_preserves_legacy_rows_and_matches_fresh_schema():
+    old = sqlite3.connect(":memory:")
+    fresh = sqlite3.connect(":memory:")
+    schema = SCHEMA.read_text()
+    old.executescript(schema.split("-- S1: stable QC identities")[0] + "COMMIT;")
+    old.execute("INSERT INTO subjects(subject_id,created_at,updated_at) VALUES('S01','before','before')")
+    old.execute("INSERT INTO subject_assets(subject_id,asset_type,storage_path,created_at,updated_at) "
+                "VALUES(1,'eeg','resourses/S01/old.edf','before','before')")
+    old.execute("INSERT INTO eeg_files(subject_id,asset_id,label,raw_path,processed_path,created_at) "
+                "VALUES(1,1,'same','resourses/S01/old.edf','output/S01/replay.bin','before')")
+    old.commit()
+    before = {table: old.execute(f"SELECT * FROM {table}").fetchall()
+              for table in ("subjects", "subject_assets", "eeg_files")}
+    upgrade_v1(old)
+    upgrade_v2(old)
+    old.commit()
+    upgrade_v1(old)
+    upgrade_v2(old)
+    old.commit()
+    fresh.executescript(schema)
+    for table, rows in before.items():
+        assert old.execute(f"SELECT * FROM {table}").fetchall() == rows
+    for table in ("ephys_recordings", "ephys_artifacts"):
+        assert schema_contract(old, table) == schema_contract(fresh, table)
+    assert old.execute("PRAGMA integrity_check").fetchone() == ("ok",)
+    old.close()
+    fresh.close()
+
+
+def populated_v1():
+    conn = sqlite3.connect(":memory:")
+    conn.execute("PRAGMA foreign_keys = ON")
+    conn.executescript(SCHEMA.read_text().split("-- S1: stable QC identities")[0] + "COMMIT;")
+    conn.execute("INSERT INTO subjects(subject_id,created_at,updated_at) VALUES('S01','before','before')")
+    conn.execute("INSERT INTO eeg_files(subject_id,label,raw_path,processed_path,created_at) "
+                 "VALUES(1,'same','original.edf','legacy-replay.bin','before')")
+    upgrade_v1(conn)
+    conn.execute("INSERT INTO ephys_recordings VALUES(?,?,?,?,?,?,?,?,?,?)",
+                 ("recording-original", 1, "same", "EEG", "session-1", "input-original",
+                  "import-key", "request-hash", '{"source":"original"}', "before"))
+    conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                 ("input-original", "recording-original", "raw_input", "raw/original",
+                  "content-hash", "manifest-hash", 2, 1234, "before"))
+    conn.commit()
+    return conn
+
+
+def test_populated_v1_preserves_ids_rows_and_legacy_while_allowing_derived_versions():
+    conn = populated_v1()
+    tables = ("subjects", "eeg_files", "ephys_recordings", "ephys_artifacts")
+    before = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
+    for _ in range(2):
+        # init_db executes the latest CREATE IF NOT EXISTS schema before upgrades.
+        conn.executescript(SCHEMA.read_text())
+        upgrade_v1(conn)
+        upgrade_v2(conn)
+        conn.commit()
+        for table, rows in before.items():
+            assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
+    assert len(conn.execute("PRAGMA table_info(ephys_artifacts)").fetchall()) == 9
+    for artifact_id in ("derived-1", "derived-2"):
+        conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                     (artifact_id, "recording-original", "derived_voltage", artifact_id,
+                      "derived-content", "derived-manifest", 1, 600, "after"))
+    conn.commit()
+    upgraded = conn.execute("SELECT * FROM ephys_artifacts ORDER BY artifact_id").fetchall()
+    upgrade_v2(conn)
+    assert conn.execute("SELECT * FROM ephys_artifacts ORDER BY artifact_id").fetchall() == upgraded
+    with pytest.raises(sqlite3.IntegrityError):
+        conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                     ("input-duplicate", "recording-original", "raw_input", "raw/duplicate",
+                      "other", "other", 1, 1, "after"))
+    with pytest.raises(sqlite3.IntegrityError):
+        conn.execute("INSERT INTO ephys_artifacts VALUES(?,?,?,?,?,?,?,?,?)",
+                     ("unknown-kind", "recording-original", "unreviewed_other", "other/path",
+                      "other", "other", 1, 1, "after"))
+    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
+    assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
+    conn.execute("DELETE FROM subjects WHERE id=1")
+    assert conn.execute("SELECT * FROM ephys_artifacts").fetchall() == []
+    conn.close()
+
+
+def test_v2_respects_callers_transaction_and_dict_row_factory():
+    conn = populated_v1()
+    conn.row_factory = lambda cursor, row: {col[0]: row[i] for i, col in enumerate(cursor.description)}
+    original = conn.execute("SELECT * FROM ephys_artifacts").fetchall()
+    conn.execute("INSERT INTO subjects(subject_id,created_at,updated_at) VALUES('pending','after','after')")
+    upgrade_v2(conn)
+    assert conn.in_transaction
+    conn.rollback()
+    assert conn.execute("SELECT id FROM subjects WHERE subject_id='pending'").fetchall() == []
+    assert conn.execute("SELECT * FROM ephys_artifacts").fetchall() == original
+    assert "derived_voltage" not in conn.execute("SELECT sql FROM sqlite_master WHERE name='ephys_artifacts'").fetchone()["sql"]
+    upgrade_v2(conn)
+    assert conn.execute("SELECT * FROM ephys_artifacts").fetchall() == original
+    conn.close()
+
+
+def test_v2_rebuild_failure_rolls_back_without_losing_input_or_legacy_rows():
+    conn = populated_v1()
+    before = {table: conn.execute(f"SELECT * FROM {table}").fetchall()
+              for table in ("subjects", "eeg_files", "ephys_recordings", "ephys_artifacts")}
+    schema_before = schema_contract(conn, "ephys_artifacts")
+
+    def deny_drop(action, table, _column, _database, _trigger):
+        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TABLE and table == "ephys_artifacts" else sqlite3.SQLITE_OK
+
+    conn.set_authorizer(deny_drop)
+    with pytest.raises(sqlite3.DatabaseError):
+        upgrade_v2(conn)
+    # Python 3.9's sqlite wrapper keeps a callback trampoline for None and then
+    # calls it, producing TypeError → "not authorized" on the next SELECT.
+    # Restore an allow-all authorizer without changing the migration under test.
+    conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
+    for table, rows in before.items():
+        assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
+    assert schema_contract(conn, "ephys_artifacts") == schema_before
+    assert conn.execute("SELECT name FROM sqlite_master WHERE name='ephys_artifacts_v2_new'").fetchone() is None
+    upgrade_v2(conn)
+    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
+    conn.close()
diff --git a/backend/tests/contract/test_ephys_npy_reader.py b/backend/tests/contract/test_ephys_npy_reader.py
new file mode 100644
index 0000000000000000000000000000000000000000..0e68c8933a2a992254b3263915f1c18d8d77bdb0
--- /dev/null
+++ b/backend/tests/contract/test_ephys_npy_reader.py
@@ -0,0 +1,132 @@
+"""Derived voltage windows retain identity, SI units and nonzero stored origins."""
+import numpy as np
+import pytest
+
+from backend.app.processing.ephys_qc.readers.edf import EphysReaderError
+from backend.app.processing.ephys_qc.readers.npy import NpyVoltageReader
+
+
+def stream():
+    return {"stream_id": "derived:stream", "source_id": "derived:source", "clock_id": "parent:clock",
+            "sampling_frequency_hz": 2, "sample_index_origin": 100, "sample_count": 6,
+            "channels": [{"channel_id": "c0", "raw_name": "same", "unit": "V", "source_channel_ids": ["parent:c0"]},
+                         {"channel_id": "c1", "raw_name": "same", "unit": "V", "source_channel_ids": ["parent:c1"]}],
+            "segments": [{"segment_id": "s0", "start_sample": 100, "stop_sample": 104,
+                          "recording_start_seconds": .25, "parent_stream_id": "parent:stream", "parent_segment_id": "parent:s0",
+                          "parent_start_sample": 10, "parent_stop_sample": 14, "clock_mapping": "identity"},
+                         {"segment_id": "s1", "start_sample": 104, "stop_sample": 106,
+                          "recording_start_seconds": 10.25, "parent_stream_id": "parent:stream", "parent_segment_id": "parent:s1",
+                          "parent_start_sample": 14, "parent_stop_sample": 16, "clock_mapping": "identity"}]}
+
+
+def fixture(tmp_path, data=None, metadata=None):
+    data = np.array([[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]], dtype="<f8") * 1e-6 if data is None else data
+    path = tmp_path / "voltage.npy"
+    np.save(path, data, allow_pickle=False)
+    return path, stream() if metadata is None else metadata, data
+
+
+def test_nonzero_origin_duplicate_names_order_and_independent_copy(tmp_path):
+    path, metadata, data = fixture(tmp_path)
+    reader = NpyVoltageReader(path, metadata)
+    observed = reader.read_window("derived:stream", "s0", ["c1", "c0"], 101, 104)
+    np.testing.assert_array_equal(observed, data[[1, 0], 1:4])
+    assert observed.flags.c_contiguous and observed.dtype == np.dtype("<f8")
+    observed[:] = 0
+    np.testing.assert_array_equal(reader.read_window("derived:stream", "s1", ["c0"], 104, 106), data[[0], 4:6])
+    np.testing.assert_array_equal(np.load(path, allow_pickle=False), data)
+    metadata["channels"][0]["unit"] = "unknown"
+    assert reader.describe()["streams"][0]["channels"][0]["unit"] == "V"
+
+
+@pytest.mark.parametrize("first,last,segment", [(99, 101, "s0"), (103, 105, "s0"), (104, 107, "s1"), (104, 104, "s1")])
+def test_no_tail_padding_or_gap_crossing(tmp_path, first, last, segment):
+    path, metadata, _ = fixture(tmp_path)
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(path, metadata).read_window("derived:stream", segment, ["c0"], first, last)
+
+
+def test_no_adc_codes_for_derived_voltage(tmp_path):
+    path, metadata, _ = fixture(tmp_path)
+    with pytest.raises(EphysReaderError, match="physical"):
+        NpyVoltageReader(path, metadata).read_window("derived:stream", "s0", ["c0"], 100, 101, representation="digital")
+
+
+@pytest.mark.parametrize("data", [np.zeros((2, 6), dtype="<f4"), np.zeros((2, 6), dtype=">f8"),
+                                   np.zeros((6, 2), dtype="<f8"), np.zeros((2, 6, 1), dtype="<f8"),
+                                   np.asfortranarray(np.zeros((2, 6), dtype="<f8"))])
+def test_wrong_shape_dtype_or_order_are_rejected(tmp_path, data):
+    path, metadata, _ = fixture(tmp_path, data)
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(path, metadata)
+
+
+@pytest.mark.parametrize("mutation", ["unknown_unit", "duplicate_id", "missing_parent", "wrong_parent_length", "sample_hole", "overlap_clock"])
+def test_descriptor_contract_cannot_be_guessed(tmp_path, mutation):
+    metadata = stream()
+    if mutation == "unknown_unit":
+        metadata["channels"][0]["unit"] = "uV"
+    elif mutation == "duplicate_id":
+        metadata["channels"][1]["channel_id"] = "c0"
+    elif mutation == "missing_parent":
+        metadata["channels"][0]["source_channel_ids"] = []
+    elif mutation == "wrong_parent_length":
+        metadata["segments"][0]["parent_stop_sample"] = 15
+    elif mutation == "sample_hole":
+        metadata["segments"][1]["start_sample"] = 105
+    else:
+        metadata["segments"][1]["recording_start_seconds"] = .5
+    path, _, _ = fixture(tmp_path)
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(path, metadata)
+
+
+def test_npy_mmap_does_not_scan_all_values_and_rejects_nonfinite_window(tmp_path):
+    path, metadata, data = fixture(tmp_path)
+    mapped = np.lib.format.open_memmap(path, mode="r+")
+    mapped[1, 5] = np.nan
+    mapped.flush()
+    del mapped
+    reader = NpyVoltageReader(path, metadata)
+    assert isinstance(reader._data, np.memmap)
+    assert reader._data.mode == "r"
+    np.testing.assert_array_equal(reader.read_window("derived:stream", "s0", ["c0", "c1"], 100, 104), data[:, :4])
+    with pytest.raises(EphysReaderError, match="finite"):
+        reader.read_window("derived:stream", "s1", ["c1"], 104, 106)
+
+
+@pytest.mark.parametrize("change", ["truncated", "extra", "modified_after_open"])
+def test_exact_payload_and_file_identity_are_required(tmp_path, change):
+    path, metadata, _ = fixture(tmp_path)
+    if change == "modified_after_open":
+        reader = NpyVoltageReader(path, metadata)
+        mapped = np.lib.format.open_memmap(path, mode="r+")
+        mapped[0, 0] = 10
+        mapped.flush()
+        del mapped
+        with pytest.raises(EphysReaderError, match="changed"):
+            reader.read_window("derived:stream", "s0", ["c0"], 100, 101)
+    else:
+        raw = path.read_bytes()
+        path.write_bytes(raw[:-1] if change == "truncated" else raw + b"extra")
+        with pytest.raises(EphysReaderError):
+            NpyVoltageReader(path, metadata)
+
+
+def test_object_arrays_and_archives_never_load_as_voltage(tmp_path):
+    path = tmp_path / "object.npy"
+    np.save(path, np.array([["not voltage"]], dtype=object), allow_pickle=True)
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(path, stream())
+    archive = tmp_path / "archive.npz"
+    np.savez(archive, voltage=np.zeros((2, 6)))
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(archive, stream())
+
+
+@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), 10 ** 1000, 1e-309, True])
+def test_invalid_sampling_clock_is_rejected(tmp_path, frequency):
+    path, metadata, _ = fixture(tmp_path)
+    metadata["sampling_frequency_hz"] = frequency
+    with pytest.raises(EphysReaderError):
+        NpyVoltageReader(path, metadata)
diff --git a/backend/tests/contract/test_ephys_qc_review.py b/backend/tests/contract/test_ephys_qc_review.py
new file mode 100644
index 0000000000000000000000000000000000000000..8d528c4b04316a55a78bcf04e6ffc04592ef3eaa
--- /dev/null
+++ b/backend/tests/contract/test_ephys_qc_review.py
@@ -0,0 +1,295 @@
+"""Real EDF scan → audited interval decisions → immutable internal QC package."""
+import hashlib
+import csv
+import io
+import json
+import zipfile
+
+import numpy as np
+import pytest
+from flask import Flask
+from werkzeug.datastructures import FileStorage
+
+from backend.app.config import settings
+from backend.app.db import get_db
+from backend.app.domain import ephys_qc as qc
+from backend.app.domain.ephys_recordings import import_recording
+from backend.app.domain.ephys_storage import ImportConflict
+from backend.app.domain.subjects import create_subject
+from backend.tests.ephys_edf_fixtures import tal, write_edf_bdf
+
+
+@pytest.fixture
+def scanned(isolated_storage, tmp_path):
+    create_subject("SEEG-QC", "technical fixture", None)
+    wave = np.rint(50 * np.sin(np.arange(256) * 2 * np.pi * 13 / 256)).astype(int).tolist()
+    source = write_edf_bdf(tmp_path / "test.edf", [
+        {"label": "A'1", "records": [wave, wave, wave]},
+        {"label": "A'3", "records": [wave, [0] * 256, wave]}],
+        reserved="EDF+D", annotation_records=[tal(0), tal(1), tal(5)])
+    record, _ = import_recording({"subject_id": "SEEG-QC", "label": "fixture", "modality": "seeg"},
+                                [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="test.edf")])
+    run = qc.create_run(record["recording_id"], {})
+    qc.execute_scan(run["run_id"])
+    return record, qc.get_run(run["run_id"])
+
+
+def decision(run, channel=0, action="exclude", start=0, stop=128, **extra):
+    stream = run["report"]["catalog"]["streams"][0]
+    return {"channel_id": stream["channels"][channel]["channel_id"],
+            "stream_id": stream["stream_id"], "segment_id": stream["segments"][0]["segment_id"],
+            "start_sample": start, "stop_sample": stop, "action": action,
+            "reason": "Reviewed raw evidence", **extra}
+
+
+def save(run, decisions, revision=None):
+    return qc.save_review(run["run_id"], {"base_revision": run["review"]["revision"] if revision is None else revision,
+                                         "reviewer": "reviewer-one", "decisions": decisions})
+
+
+def test_real_scan_persists_and_unreviewed_is_unknown(scanned):
+    record, run = scanned
+    assert run["status"] == "succeeded" and run["report"]["windows"]
+    assert run["report"]["profile"] == "seeg_basic_v1"
+    assert run["review"]["summary"]["unknown_ratio"] == 1
+    assert run["review"]["summary"]["retained_percent"] is None
+    assert run["review"]["summary"]["reviewed_score"] == run["report"]["initial_score"]["total"]
+    assert qc.get_run(run["run_id"]) == run
+    assert qc.history(record["recording_id"])["current"]["run_id"] == run["run_id"]
+
+
+def test_overlap_union_is_channel_scoped_and_revision_conflicts(scanned):
+    _, run = scanned
+    updated = save(run, [decision(run), decision(run, start=64, stop=192)])
+    summary = updated["review"]["summary"]
+    assert summary["channels"][0]["excluded_seconds"] == .75
+    assert summary["channels"][1]["excluded_seconds"] == 0
+    assert summary["unknown_seconds"] == 5.25
+    assert updated["review"]["revision"] == 1
+    with pytest.raises(ImportConflict, match="revision"):
+        save(run, [])
+    with get_db() as conn:
+        assert conn.execute("SELECT COUNT(*) AS n FROM ephys_qc_audit WHERE run_id=?", (run["run_id"],)).fetchone()["n"] >= 2
+
+
+def test_cross_gap_unknown_channel_and_contradictory_decisions_are_rejected(scanned):
+    _, run = scanned
+    for bad in [decision(run, stop=768), decision(run, channel_id="missing"),
+                decision(run, start=0.1), decision(run, reason="")]:
+        with pytest.raises(ValueError):
+            save(run, [bad])
+    with pytest.raises(ValueError, match="conflict"):
+        save(run, [decision(run), decision(run, action="keep")])
+
+
+def test_whole_channel_keep_with_local_exclusion_and_common_time(scanned):
+    _, run = scanned
+    full = [{"channel_id": c["channel_id"], "scope": "channel", "action": "keep", "reason": "All segments reviewed"}
+            for c in run["report"]["catalog"]["streams"][0]["channels"]]
+    updated = save(run, full + [decision(run)])
+    summary = updated["review"]["summary"]
+    assert summary["unknown_ratio"] == 0 and summary["reviewed_ratio"] == 1
+    assert summary["kept_seconds"] == 5.5
+    assert summary["common_usable"][0]["duration_seconds"] == 2.5
+
+
+def test_freeze_readback_and_immutability(scanned):
+    record, run = scanned
+    frozen = qc.freeze_run(run["run_id"], {"base_revision": 0, "reviewer": "reviewer-one"})
+    assert frozen["frozen"] and frozen["review"]["summary"]["unknown_ratio"] == 1
+    assert frozen["assessment"]["status"] == "pending"
+    payload = qc.export_run(run["run_id"])
+    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
+        manifest = json.loads(archive.read("manifest.json"))
+        result = json.loads(archive.read("qc_result.json"))
+        assert result["recording_id"] == record["recording_id"]
+        assert result["scope"] == "seeg_basic_signal_only"
+        for name, checksum in manifest["files"].items():
+            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
+        assert "report.html" in archive.namelist() and "decisions.csv" in archive.namelist()
+    assert qc.export_run(run["run_id"]) == payload
+    with pytest.raises(ImportConflict, match="frozen"):
+        save(frozen, [])
+
+
+def test_waveform_event_export_preserves_stages_and_legacy_member_set(scanned):
+    from copy import deepcopy
+    from backend.app.domain.ephys_qc_export import build_package
+    _, run = scanned
+    frozen = qc.freeze_run(run["run_id"], {"base_revision": 0, "reviewer": "reviewer-one"})
+    event = {"event_id": "event-test", **decision(run, start=64, stop=400),
+             "recording_start_seconds": .25, "recording_stop_seconds": 1.5625,
+             "duration_seconds": 1.3125, "recovery_status": "right_censored",
+             "termination_reason": "range_end", "score_effect": "none", "reasons": ["baseline_shift"],
+             "stages": [{"kind": "jump", "start_sample": 64, "stop_sample": 65}], "features": {}}
+    frozen["report"]["waveform_events"] = [event]
+    with zipfile.ZipFile(io.BytesIO(build_package(frozen))) as archive:
+        rows = list(csv.DictReader(io.StringIO(archive.read("waveform_events.csv").decode("utf-8-sig"))))
+        assert rows[0]["event_id"] == event["event_id"]
+        assert int(rows[0]["stop_sample"]) == 400
+        assert json.loads(rows[0]["stages"]) == event["stages"]
+        assert rows[0]["recovery_status"] == "right_censored"
+        assert "event-test" in archive.read("report.html").decode()
+    legacy = deepcopy(frozen)
+    del legacy["report"]["waveform_events"]
+    with zipfile.ZipFile(io.BytesIO(build_package(legacy))) as archive:
+        assert "waveform_events.csv" not in archive.namelist()
+        assert len(archive.namelist()) == 7
+
+
+def test_input_tampering_prevents_freeze(scanned):
+    record, run = scanned
+    path = settings.dataset_root / record["input_artifact"]["storage_path"] / "files" / "test.edf"
+    path.chmod(0o644)
+    with path.open("r+b") as handle:
+        handle.seek(-1, 2)
+        handle.write(b"X")
+    with pytest.raises(ImportConflict):
+        qc.freeze_run(run["run_id"], {"base_revision": 0, "reviewer": "reviewer-one"})
+    assert not qc.get_run(run["run_id"])["frozen"]
+
+
+def test_failed_scan_is_queryable_and_does_not_modify_original(scanned, monkeypatch):
+    record, _ = scanned
+    from backend.app.processing.ephys_qc import seeg
+    def fail(*args, **kwargs):
+        raise RuntimeError("deliberate scanner failure")
+    monkeypatch.setattr(seeg, "scan_seeg", fail)
+    run = qc.create_run(record["recording_id"], {})
+    with pytest.raises(RuntimeError, match="scanner failure"):
+        qc.execute_scan(run["run_id"])
+    result = qc.get_run(run["run_id"])
+    assert result["status"] == "failed" and "scanner failure" in result["error"]
+    from backend.app.domain.ephys_recordings import verify_recording
+    assert verify_recording(record["recording_id"])["valid"]
+
+
+def test_api_contract_and_job_submission(scanned):
+    from backend.app.api.v1.ephys_qc import ephys_qc_bp
+    record, run = scanned
+    app = Flask(__name__)
+    app.register_blueprint(ephys_qc_bp)
+    queued = []
+    class Runner:
+        def enqueue(self, job_id):
+            queued.append(job_id)
+    app.job_runner = Runner()
+    client = app.test_client()
+    prefix = "/api/v1/ephys"
+    response = client.post(f"{prefix}/recordings/{record['recording_id']}/qc/runs", json={})
+    assert response.status_code == 202
+    assert queued == [response.get_json()["data"]["job_id"]]
+    response = client.patch(f"{prefix}/qc/runs/{run['run_id']}/review",
+                            json={"base_revision": 0, "reviewer": "r", "decisions": [decision(run)]})
+    assert response.status_code == 200
+    response = client.patch(f"{prefix}/qc/runs/{run['run_id']}/review",
+                            json={"base_revision": 0, "reviewer": "r", "decisions": []})
+    assert response.status_code == 409
+
+
+def test_migration_repeat_retains_reviews_and_cancelled_jobs_are_visible(scanned):
+    from backend.app.migrations import ephys_qc_v1
+    record, run = scanned
+    changed = save(run, [decision(run)])
+    with get_db() as conn:
+        ephys_qc_v1.upgrade(conn)
+        ephys_qc_v1.upgrade(conn)
+    assert qc.get_run(run["run_id"])["review"] == changed["review"]
+    pending = qc.create_run(record["recording_id"], {})
+    with get_db() as conn:
+        conn.execute("UPDATE jobs SET status='cancelled',result=? WHERE id=?",
+                     (json.dumps({"error": "cancelled by reviewer"}), pending["job_id"]))
+    assert qc.get_run(pending["run_id"])["status"] == "cancelled"
+    assert "reviewer" in qc.get_run(pending["run_id"])["error"]
+
+
+def test_existing_decision_keeps_original_reviewer_when_saving_other_changes(scanned):
+    _, run = scanned
+    first = save(run, [decision(run)])
+    second = qc.save_review(run["run_id"], {"base_revision": 1, "reviewer": "second-reviewer",
+                                           "decisions": first["review"]["decisions"] + [decision(run, channel=1)]})
+    assert second["review"]["decisions"][0]["reviewer"] == "reviewer-one"
+    assert second["review"]["decisions"][1]["reviewer"] == "second-reviewer"
+
+
+def test_actual_job_dispatch_scans_and_logs(scanned):
+    from backend.app.domain.jobs import get_job
+    from backend.app.jobs.tasks import run_job
+    record, _ = scanned
+    pending = qc.create_run(record["recording_id"], {})
+    success, log, result = run_job(get_job(pending["job_id"]))
+    assert success and log and result["run_id"] == pending["run_id"]
+    assert qc.get_run(pending["run_id"])["status"] == "succeeded"
+
+
+def test_partial_scan_cannot_be_presented_as_full_automatic_assessment(scanned):
+    record, original = scanned
+    stream = original["report"]["catalog"]["streams"][0]
+    pending = qc.create_run(record["recording_id"], {"config": {"ranges": [
+        {"stream_id": stream["stream_id"], "segment_id": stream["segments"][0]["segment_id"],
+         "start_sample": 0, "stop_sample": 256}]}})
+    qc.execute_scan(pending["run_id"])
+    result = qc.freeze_run(pending["run_id"], {"base_revision": 0, "reviewer": "reviewer-one"})
+    assert result["report"]["coverage"]["scanned_fraction"] == pytest.approx(1/3)
+    assert result["review"]["summary"]["unknown_ratio"] == 1
+    assert any("自动特征" in reason for reason in result["assessment"]["limitations"])
+
+
+def test_manual_retention_does_not_dismiss_risk_and_partial_evidence_review_is_weighted(scanned):
+    _, run = scanned
+    flagged = next(w for w in run["report"]["windows"] if "flat" in w["flags"])
+    half = (flagged["start_sample"] + flagged["stop_sample"]) // 2
+    item = {key: flagged[key] for key in ("channel_id", "stream_id", "segment_id", "start_sample", "stop_sample")}
+    item.update(action="keep", reason="Manually retained", evidence_window_ids=[flagged["window_id"]])
+    manual = save(run, [item])
+    assert manual["review"]["summary"]["reviewed_score"] == run["report"]["initial_score"]["total"]
+    assert manual["review"]["summary"]["retained_percent"] == 100
+    assert manual["review"]["summary"]["evidence_reviewed_ratio"] == 0
+    item.update(response="rejected", stop_sample=half)
+    partial = save(manual, [item])
+    summary = partial["review"]["summary"]
+    assert run["report"]["initial_score"]["total"] < summary["reviewed_score"] < 100
+    assert summary["technical_score"]["reviewed_channel_seconds"] == pytest.approx(flagged["duration_seconds"] / 2)
+    assert summary["unknown_ratio"] > 0
+    item["stop_sample"] = flagged["stop_sample"]
+    complete = save(partial, [item])
+    assert complete["review"]["summary"]["reviewed_score"] == 100
+
+
+def test_evidence_from_another_segment_cannot_change_score(scanned):
+    _, run = scanned
+    stream = run["report"]["catalog"]["streams"][0]
+    other = next(w for w in run["report"]["windows"] if w["segment_id"] == stream["segments"][1]["segment_id"])
+    item = decision(run, response="confirmed", evidence_window_ids=[other["window_id"]])
+    with pytest.raises(ValueError, match="Evidence"):
+        save(run, [item])
+
+
+def test_cross_window_event_review_scores_only_its_exact_samples(scanned):
+    record, original = scanned
+    pending = qc.create_run(record["recording_id"], {"config": {"window_seconds": 1}})
+    qc.execute_scan(pending["run_id"])
+    run = qc.get_run(pending["run_id"])
+    channel = run["report"]["catalog"]["streams"][0]["channels"][0]["channel_id"]
+    windows = [w for w in run["report"]["windows"] if w["channel_id"] == channel and w["start_sample"] < 512]
+    assert len(windows) == 2
+    item = decision(run, start=128, stop=384, response="confirmed",
+                    evidence_window_ids=[w["window_id"] for w in windows])
+    updated = save(run, [item])
+    summary = updated["review"]["summary"]
+    assert summary["channels"][0]["excluded_seconds"] == 1
+    assert summary["technical_score"]["reviewed_channel_seconds"] == 1
+    assert summary["reviewed_score"] < run["report"]["initial_score"]["total"]
+    saved = qc.get_run(run["run_id"])["review"]["decisions"][0]
+    assert (saved["start_sample"], saved["stop_sample"]) == (128, 384)
+    assert saved["evidence_window_ids"] == item["evidence_window_ids"]
+
+
+def test_fully_excluded_recording_is_not_evaluated_usable(scanned):
+    _, run = scanned
+    decisions = [{"channel_id": c["channel_id"], "scope": "channel", "action": "exclude", "reason": "No retained signal"}
+                 for c in run["report"]["catalog"]["streams"][0]["channels"]]
+    saved = save(run, decisions)
+    frozen = qc.freeze_run(run["run_id"], {"base_revision": saved["review"]["revision"], "reviewer": "reviewer-one"})
+    assert frozen["assessment"]["availability"] == "unusable"
+    assert frozen["review"]["summary"]["kept_seconds"] == 0
diff --git a/backend/tests/contract/test_ephys_recordings.py b/backend/tests/contract/test_ephys_recordings.py
new file mode 100644
index 0000000000000000000000000000000000000000..e600575ccd8f749c1bb615fe9bd127ea585e3f14
--- /dev/null
+++ b/backend/tests/contract/test_ephys_recordings.py
@@ -0,0 +1,537 @@
+"""S1 storage/API contracts, always against throwaway storage.
+
+These tests validate bytes and identity, not EDF decoding or signal quality.
+Run pytest with process-level BRAINVIZ storage/DB settings in a fresh directory:
+the parent conftest session fixture runs before function-level isolation.
+"""
+import io
+import json
+import sqlite3
+import stat
+import zipfile
+from concurrent.futures import ThreadPoolExecutor
+
+import pytest
+from flask import Flask
+from werkzeug.datastructures import FileStorage
+
+from backend.app.config import settings
+from backend.app.db import get_db, init_db
+from backend.app.domain import ephys_recordings as recordings
+from backend.app.domain.assets import AssetError, register_asset, reset_asset_group
+from backend.app.domain.eeg import list_eeg_files, record_eeg_file
+from backend.app.domain.ephys_storage import ImportConflict
+from backend.app.domain.subjects import create_subject
+
+
+@pytest.fixture(autouse=True)
+def isolated(isolated_storage):
+    create_subject("S01", "S1 fixture", None)
+    return isolated_storage
+
+
+def upload(data=b"first recording bytes", name="same.edf"):
+    return FileStorage(stream=io.BytesIO(data), filename=name)
+
+
+def payload(**overrides):
+    return {"subject_id": "S01", "label": "same", "modality": "eeg", **overrides}
+
+
+def imported(data=b"first recording bytes", **overrides):
+    return recordings.import_recording(payload(**overrides), uploads=[upload(data)])[0]
+
+
+def recording_root(recording):
+    return settings.dataset_root / recording["input_artifact"]["storage_path"]
+
+
+def snapshot_files(recording):
+    root = recording_root(recording) / "files"
+    return {entry["relative_path"]: (root / entry["relative_path"]).read_bytes()
+            for entry in recording["manifest"]["files"]}
+
+
+def zip_upload(members):
+    stream = io.BytesIO()
+    with zipfile.ZipFile(stream, "w") as archive:
+        for name, contents in members:
+            archive.writestr(name, contents)
+    stream.seek(0)
+    return FileStorage(stream=stream, filename="recording.zip")
+
+
+def assert_no_recording_or_partial_files():
+    assert recordings.list_recordings("S01") == []
+    destination = settings.resourses_root / "S01" / "ephys-recordings"
+    assert not destination.exists() or list(destination.iterdir()) == []
+
+
+def test_same_subject_label_and_bytes_get_distinct_ids_without_import_key():
+    first = imported()
+    second = imported()
+    seeg = imported(modality="seeg")
+    assert len({item["recording_id"] for item in (first, second, seeg)}) == 3
+    assert len({item["input_artifact_id"] for item in (first, second, seeg)}) == 3
+    assert len({recording_root(item) for item in (first, second, seeg)}) == 3
+    assert first["input_artifact"]["content_sha256"] == second["input_artifact"]["content_sha256"]
+    for item in (first, second, seeg):
+        assert item["status"] == "stored_unreviewed"
+        assert item["scientific_validity"] == "pending"
+        assert list(snapshot_files(item).values()) == [b"first recording bytes"]
+        assert recordings.get_recording(item["recording_id"])["recording_id"] == item["recording_id"]
+    assert len(recordings.list_recordings("S01")) == 3
+
+
+def test_import_key_is_idempotent_but_conflicts_on_input_or_description():
+    description = payload(import_key="upload-001", session_id="session-A",
+                          metadata={"metadata_sources": ["operator"]})
+    first, reused = recordings.import_recording(description, uploads=[upload()])
+    assert reused is False
+    again, reused = recordings.import_recording(description, uploads=[upload()])
+    assert reused is True
+    assert again["recording_id"] == first["recording_id"]
+    assert again["input_artifact_id"] == first["input_artifact_id"]
+    with pytest.raises(ImportConflict):
+        recordings.import_recording(description, uploads=[upload(b"changed bytes")])
+    for change in ({"label": "renamed"}, {"modality": "seeg"}, {"session_id": "session-B"},
+                   {"metadata": {"metadata_sources": ["changed"]}}):
+        with pytest.raises(ImportConflict):
+            recordings.import_recording({**description, **change}, uploads=[upload()])
+    assert len(recordings.list_recordings("S01")) == 1
+    assert snapshot_files(first) == snapshot_files(recordings.get_recording(first["recording_id"]))
+
+
+def test_concurrent_same_key_requests_publish_one_verified_input():
+    def request(_):
+        return recordings.import_recording(payload(import_key="concurrent-key"), uploads=[upload()])
+    with ThreadPoolExecutor(max_workers=2) as executor:
+        results = list(executor.map(request, range(2)))
+    assert sorted(reused for _, reused in results) == [False, True]
+    assert len({item["recording_id"] for item, _ in results}) == 1
+    assert len(recordings.list_recordings("S01")) == 1
+    assert recordings.verify_recording(results[0][0]["recording_id"])["valid"] is True
+    assert len(list((settings.resourses_root / "S01" / "ephys-recordings").iterdir())) == 1
+
+
+def test_idempotent_retry_rejects_a_tampered_existing_snapshot():
+    snapshot = imported(import_key="integrity-key")
+    first_member = snapshot["manifest"]["files"][0]["relative_path"]
+    path = recording_root(snapshot) / "files" / first_member
+    path.chmod(0o644)
+    path.write_bytes(b"external corruption")
+    with pytest.raises(ImportConflict, match="fingerprint"):
+        imported(import_key="integrity-key")
+    assert len(recordings.list_recordings("S01")) == 1
+    assert path.read_bytes() == b"external corruption"  # no silent repair or replacement
+
+
+def test_legacy_upload_overwrite_and_reset_cannot_change_copied_input():
+    asset = register_asset("S01", "eeg", upload(b"old acquisition"))
+    old_source = settings.dataset_root / asset["storage_path"]
+    snapshot, _ = recordings.import_recording(payload(asset_ids=[asset["id"]]))
+    assert set(snapshot_files(snapshot).values()) == {b"old acquisition"}
+    register_asset("S01", "eeg", upload(b"new acquisition"))
+    assert old_source.read_bytes() == b"new acquisition"
+    assert set(snapshot_files(snapshot).values()) == {b"old acquisition"}
+    reset_asset_group("S01", "eeg")
+    assert not old_source.exists()
+    assert set(snapshot_files(snapshot).values()) == {b"old acquisition"}
+    assert recordings.verify_recording(snapshot["recording_id"])["valid"] is True
+
+
+def test_old_asset_upload_and_reset_cannot_target_reserved_recording_namespace():
+    snapshot = imported()
+    # A legacy/corrupt row with the reserved type must not be deleted before validation.
+    legacy = register_asset("S01", "eeg", upload(b"legacy asset bytes", "legacy.edf"))
+    with get_db() as connection:
+        connection.execute("UPDATE subject_assets SET asset_type=? WHERE id=?",
+                           ("ephys-recordings", legacy["id"]))
+        before = connection.execute("SELECT * FROM subject_assets WHERE id=?", (legacy["id"],)).fetchone()
+    with pytest.raises(AssetError, match="reserved"):
+        register_asset("S01", "ephys-recordings", upload(b"overwrite attempt"))
+    with pytest.raises(AssetError, match="reserved"):
+        reset_asset_group("S01", "ephys-recordings")
+    with get_db() as connection:
+        after = connection.execute("SELECT * FROM subject_assets WHERE id=?", (legacy["id"],)).fetchone()
+    assert after == before
+    assert (settings.dataset_root / legacy["storage_path"]).read_bytes() == b"legacy asset bytes"
+    assert set(snapshot_files(snapshot).values()) == {b"first recording bytes"}
+    assert recordings.verify_recording(snapshot["recording_id"])["valid"] is True
+
+
+def test_asset_replaced_between_row_lookup_and_copy_cannot_keep_old_provenance(monkeypatch):
+    original = register_asset("S01", "eeg", upload(b"original bytes"))
+    copy_path = recordings.SnapshotWriter.add_path
+    replacements = []
+
+    def replace_then_copy(writer, source, subject_root):
+        reset_asset_group("S01", "eeg")
+        replacement = register_asset("S01", "eeg", upload(b"replacement bytes"))
+        replacements.append(replacement)
+        assert replacement["storage_path"] == original["storage_path"]
+        return copy_path(writer, source, subject_root)
+
+    monkeypatch.setattr(recordings.SnapshotWriter, "add_path", replace_then_copy)
+    with pytest.raises(ImportConflict, match="entry changed"):
+        recordings.import_recording(payload(asset_ids=[original["id"]]))
+    assert_no_recording_or_partial_files()
+    assert len(replacements) == 1
+    assert replacements[0]["id"] != original["id"]
+    assert (settings.dataset_root / replacements[0]["storage_path"]).read_bytes() == b"replacement bytes"
+
+
+def test_legacy_row_updated_between_lookup_and_copy_cannot_keep_stale_provenance(monkeypatch):
+    asset = register_asset("S01", "eeg", upload(b"unchanged raw bytes"))
+    processed = settings.output_root / "S01" / "legacy-playback.bin"
+    processed.parent.mkdir(parents=True, exist_ok=True)
+    processed.write_bytes(b"legacy playback")
+    record_eeg_file("S01", "same", asset["storage_path"], str(processed),
+                    {"channels": ["C3"], "metadata": {"duration": 1}}, asset_id=asset["id"], mode="EEG")
+    legacy_id = list_eeg_files("S01")[0]["id"]
+    copy_path = recordings.SnapshotWriter.add_path
+
+    def update_then_copy(writer, source, subject_root):
+        record_eeg_file("S01", "same", asset["storage_path"], str(processed),
+                        {"channels": ["C4"], "metadata": {"duration": 2}}, asset_id=asset["id"], mode="EEG")
+        return copy_path(writer, source, subject_root)
+
+    monkeypatch.setattr(recordings.SnapshotWriter, "add_path", update_then_copy)
+    with pytest.raises(ImportConflict, match="entry changed"):
+        recordings.import_recording(payload(legacy_eeg_id=legacy_id))
+    assert_no_recording_or_partial_files()
+    assert list_eeg_files("S01")[0]["metadata"]["channels"] == ["C4"]
+    assert processed.read_bytes() == b"legacy playback"
+
+
+@pytest.mark.parametrize("change", ["bytes", "add_member", "delete_member"])
+def test_later_source_copy_rechecks_earlier_source_bytes_and_file_set(monkeypatch, change):
+    first = register_asset("S01", "eeg", zip_upload([("first.edf", b"first")]))
+    second = register_asset("S01", "seeg", upload(b"second", "second.edf"))
+    first_root = settings.dataset_root / first["storage_path"]
+    second_path = settings.dataset_root / second["storage_path"]
+    copy_path = recordings.SnapshotWriter.add_path
+    mutations = []
+
+    def mutate_earlier_then_copy(writer, source, subject_root):
+        if source == second_path:
+            mutations.append(change)
+            if change == "bytes":
+                (first_root / "first.edf").write_bytes(b"FIRST")
+            elif change == "add_member":
+                (first_root / "new-events.tsv").write_bytes(b"new events")
+            else:
+                (first_root / "first.edf").unlink()
+        return copy_path(writer, source, subject_root)
+
+    monkeypatch.setattr(recordings.SnapshotWriter, "add_path", mutate_earlier_then_copy)
+    with pytest.raises(ImportConflict, match="Source .* changed"):
+        recordings.import_recording(payload(asset_ids=[first["id"], second["id"]]))
+    assert mutations == [change]
+    assert_no_recording_or_partial_files()
+    assert second_path.read_bytes() == b"second"
+
+
+def test_linking_legacy_eeg_keeps_existing_playback_record_and_processed_bytes():
+    asset = register_asset("S01", "eeg", upload(b"legacy raw"))
+    processed = settings.output_root / "S01" / "electrophysiology" / "eeg" / "playback.bin"
+    processed.parent.mkdir(parents=True, exist_ok=True)
+    processed.write_bytes(b"existing playback bytes")
+    record_eeg_file("S01", "same", asset["storage_path"], str(processed),
+                    {"channels": ["C3"], "metadata": {"duration": 1, "window_sec": 1}},
+                    asset_id=asset["id"], mode="EEG")
+    before = list_eeg_files("S01")
+    snapshot, _ = recordings.import_recording(payload(legacy_eeg_id=before[0]["id"]))
+    assert list_eeg_files("S01") == before
+    assert processed.read_bytes() == b"existing playback bytes"
+    assert set(snapshot_files(snapshot).values()) == {b"legacy raw"}
+    init_db()
+    init_db()
+    assert list_eeg_files("S01") == before
+    assert recordings.get_recording(snapshot["recording_id"])["input_artifact_id"] == snapshot["input_artifact_id"]
+    assert recordings.verify_recording(snapshot["recording_id"])["valid"] is True
+
+
+def test_zip_manifest_hashes_each_member_and_changed_member_sets_have_new_content_hash():
+    def ingest(members):
+        return recordings.import_recording(payload(), uploads=[zip_upload(members)])[0]
+    baseline = ingest([("session/data.bin", b"signal"), ("session/events.tsv", b"events")])
+    added = ingest([("session/data.bin", b"signal"), ("session/events.tsv", b"events"), ("sidecar.json", b"{}")])
+    deleted = ingest([("session/data.bin", b"signal")])
+    changed = ingest([("session/data.bin", b"SIGNAL"), ("session/events.tsv", b"events")])
+    assert len({item["input_artifact"]["content_sha256"] for item in (baseline, added, deleted, changed)}) == 4
+    assert baseline["input_artifact"]["file_count"] == 2
+    assert baseline["input_artifact"]["size_bytes"] == len(b"signalevents")
+    assert set(snapshot_files(baseline).values()) == {b"signal", b"events"}
+    assert all(len(member["sha256"]) == 64 for member in baseline["manifest"]["files"])
+
+
+@pytest.mark.parametrize("mutation", ["add", "delete", "change"])
+def test_verify_detects_member_set_or_content_tampering(mutation):
+    snapshot, _ = recordings.import_recording(payload(), uploads=[zip_upload([("a.bin", b"alpha"), ("b.bin", b"bravo")])])
+    directory = recording_root(snapshot) / "files"
+    member = directory / snapshot["manifest"]["files"][0]["relative_path"]
+    if mutation == "add":
+        (directory / "unlisted.bin").write_bytes(b"extra")
+    elif mutation == "delete":
+        member.unlink()
+    else:
+        member.chmod(0o644)  # simulate an external owner bypassing read-only flags
+        member.write_bytes(b"ALPHA")  # same length: byte fingerprint, not size alone
+    result = recordings.verify_recording(snapshot["recording_id"])
+    assert result["valid"] is False
+    assert result["issues"]
+
+
+def test_get_never_trusts_a_changed_manifest():
+    snapshot = imported()
+    path = recording_root(snapshot) / "manifest.json"
+    original = json.loads(path.read_text())
+    path.chmod(0o644)
+    path.write_text(json.dumps({**original, "label": "tampered"}))
+    with pytest.raises(ValueError):
+        recordings.get_recording(snapshot["recording_id"])
+    result = recordings.verify_recording(snapshot["recording_id"])
+    assert result["valid"] is False and result["issues"]
+
+
+@pytest.mark.parametrize("name", ["../escape.bin", "/escape.bin", "nested/../../escape.bin", "..\\escape.bin"])
+def test_upload_and_zip_path_traversal_are_rejected(name):
+    with pytest.raises(ValueError):
+        recordings.import_recording(payload(), uploads=[upload(name=name)])
+    assert_no_recording_or_partial_files()
+    with pytest.raises(ValueError):
+        recordings.import_recording(payload(), uploads=[zip_upload([(name, b"escape")])])
+    assert_no_recording_or_partial_files()
+
+
+def test_archive_symlink_and_asset_symlink_are_rejected(tmp_path):
+    link = zipfile.ZipInfo("linked.edf")
+    link.create_system = 3
+    link.external_attr = (stat.S_IFLNK | 0o777) << 16
+    stream = io.BytesIO()
+    with zipfile.ZipFile(stream, "w") as archive:
+        archive.writestr(link, "../../outside.edf")
+    stream.seek(0)
+    with pytest.raises(ValueError):
+        recordings.import_recording(payload(), uploads=[FileStorage(stream=stream, filename="links.zip")])
+    assert_no_recording_or_partial_files()
+    asset = register_asset("S01", "eeg", upload())
+    source = settings.dataset_root / asset["storage_path"]
+    source.unlink()
+    outside = tmp_path / "outside.edf"
+    outside.write_bytes(b"must not follow")
+    source.symlink_to(outside)
+    with pytest.raises(ValueError):
+        recordings.import_recording(payload(asset_ids=[asset["id"]]))
+    assert_no_recording_or_partial_files()
+
+
+def test_failed_stream_copy_rolls_back_database_and_partial_input():
+    class FailedRead(io.BytesIO):
+        def read(self, size=-1):
+            if self.tell() > 0:
+                raise OSError("synthetic input read failure")
+            return super().read(size)
+    broken = FileStorage(stream=FailedRead(b"partial bytes"), filename="broken.edf")
+    with pytest.raises(OSError, match="synthetic input read failure"):
+        recordings.import_recording(payload(), uploads=[upload(), broken])
+    assert_no_recording_or_partial_files()
+
+
+def test_invalid_archive_after_valid_input_leaves_no_partial_recording():
+    with pytest.raises((ValueError, zipfile.BadZipFile)):
+        recordings.import_recording(payload(), uploads=[upload(), upload(b"not a zip", "broken.zip")])
+    assert_no_recording_or_partial_files()
+
+
+def test_database_failure_after_publication_rolls_back_rows_and_snapshot():
+    with get_db() as connection:
+        connection.execute("""CREATE TRIGGER fail_s1_artifact_insert BEFORE INSERT ON ephys_artifacts
+                            BEGIN SELECT RAISE(ABORT, 'synthetic artifact insert failure'); END""")
+    with pytest.raises(sqlite3.IntegrityError, match="synthetic artifact insert failure"):
+        imported()
+    assert_no_recording_or_partial_files()
+    with get_db() as connection:
+        assert connection.execute("SELECT count(*) AS n FROM ephys_artifacts").fetchone()["n"] == 0
+
+
+def test_duplicate_raw_names_are_preserved_but_duplicate_channel_ids_rejected():
+    metadata = {"channels": [{"channel_id": "ch-1", "raw_name": "A1"},
+                             {"channel_id": "ch-2", "raw_name": "A1"}],
+                "metadata_sources": ["operator"], "streams": [], "segments": []}
+    snapshot = imported(metadata=metadata)
+    assert snapshot["metadata"] == metadata
+    assert snapshot["manifest"]["metadata"] == metadata
+    assert all("type" not in item for item in snapshot["metadata"]["channels"])
+    bad = {**metadata, "channels": [{"channel_id": "ch-1", "raw_name": "A1"},
+                                    {"channel_id": "ch-1", "raw_name": "A2"}]}
+    with pytest.raises(ValueError):
+        imported(metadata=bad)
+    assert len(recordings.list_recordings("S01")) == 1
+
+
+def test_foreign_subject_asset_is_not_importable():
+    create_subject("S02", "other subject", None)
+    asset = register_asset("S02", "eeg", upload())
+    with pytest.raises((ValueError, LookupError)):
+        recordings.import_recording(payload(asset_ids=[asset["id"]]))
+    assert_no_recording_or_partial_files()
+
+
+def test_configured_storage_alias_is_supported_but_child_symlinks_are_not(tmp_path, monkeypatch):
+    # macOS /var -> /private/var is a configured-root alias, not an escaping input.
+    original_dataset = settings.dataset_root
+    alias = tmp_path / "dataset-alias"
+    alias.symlink_to(original_dataset, target_is_directory=True)
+    monkeypatch.setattr(settings, "dataset_root", alias)
+    monkeypatch.setattr(settings, "resourses_root", alias / "resourses")
+    snapshot = imported()
+    assert recordings.verify_recording(snapshot["recording_id"])["valid"]
+    asset = register_asset("S01", "eeg", upload(b"aliased source"))
+    copied, _ = recordings.import_recording(payload(asset_ids=[asset["id"]]))
+    assert set(snapshot_files(copied).values()) == {b"aliased source"}
+    source = settings.dataset_root / asset["storage_path"]
+    source.unlink()
+    source.symlink_to(recording_root(snapshot) / "files" / "same.edf")
+    with pytest.raises(ValueError, match="Symlink"):
+        recordings.import_recording(payload(asset_ids=[asset["id"]]))
+
+
+@pytest.fixture
+def qc_client():
+    from backend.app.api.v1.ephys import ephys_bp
+    application = Flask(__name__)
+    application.config["TESTING"] = True
+    application.register_blueprint(ephys_bp, url_prefix="/api/v1/ephys")
+    return application.test_client()
+
+
+def test_api_import_list_get_verify_and_conflict(qc_client):
+    asset = register_asset("S01", "eeg", upload())
+    request = payload(asset_ids=[asset["id"]], import_key="api-001")
+    response = qc_client.post("/api/v1/ephys/recordings", json=request)
+    assert response.status_code in (200, 201), response.get_json()
+    recording = response.get_json()["data"]
+    recording_id = recording["recording_id"]
+    assert recording["scientific_validity"] == "pending"
+    assert qc_client.get(f"/api/v1/ephys/recordings/{recording_id}").get_json()["data"]["recording_id"] == recording_id
+    listed = qc_client.get("/api/v1/ephys/recordings?subject_id=S01")
+    assert listed.status_code == 200
+    assert [item["recording_id"] for item in listed.get_json()["data"]] == [recording_id]
+    verified = qc_client.post(f"/api/v1/ephys/recordings/{recording_id}/verify")
+    assert verified.status_code == 200
+    assert verified.get_json()["data"]["valid"] is True
+    repeated = qc_client.post("/api/v1/ephys/recordings", json=request)
+    assert repeated.status_code in (200, 201)
+    assert repeated.get_json()["meta"]["reused"] is True
+    conflict = qc_client.post("/api/v1/ephys/recordings", json={**request, "label": "changed"})
+    assert conflict.status_code == 409
+    assert "error" in conflict.get_json()
+    missing = qc_client.get("/api/v1/ephys/recordings/not-a-recording")
+    assert missing.status_code == 404
+    assert "error" in missing.get_json()
+
+
+def test_api_multipart_upload_preserves_raw_bytes(qc_client):
+    response = qc_client.post("/api/v1/ephys/recordings", data={
+        "metadata": json.dumps(payload(modality="mixed", metadata={"metadata_sources": ["declared"]})),
+        "files": (io.BytesIO(b"multipart acquisition"), "raw.edf"),
+    }, content_type="multipart/form-data")
+    assert response.status_code == 201, response.get_json()
+    recording = response.get_json()["data"]
+    assert recording["modality"] == "mixed"
+    assert snapshot_files(recording) == {"raw.edf": b"multipart acquisition"}
+
+
+def legacy_session_upload(client, asset_type, data, transfer):
+    """Exercise the existing HTTP session chain, with scientific jobs disabled."""
+    created = client.post("/api/subjects/S01/upload-sessions", json={
+        "asset_type": asset_type, "auto_run": False,
+        "files": [{"name": "same.edf", "size": len(data)}],
+    })
+    assert created.status_code == 200, created.get_json()
+    session_id = created.get_json()["session_id"]
+    base = f"/api/subjects/S01/upload-sessions/{session_id}"
+    if transfer == "files":
+        received = client.post(base + "/files", data={
+            "relative_path": "same.edf", "file": (io.BytesIO(data), "same.edf"),
+        }, content_type="multipart/form-data")
+        assert received.status_code == 200, received.get_json()
+        assert received.get_json()["files_done"] == 1
+    else:
+        split = len(data) // 2
+        for offset, block in ((0, data[:split]), (split, data[split:])):
+            received = client.post(base + "/chunks", data={
+                "filename": "same.edf", "offset": str(offset), "total_size": str(len(data)),
+                "chunk": (io.BytesIO(block), "part"),
+            }, content_type="multipart/form-data")
+            assert received.status_code == 200, received.get_json()
+    finalized = client.post(base + "/finalize")
+    assert finalized.status_code == 201, finalized.get_json()
+    body = finalized.get_json()
+    assert body["auto_jobs"] == []
+    assert len(body["saved"]) == 1
+    return body["saved"][0]
+
+
+def write_legacy_playback_fixture():
+    """Known JSON output for route compatibility, not output of an actual pipeline."""
+    directory = settings.output_root / "S01" / "electrophysiology" / "seeg" / "same_fixture"
+    directory.mkdir(parents=True, exist_ok=True)
+    manifest = {
+        "metadata": {"window_sec": 0.5, "n_chunks": 1, "n_total_frames": 3, "synthetic": True},
+        "monopolar": {"channels": [
+            {"name": "A1", "electrode_name": "A", "contact_index": 1},
+            {"name": "A2", "electrode_name": "A", "contact_index": 2},
+        ]},
+        "bipolar": {"channels": [{"name": "A1-A2", "electrode_name": "A", "contact_index": 1}]},
+        "chunks": [{"index": 0, "start_frame": 0, "end_frame": 3, "n_frames": 3,
+                    "start_time": 0, "end_time": 1.5}],
+    }
+    chunk = {
+        "chunk_index": 0, "start_frame": 0, "end_frame": 3, "n_frames": 3,
+        "start_time": 0, "end_time": 1.5,
+        "monopolar_data": [[1, 2, 3], [4, 5, 6]], "bipolar_data": [[-3, -3, -3]],
+    }
+    (directory / "manifest.json").write_text(json.dumps(manifest))
+    (directory / "chunk_000.json").write_text(json.dumps(chunk))
+    return chunk
+
+
+@pytest.mark.parametrize(("modality", "transfer"), [("eeg", "files"), ("seeg", "chunks")])
+def test_real_app_upload_qc_snapshot_replacement_and_legacy_playback_chain(client, modality, transfer):
+    # Uses create_app's registered routes; no lightweight blueprint substitution.
+    original_bytes = b"original acquisition fixture"
+    replacement_bytes = b"replacement acquisition fixture"
+    original = legacy_session_upload(client, modality, original_bytes, transfer)
+    expected_chunk = write_legacy_playback_fixture()
+    urls = ["/api/seeg/S01/manifest/same", "/api/seeg/S01/chunk/same/0",
+            "/api/seeg/S01/playback/same?mode=monopolar", "/api/seeg/S01/playback/same?mode=bipolar"]
+
+    def read_playback():
+        responses = [client.get(url) for url in urls]
+        assert all(response.status_code == 200 for response in responses)
+        assert responses[1].get_json() == expected_chunk
+        assert responses[2].get_json()["first_chunk"] == expected_chunk
+        assert responses[2].get_json()["contact_map"] == {"A:1": 0, "A:2": 1}
+        assert responses[3].get_json()["contact_map"] == {"A:1": 0}
+        return [response.data for response in responses]
+
+    before = read_playback()
+    imported_response = client.post("/api/v1/ephys/recordings", json=payload(
+        modality=modality, asset_ids=[original["id"]], import_key=f"route-chain-{modality}",
+    ))
+    assert imported_response.status_code == 201, imported_response.get_json()
+    snapshot = imported_response.get_json()["data"]
+    assert snapshot_files(snapshot) == {"same.edf": original_bytes}
+    assert read_playback() == before
+    replacement = legacy_session_upload(client, modality, replacement_bytes, transfer)
+    assert replacement["id"] != original["id"]
+    assert replacement["storage_path"] == original["storage_path"]
+    assert (settings.dataset_root / replacement["storage_path"]).read_bytes() == replacement_bytes
+    assert snapshot_files(snapshot) == {"same.edf": original_bytes}
+    verification = client.post(f"/api/v1/ephys/recordings/{snapshot['recording_id']}/verify")
+    assert verification.status_code == 200
+    assert verification.get_json()["data"]["valid"] is True
+    assert read_playback() == before
diff --git a/backend/tests/contract/test_ephys_samples_api.py b/backend/tests/contract/test_ephys_samples_api.py
new file mode 100644
index 0000000000000000000000000000000000000000..2e5cb30b85a78c36e51a81fff639802b37d69f8d
--- /dev/null
+++ b/backend/tests/contract/test_ephys_samples_api.py
@@ -0,0 +1,166 @@
+"""S2 identity-bound binary windows and legacy isolation, using byte-authored data."""
+import io
+import json
+import os
+import struct
+
+import numpy as np
+import pytest
+from flask import Flask
+from werkzeug.datastructures import FileStorage
+
+from backend.app.api.v1.ephys import ephys_bp
+from backend.app.api.v1.ephys_samples import ephys_samples_bp
+from backend.app.config import settings
+from backend.app.domain import ephys_signals
+from backend.app.domain.ephys_recordings import import_recording
+from backend.app.domain.subjects import create_subject
+from backend.tests.ephys_edf_fixtures import tal, write_edf_bdf
+
+
+@pytest.fixture
+def example(isolated_storage, tmp_path):
+    create_subject("S2", "synthetic", None)
+    source = write_edf_bdf(tmp_path / "fixture.edf", [
+        {"label": "same", "records": [[-2, -1, 0, 1], [2, 3, 4, 5], [6, 7, 8, 9]]},
+        {"label": "same", "records": [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], "unit": "mV"},
+        {"label": "TRIG", "records": [[0, 1], [0, 2], [0, 3]], "unit": ""}],
+        reserved="EDF+D", annotation_records=[tal(.25) + tal(.5, "pulse"), tal(1.25), tal(5.25)])
+    record, _ = import_recording({"subject_id": "S2", "label": "fixture", "modality": "mixed"},
+                                [FileStorage(stream=io.BytesIO(source.read_bytes()), filename="fixture.edf")])
+    app = Flask(__name__)
+    app.config["TESTING"] = True
+    app.register_blueprint(ephys_bp)
+    app.register_blueprint(ephys_samples_bp)
+    client = app.test_client()
+    aid = record["input_artifact_id"]
+    catalog = client.get(f"/api/v1/ephys/artifacts/{aid}/signals").get_json()["data"]
+    stream = catalog["streams"][0]
+    params = [("stream_id", stream["stream_id"]), ("segment_id", stream["segments"][0]["segment_id"]),
+              ("channel_id", stream["channels"][1]["channel_id"]),
+              ("channel_id", stream["channels"][0]["channel_id"]),
+              ("start_sample", 3), ("stop_sample", 6), ("representation", "physical")]
+    return client, record, catalog, params
+
+
+def decode(response):
+    assert response.status_code == 200, response.get_json()
+    assert response.mimetype == "application/octet-stream"
+    raw = response.data
+    assert raw[:4] == b"EPHY"
+    length = struct.unpack("<I", raw[4:8])[0]
+    meta = json.loads(raw[8:8 + length])
+    offset = (8 + length + 7) // 8 * 8
+    assert not any(raw[8 + length:offset])
+    array = np.frombuffer(raw[offset:], dtype="<f8").reshape(meta["shape"])
+    return meta, array
+
+
+def endpoint(record, name="samples"):
+    return f"/api/v1/ephys/artifacts/{record['input_artifact_id']}/{name}"
+
+
+def test_catalog_and_binary_preserve_order_calibration_and_source_time(example):
+    client, record, catalog, params = example
+    assert len(catalog["streams"]) == 2
+    assert catalog["scientific_validity"] == "pending"
+    assert catalog["processing"] == []
+    assert catalog["streams"][0]["channels"][0]["raw_name"] == "same"
+    meta, data = decode(client.get(endpoint(record), query_string=params))
+    assert meta["artifact_id"] == record["input_artifact_id"]
+    assert meta["recording_id"] == record["recording_id"]
+    assert meta["content_sha256"] == record["input_artifact"]["content_sha256"]
+    assert meta["channel_ids"] == [v for k, v in params if k == "channel_id"]
+    assert meta["shape"] == [2, 3] and meta["units"] == ["V", "V"]
+    assert meta["dtype"] == "<f8" and meta["order"] == "C"
+    assert meta["data_kind"] == "raw_samples" and meta["processing"] == []
+    assert meta["recording_start_seconds"] == 1.0
+    assert meta["clock_alignment"] == "source_local_unaligned"
+    np.testing.assert_allclose(data, [[.009, .008, .007], [1e-6, 2e-6, 3e-6]], atol=1e-15)
+    digital = [(k, "digital" if k == "representation" else v) for k, v in params]
+    dmeta, codes = decode(client.get(endpoint(record), query_string=digital))
+    np.testing.assert_array_equal(codes, [[9, 8, 7], [1, 2, 3]])
+    assert dmeta["units"] == ["count", "count"]
+
+
+def test_overview_is_distinct_minmax_with_short_final_bin(example):
+    client, record, _, params = example
+    query = [(k, 0 if k == "start_sample" else 7 if k == "stop_sample" else
+              "digital" if k == "representation" else v) for k, v in params] + [("bin_samples", 3)]
+    meta, values = decode(client.get(endpoint(record, "overview"), query_string=query))
+    assert meta["data_kind"] == "minmax_overview" and meta["shape"] == [2, 3, 2]
+    assert meta["bin_start_samples"] == [0, 3, 6] and meta["bin_stop_samples"] == [3, 6, 7]
+    np.testing.assert_array_equal(values, [[[10, 12], [7, 9], [6, 6]], [[-2, 0], [1, 3], [4, 4]]])
+
+
+@pytest.mark.parametrize("field,value", [("stop_sample", 9), ("start_sample", -1), ("start_sample", 3.5),
+                                        ("representation", "filtered"), ("channel_id", "unknown"),
+                                        ("stop_sample", 1000000000), ("segment_id", "unknown")])
+def test_invalid_or_gap_crossing_windows_never_yield_binary(example, field, value):
+    client, record, _, params = example
+    query = [(k, value if k == field else v) for k, v in params]
+    response = client.get(endpoint(record), query_string=query)
+    assert response.status_code in (400, 404)
+    assert "error" in response.get_json()
+
+
+def test_second_segment_keeps_gap_and_events_keep_source_clock(example):
+    client, record, catalog, params = example
+    span = catalog["streams"][0]["segments"][1]
+    query = [(k, span["segment_id"] if k == "segment_id" else 8 if k == "start_sample" else
+              9 if k == "stop_sample" else v) for k, v in params]
+    meta, data = decode(client.get(endpoint(record), query_string=query))
+    assert meta["recording_start_seconds"] == 5.25 and data.shape == (2, 1)
+    events = client.get(endpoint(record, "events"), query_string={"offset": 0, "limit": 1}).get_json()["data"]
+    assert events["total"] == 1 and len(events["events"]) == 1
+    event = events["events"][0]
+    assert event["text"] == "pulse" and event["clock_alignment"] == "source_local_unaligned"
+    assert event["sample_mappings"][0]["sample_index"] == 1
+
+
+def test_warm_reader_cache_rechecks_fingerprint_and_rejects_same_size_tampering(example, monkeypatch):
+    client, record, _, params = example
+    checks = []
+    original = ephys_signals.inspect_snapshot
+
+    def observed(*args):
+        checks.append(args)
+        return original(*args)
+
+    monkeypatch.setattr(ephys_signals, "inspect_snapshot", observed)
+    assert client.get(endpoint(record), query_string=params).status_code == 200
+    assert checks == []  # Already SHA256-verified catalog, exact stat guard still matches.
+    path = settings.dataset_root / record["input_artifact"]["storage_path"] / "files" / "fixture.edf"
+    before = path.stat()
+    path.chmod(0o644)
+    with path.open("r+b") as handle:
+        handle.seek(-1, 2)
+        handle.write(b"X")
+    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
+    response = client.get(endpoint(record), query_string=params)
+    assert response.status_code == 409 and checks
+    assert response.get_json()["error"]["code"] == "EPHYS_INPUT_CHANGED"
+
+
+def test_mutation_during_window_is_discarded(example, monkeypatch):
+    client, record, catalog, params = example
+    entry = ephys_signals.verified_catalog(record["input_artifact_id"])
+    reader = entry["readers"][catalog["streams"][0]["stream_id"]]
+    original = reader.read_window
+
+    def mutated(*args, **kwargs):
+        data = original(*args, **kwargs)
+        (entry["root"] / "files" / "unexpected.txt").write_bytes(b"changed input set")
+        return data
+
+    monkeypatch.setattr(reader, "read_window", mutated)
+    assert client.get(endpoint(record), query_string=params).status_code == 409
+
+
+def test_unreadable_source_is_explicit_without_hiding_readable_voltage(example):
+    client, _, _, _ = example
+    record, _ = import_recording({"subject_id": "S2", "label": "unsupported"},
+                                [FileStorage(stream=io.BytesIO(b"not EDF"), filename="bad.edf")])
+    catalog = client.get(endpoint(record, "signals")).get_json()["data"]
+    assert catalog["status"] == "unreadable" and catalog["errors"]
+    assert catalog["streams"] == []
diff --git a/backend/tests/contract/test_seeg_qc_metrics.py b/backend/tests/contract/test_seeg_qc_metrics.py
new file mode 100644
index 0000000000000000000000000000000000000000..0615e07c3bd72cf0426f6ce19481164363cacd84
--- /dev/null
+++ b/backend/tests/contract/test_seeg_qc_metrics.py
@@ -0,0 +1,234 @@
+"""SEEG QC counterexamples: candidate evidence must not erase real activity."""
+import json
+
+import numpy as np
+import pytest
+
+from backend.app.processing.ephys_qc.seeg import scan_seeg
+from backend.app.processing.ephys_qc.seeg_channels import resolve_seeg_channels
+from backend.app.processing.ephys_qc.seeg_scoring import compute_quality_score
+
+
+def fixture(data, fs=500, names=None, segments=None, units=None):
+    data = np.asarray(data, dtype=float)
+    if data.ndim == 1:
+        data = data[None, :]
+    names = names or ["A{}".format(i + 1) for i in range(len(data))]
+    channels = [{"channel_id": "c{}".format(i), "raw_name": name,
+                 "unit": (units or ["V"] * len(data))[i], "channel_type": "unknown",
+                 "quantity": "voltage" if (units or ["V"] * len(data))[i] == "V" else "unknown"}
+                for i, name in enumerate(names)]
+    segments = segments or [{"segment_id": "p", "start_sample": 0,
+                            "stop_sample": data.shape[1], "recording_start_seconds": 0}]
+    catalog = {"streams": [{"stream_id": "s", "source_id": "src", "clock_id": "src",
+                           "sampling_frequency_hz": fs, "channels": channels,
+                           "segments": segments, "sample_count": data.shape[1]}], "sources": []}
+    calls = []
+
+    def read(stream, segment, ids, start, stop, representation="physical"):
+        calls.append((stream, segment, ids, start, stop))
+        return data[[int(c[1:]) for c in ids], start:stop]
+
+    return catalog, read, calls
+
+
+def test_local_flat_window_does_not_discard_whole_channel_or_other_channels():
+    rng = np.random.default_rng(28)
+    x = rng.normal(0, 20e-6, (2, 7500))
+    x[0, 2500:5000] = 0
+    catalog, read, calls = fixture(x)
+    report = scan_seeg(catalog, read)
+    flat = [w for w in report["windows"] if "flat" in w["flags"]]
+    assert [(w["channel_id"], w["start_sample"], w["stop_sample"]) for w in flat] == [("c0", 2500, 5000)]
+    assert report["channels"][0]["flagged_seconds"] == 5
+    assert report["channels"][1]["flagged_seconds"] == 0
+    assert max(stop - start for _, _, _, start, stop in calls) == 2500
+    assert report["initial_score"]["total"] == 83
+
+
+def test_segments_and_partial_tail_keep_real_clock_and_coverage():
+    x = np.random.default_rng(2).normal(0, 1e-5, 3100)
+    segments = [{"segment_id": "p", "start_sample": 0, "stop_sample": 1500, "recording_start_seconds": 0},
+                {"segment_id": "q", "start_sample": 1500, "stop_sample": 3100, "recording_start_seconds": 10}]
+    catalog, read, calls = fixture(x, segments=segments)
+    report = scan_seeg(catalog, read, config={"window_seconds": 2})
+    assert all(not start < 1500 < stop for _, _, _, start, stop in calls)
+    assert [w["duration_seconds"] for w in report["windows"]] == [2, 1, 2, 1.2]
+    assert report["integrity"]["gaps"][0]["duration_seconds"] == 7
+    assert report["coverage"]["scanned_channel_seconds"] == pytest.approx(6.2)
+    partial = scan_seeg(catalog, read, config={"ranges": [{"stream_id": "s", "segment_id": "q",
+                                                        "start_sample": 1500, "stop_sample": 2000}]})
+    assert partial["coverage"]["scanned_fraction"] == pytest.approx(500 / 3100)
+    assert partial["coverage"]["scope"] == "selected_ranges"
+
+
+@pytest.mark.parametrize("frequency", [50, 60])
+def test_line_noise_uses_configured_frequency_and_neighbor_background(frequency):
+    fs = 500
+    t = np.arange(fs * 5) / fs
+    x = 1e-4 * np.sin(2 * np.pi * frequency * t) + np.random.default_rng(9).normal(0, 1e-5, len(t))
+    catalog, read, _ = fixture(x)
+    report = scan_seeg(catalog, read, config={"line_frequency": frequency})
+    assert report["windows"][0]["flags"] == ["line_noise"]
+    assert report["windows"][0]["metrics"]["line_noise_db"] > 12
+    assert report["spectra"][0]["status"] == "computed"
+
+
+def test_high_frequency_sharp_events_and_absent_alpha_are_not_badness():
+    fs = 500
+    t = np.arange(fs * 5) / fs
+    x = 1e-4 * np.sin(2 * np.pi * 83 * t)
+    x[1234:1240] += np.array([0, .001, .003, .001, 0, -.0005])
+    catalog, read, _ = fixture(x)
+    report = scan_seeg(catalog, read)
+    assert report["windows"][0]["flags"] == []
+    assert report["initial_score"]["total"] == 100
+    assert report["windows"][0]["metrics"]["high_frequency_ratio"] > .3
+
+
+def test_identity_parser_preserves_zero_padding_primes_and_existing_bipolar():
+    names = ["A1", "A01", "A'1", "A1-A2", "A3", "A1A3", "ECG", "Fp1"]
+    catalog, _, _ = fixture(np.ones((len(names), 10)), names=names)
+    mapped = resolve_seeg_channels(catalog)
+    assert [c["raw_name"] for c in mapped] == names
+    assert len({c["channel_id"] for c in mapped}) == len(names)
+    assert mapped[2]["candidate"]["shaft"] == "A'"
+    assert mapped[3]["candidate"]["contact_names"] == ["A1", "A2"]
+    assert mapped[3]["measurement_kind"] == "unknown"
+    assert mapped[5]["candidate"]["shaft"] is None
+    assert mapped[7]["channel_type"] == "unknown"
+    assert all(not c.get("bipolar_pairs") for c in mapped)
+    assert "possible_alias" in mapped[0]["issues"]
+    assert "missing_contact_numbers" in mapped[4]["issues"]
+
+
+def test_confirmed_mapping_is_explicit_and_cannot_hide_cross_gap_pairing():
+    catalog, _, _ = fixture(np.ones((2, 10)), names=["A1", "A3"])
+    mapping = [{"channel_id": "c0", "channel_type": "seeg", "status": "confirmed",
+                "measurement_kind": "bipolar", "contact_names": ["A1", "A3"], "reference": "A3"}]
+    resolved = resolve_seeg_channels(catalog, mapping)
+    assert resolved[0]["contact_names"] == ["A1", "A3"]
+    assert "nonadjacent_bipolar_contacts" in resolved[0]["issues"]
+    assert resolved[0]["status"] == "confirmed"
+
+
+def test_duplicate_signals_are_only_relation_candidates():
+    x = np.random.default_rng(4).normal(0, 1e-5, 2500)
+    catalog, read, _ = fixture(np.vstack([x, x]))
+    report = scan_seeg(catalog, read)
+    assert report["initial_score"]["total"] == 100
+    assert any(r["kind"] == "duplicate_signal_candidate" for r in report["relations"])
+
+
+def test_unknown_units_have_no_voltage_metrics_or_false_full_score():
+    catalog, read, _ = fixture(np.zeros(2500), units=["Ohm"])
+    report = scan_seeg(catalog, read)
+    assert report["windows"][0]["metrics"]["rms_v"] is None
+    assert report["windows"][0]["score"] is None
+    assert report["initial_score"]["total"] is None
+    json.dumps(report, allow_nan=False)
+
+
+def test_unconfirmed_file_boundary_is_review_evidence_without_deduction():
+    x = np.random.default_rng(5).normal(0, 1e-5, 2500)
+    x[500:600] = 1e-3
+    catalog, read, _ = fixture(x)
+    report = scan_seeg(catalog, read)
+    assert "boundary_dwell" not in report["windows"][0]["flags"]
+    catalog["streams"][0]["channels"][0].update(physical_min=-1000, physical_max=1000, scale_to_si=1e-6,
+                                                digital_min=-32768, digital_max=32767)
+    original = x.copy()
+    report = scan_seeg(catalog, read)
+    window = report["windows"][0]
+    assert "boundary_dwell" in window["flags"]
+    assert "clipping" not in window["flags"]
+    assert window["metrics"]["boundary_dwell_fraction"] == pytest.approx(.04)
+    assert window["metrics"]["longest_boundary_dwell_seconds"] == pytest.approx(.2)
+    assert window["metrics"]["clipping_fraction"] == pytest.approx(.04)
+    assert window["metrics"]["adc_range_status"] == "unconfirmed"
+    assert window["metrics"]["boundary_dwell_score_effect"] == "none"
+    assert window["score"] == report["initial_score"]["total"] == 100
+    clipping = next(c for c in report["initial_score"]["components"] if c["flag"] == "clipping")
+    assert clipping["evaluated_channel_seconds"] == 0
+    np.testing.assert_array_equal(x, original)
+
+
+def test_boundary_evidence_does_not_suppress_independent_flat_detection():
+    catalog, read, _ = fixture(np.full(2500, 1e-3))
+    catalog["streams"][0]["channels"][0].update(physical_min=-1000, physical_max=1000, scale_to_si=1e-6,
+                                                digital_min=-32768, digital_max=32767)
+    window = scan_seeg(catalog, read)["windows"][0]
+    assert "boundary_dwell" in window["flags"]
+    assert "flat" in window["flags"]
+    assert "clipping" not in window["flags"]
+    assert window["score"] == 0
+
+
+def test_nonfinite_and_constant_have_no_spectral_fabrication():
+    catalog, read, _ = fixture(np.vstack([np.full(2500, np.nan), np.zeros(2500)]))
+    report = scan_seeg(catalog, read)
+    assert all(s["status"] == "insufficient_data" for s in report["spectra"])
+    assert report["windows"][0]["flags"] == ["nonfinite"]
+    assert report["windows"][1]["flags"] == ["flat"]
+    json.dumps(report, allow_nan=False)
+
+
+@pytest.mark.parametrize("config", [{"window_seconds": 0}, {"window_seconds": True}, {"line_frequency": 55},
+                                     {"unknown_option": 1}, {"ranges": []}, {"max_windows": 0}])
+def test_invalid_configuration_is_rejected(config):
+    catalog, read, _ = fixture(np.ones(100))
+    with pytest.raises(ValueError):
+        scan_seeg(catalog, read, config=config)
+
+
+def test_score_uses_max_penalty_and_review_is_duration_weighted():
+    windows = [{"window_id": "a", "duration_seconds": 2, "flags": ["flat", "clipping"],
+                "score": 0, "score_eligible": True},
+               {"window_id": "b", "duration_seconds": 8, "flags": [], "score": 100, "score_eligible": True}]
+    assert compute_quality_score(windows)["total"] == 80
+    assert compute_quality_score(windows, {"a": "rejected"})["total"] == 100
+    assert compute_quality_score(windows, {"b": "accepted"})["total"] == 0
+
+
+def test_vendor_prefix_does_not_set_modality_and_auxiliary_is_not_scored():
+    rng = np.random.default_rng(41)
+    names = ["EEG A1-Ref", "POL A3", "POL C1", "ECG", "TRIG"]
+    catalog, read, _ = fixture(rng.normal(0, 1e-5, (5, 2500)), names=names)
+    report = scan_seeg(catalog, read)
+    assert report["channel_map"][0]["raw_name"] == "EEG A1-Ref"
+    assert report["channel_map"][0]["candidate"]["shaft"] == "A"
+    assert report["channel_map"][0]["candidate"]["reference"] == "Ref"
+    assert report["channel_map"][0]["channel_type"] == "unknown"
+    assert [w["score"] for w in report["windows"]] == [100, 100, 100, None, None]
+
+
+def test_explicit_seeg_type_overrides_name_candidate_without_modifying_signal():
+    x = np.random.default_rng(3).normal(0, 1e-5, (2, 2500))
+    catalog, read, _ = fixture(x, names=["ECG1", "C3"])
+    report = scan_seeg(catalog, read, channel_map=[{"channel_id": "c0", "channel_type": "seeg", "status": "confirmed"}])
+    assert report["windows"][0]["score"] == 100
+    assert report["windows"][1]["score"] == 100
+
+
+def test_bad_ranges_and_resources_fail_before_reading():
+    catalog, read, calls = fixture(np.ones(6000))
+    span = {"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 1000}
+    for config in ({"ranges": [span, span]}, {"ranges": [{**span, "stop_sample": 7000}]},
+                   {"max_scanned_samples": 100}, {"max_windows": 1}):
+        with pytest.raises(ValueError):
+            scan_seeg(catalog, read, config=config)
+    assert calls == []
+
+
+def test_review_cannot_turn_unknown_unit_into_a_full_score():
+    catalog, read, _ = fixture(np.zeros(2500), units=["unknown"])
+    report = scan_seeg(catalog, read)
+    score = compute_quality_score(report["windows"], {report["windows"][0]["window_id"]: "rejected"})
+    assert score["total"] is None
+    assert score["unassessed_channel_seconds"] == 5
+
+
+def test_numeric_overflow_is_explicit_and_not_a_json_nan_result():
+    catalog, read, _ = fixture(np.full(2500, 1e200))
+    with pytest.raises(ValueError, match="numeric|Numerical"):
+        scan_seeg(catalog, read)
diff --git a/backend/tests/contract/test_seeg_qc_waveform.py b/backend/tests/contract/test_seeg_qc_waveform.py
new file mode 100644
index 0000000000000000000000000000000000000000..f4ce740941cc187f8106d598b6edd91ceb57747d
--- /dev/null
+++ b/backend/tests/contract/test_seeg_qc_waveform.py
@@ -0,0 +1,171 @@
+"""Contextual SEEG waveform events, including counterexamples and censoring."""
+import json
+
+import numpy as np
+import pytest
+
+from backend.app.processing.ephys_qc.seeg import scan_seeg
+from backend.tests.contract.test_seeg_qc_metrics import fixture
+
+
+def baseline(seconds=12, fs=500):
+    return np.random.default_rng(73).normal(0, 8e-6, int(seconds * fs))
+
+
+def scan(values, seconds=5, segments=None, config=None):
+    catalog, read, calls = fixture(values, segments=segments)
+    return scan_seeg(catalog, read, config={"window_seconds": seconds, **(config or {})}), calls
+
+
+def event_signature(report):
+    return [(e["start_sample"], e["stop_sample"], e["recovery_status"],
+             [(s["kind"], s["start_sample"], s["stop_sample"]) for s in e["stages"]])
+            for e in report["waveform_events"]]
+
+
+def test_step_plateau_recovery_is_one_event_across_qc_windows():
+    x = baseline()
+    x[2400:4100] += .001
+    report, _ = scan(x)
+    events = report["waveform_events"]
+    assert len(events) == 1
+    event = events[0]
+    assert event["start_sample"] == 2400
+    assert event["stop_sample"] >= 4250
+    assert event["recovery_status"] == "confirmed"
+    assert {s["kind"] for s in event["stages"]} >= {"jump", "plateau", "recovery"}
+    assert event["score_effect"] == "none"
+    assert len([w for w in report["windows"] if event["event_id"] in w["waveform_event_ids"]]) == 2
+    assert all(w["score"] == 100 for w in report["windows"])
+    smaller, _ = scan(x, seconds=.5)
+    assert event_signature(smaller) == event_signature(report)
+    json.dumps(report, allow_nan=False)
+
+
+def test_ramp_into_plateau_has_drift_and_recovery_stages():
+    x = baseline()
+    x[1500:2000] += np.linspace(0, .0015, 500, endpoint=False)
+    x[2000:3250] += .0015
+    x[3250:3750] += np.linspace(.0015, 0, 500, endpoint=False)
+    report, _ = scan(x)
+    assert len(report["waveform_events"]) == 1
+    event = report["waveform_events"][0]
+    assert event["recovery_status"] == "confirmed"
+    assert {s["kind"] for s in event["stages"]} >= {"drift", "plateau", "recovery"}
+    assert event["start_sample"] <= 1600
+    assert event["stop_sample"] >= 3750
+
+
+def test_no_return_before_tail_is_right_censored_not_fake_recovery():
+    x = baseline(seconds=7)
+    x[2400:] += .001
+    report, _ = scan(x)
+    event = report["waveform_events"][0]
+    assert event["stop_sample"] == len(x)
+    assert event["recovery_status"] == "right_censored"
+    assert event["termination_reason"] == "segment_end"
+    assert not any(s["kind"] == "recovery" for s in event["stages"])
+
+
+def test_gap_and_unscanned_ranges_break_context():
+    x = baseline(seconds=10)
+    x[1500:2500] += .001
+    segments = [{"segment_id": "a", "start_sample": 0, "stop_sample": 2500, "recording_start_seconds": 0},
+                {"segment_id": "b", "start_sample": 2500, "stop_sample": 5000, "recording_start_seconds": 20}]
+    report, _ = scan(x, segments=segments)
+    assert len(report["waveform_events"]) == 1
+    event = report["waveform_events"][0]
+    assert event["segment_id"] == "a" and event["stop_sample"] == 2500
+    assert event["recovery_status"] == "right_censored"
+    selected, _ = scan(x, config={"ranges": [{"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 2000},
+                                            {"stream_id": "s", "segment_id": "p", "start_sample": 2300, "stop_sample": 5000}]})
+    assert selected["waveform_events"][0]["stop_sample"] == 2000
+    assert selected["waveform_events"][0]["termination_reason"] == "range_end"
+
+
+def test_adjacent_explicit_ranges_preserve_same_event_context():
+    x = baseline()
+    x[2400:4100] += .001
+    report, _ = scan(x)
+    adjacent, _ = scan(x, config={"ranges": [{"stream_id": "s", "segment_id": "p", "start_sample": 0, "stop_sample": 2500},
+                                            {"stream_id": "s", "segment_id": "p", "start_sample": 2500, "stop_sample": len(x)}]})
+    assert event_signature(adjacent) == event_signature(report)
+
+
+def test_nonfinite_breaks_continuity_and_does_not_bridge_recovery():
+    x = baseline()
+    x[1500:3500] += .001
+    x[2500:2550] = np.nan
+    report, _ = scan(x)
+    first = report["waveform_events"][0]
+    assert first["stop_sample"] == 2500
+    assert first["termination_reason"] == "nonfinite_gap"
+    assert first["recovery_status"] == "right_censored"
+
+
+@pytest.mark.parametrize("kind", ["single_spike", "oscillation", "slow_wave", "constant"])
+def test_physiological_counterexamples_do_not_emit_joint_artifact(kind):
+    fs = 500
+    x = baseline(seconds=20)
+    t = np.arange(len(x)) / fs
+    if kind == "single_spike":
+        x[2400:2403] += [0.002, -.003, .001]
+    elif kind == "oscillation":
+        x += .0004 * np.sin(2 * np.pi * 12 * t)
+    elif kind == "slow_wave":
+        x += .0006 * np.sin(2 * np.pi * .2 * t)
+    else:
+        x[:] = .001
+    report, _ = scan(x)
+    assert report["waveform_events"] == []
+
+
+def test_brief_baseline_crossing_does_not_confirm_recovery():
+    x = baseline()
+    x[1500:] += .001
+    x[3000:3020] -= .001
+    report, _ = scan(x)
+    assert len(report["waveform_events"]) == 1
+    assert report["waveform_events"][0]["recovery_status"] == "right_censored"
+
+
+def test_waveform_resource_budget_is_preflight_not_silent_sampling():
+    with pytest.raises(ValueError, match="waveform|Waveform"):
+        scan(baseline(), config={"max_waveform_bins": 10})
+
+
+def test_large_event_residual_offset_does_not_count_as_recovery():
+    x = baseline(seconds=20)
+    x[1500:2500] += .003
+    x[2500:3500] += np.linspace(.003, .0003, 1000, endpoint=False)
+    x[3500:] += .0003
+    report, _ = scan(x)
+    assert len(report["waveform_events"]) == 1
+    event = report["waveform_events"][0]
+    assert event["recovery_status"] == "right_censored"
+    assert event["features"]["recovery_tolerance_v"] < .0001
+
+
+def test_exact_subsecond_plateau_boundaries_survive_bin_and_qc_edges():
+    x = baseline()
+    x[2400:3000] += .002
+    x[2423:2544] = .002
+    x[3000:3500] += np.linspace(.002, 0, 500, endpoint=False)
+    report, _ = scan(x)
+    exact = [s for e in report["waveform_events"] for s in e["stages"] if s["kind"] == "plateau"
+             and s["features"]["method"] == "exact_constant_run"]
+    assert any(s["start_sample"] == 2423 and s["stop_sample"] == 2544 for s in exact)
+
+
+def test_isolated_prior_spike_is_not_joined_to_later_connected_drift():
+    x = baseline()
+    x[2200:2203] += [.003, -.003, .001]
+    x[2500:2530] += np.linspace(0, .0008, 30, endpoint=False)
+    x[2530:3500] += .003
+    report, _ = scan(x)
+    assert len(report["waveform_events"]) == 1
+    event = report["waveform_events"][0]
+    assert 2500 <= event["start_sample"] < 2530
+    assert event["stages"][0]["kind"] == "drift"
+    assert all(s["stop_sample"] > s["start_sample"] >= event["start_sample"] for s in event["stages"])
+    assert all(s["stop_sample"] <= event["stop_sample"] for s in event["stages"])
diff --git a/backend/tests/ephys_edf_fixtures.py b/backend/tests/ephys_edf_fixtures.py
new file mode 100644
index 0000000000000000000000000000000000000000..04c745b3dc8336bba2ed33f72bb88f9bf9735000
--- /dev/null
+++ b/backend/tests/ephys_edf_fixtures.py
@@ -0,0 +1,65 @@
+"""Small byte-authored EDF/BDF fixtures; no scientific reader writes the oracle."""
+from pathlib import Path
+
+import numpy as np
+
+
+def write_edf_bdf(path, channels, *, bdf=False, reserved="", duration="1",
+                  annotation_records=None, record_count=None):
+    """channels: dictionaries with label, records (digital lists), optional calibration."""
+    channels = [dict(channel) for channel in channels]
+    width = 3 if bdf else 2
+    if annotation_records is not None:
+        size = max(128, max(map(len, annotation_records)))
+        samples = (size + width - 1) // width
+        channels.append({"label": ("BDF" if bdf else "EDF") + " Annotations",
+                         "records": annotation_records, "annotation": True,
+                         "samples": samples, "unit": ""})
+    n_records = len(channels[0]["records"])
+    n_signals = len(channels)
+    limit = 2 ** (width * 8 - 1)
+
+    def field(value, size):
+        raw = value if isinstance(value, bytes) else str(value).encode("ascii")
+        assert len(raw) <= size
+        return raw.ljust(size, b" ")
+
+    header = (b"\xffBIOSEMI" if bdf else b"0       ")
+    header += field("X", 80) + field("Startdate X X X X", 80)
+    header += b"01.01.2412.00.00" + field((n_signals + 1) * 256, 8)
+    header += field(reserved, 44) + field(n_records if record_count is None else record_count, 8)
+    header += field(duration, 8) + field(n_signals, 4)
+    fields = [("label", 16, ""), ("transducer", 80, ""), ("unit", 8, "uV"),
+              ("physical_min", 8, -limit), ("physical_max", 8, limit - 1),
+              ("digital_min", 8, -limit), ("digital_max", 8, limit - 1),
+              ("prefilter", 80, ""), ("samples", 8, None), ("reserved", 32, "")]
+    for key, size, default in fields:
+        for channel in channels:
+            value = channel.get(key, default)
+            if key == "samples" and value is None:
+                value = len(channel["records"][0])
+            header += field(value, size)
+    payload = bytearray()
+    for record in range(n_records):
+        for channel in channels:
+            values = channel["records"][record]
+            if channel.get("annotation"):
+                payload.extend(values.ljust(channel["samples"] * width, b"\0"))
+            elif bdf:
+                for value in values:
+                    payload.extend((int(value) & 0xffffff).to_bytes(3, "little"))
+            else:
+                payload.extend(np.asarray(values, dtype="<i2").tobytes())
+    Path(path).write_bytes(header + payload)
+    return Path(path)
+
+
+def tal(onset, text=None, duration=None):
+    onset = str(onset)
+    if not onset.startswith(("+", "-")):
+        onset = "+" + onset
+    stamp = onset.encode("ascii")
+    if duration is not None:
+        stamp += b"\x15" + str(duration).encode("ascii")
+    text = b"" if text is None else (text if isinstance(text, bytes) else text.encode("utf-8"))
+    return stamp + b"\x14" + text + b"\x14\x00"
diff --git a/backend/tests/ephys_fixtures.py b/backend/tests/ephys_fixtures.py
new file mode 100644
index 0000000000000000000000000000000000000000..965b87f67609e7da65f4580e1c4439dfeb454d42
--- /dev/null
+++ b/backend/tests/ephys_fixtures.py
@@ -0,0 +1,176 @@
+"""Rebuildable S0 signal-contract examples; not a file decoder or QC algorithm.
+
+Call ``build_ephys_fixture(tmp_path / "recording")``. All writes stay beneath
+the supplied directory; this module does not import application storage. The
+manifest is a synthetic fixture schema, not a claim that production readers
+already implement it. No random state, wall-clock time, or absolute path enters
+the fixture bytes.
+"""
+from __future__ import annotations
+
+import hashlib
+import json
+from pathlib import Path
+
+import numpy as np
+
+
+def build_ephys_fixture(directory: Path) -> dict:
+    """Create a new small numpy/JSON fixture, refusing to overwrite a directory.
+
+    Arrays use channel x sample order; every segment has its own sample origin.
+    Voltage rows have equivalent V/uV representations; TRIG is dimensionless in
+    both. Physical sites, recorded channels, and a recorded bipolar derivation
+    are deliberately separate. Expected decisions are examples, not automatic
+    recommendations to remove data.
+    """
+    root = Path(directory)
+    root.mkdir(parents=True, exist_ok=False)
+    channel_names = [
+        "C3", "C4", "A1", "A3", "A1-A2", "G3", "G1", "G4", "G2", "ECG", "TRIG",
+    ]
+    channel_types = [
+        "EEG", "EEG", "SEEG", "SEEG", "SEEG", "ECOG", "ECOG", "ECOG", "ECOG", "ECG", "TRIG",
+    ]
+    groups = [
+        "scalp", "scalp", "shaft-A", "shaft-A", "shaft-A", "grid-G", "grid-G",
+        "grid-G", "grid-G", "auxiliary", "auxiliary",
+    ]
+    channels = []
+    for index, (name, kind, group) in enumerate(zip(channel_names, channel_types, groups)):
+        channel = {
+            "channel_id": name, "array_index": index, "type": kind, "group_id": group,
+            "quantity": "digital_state" if kind == "TRIG" else "voltage",
+            "electrode_ids": [] if kind == "TRIG" else [name],
+            "reference_electrode_ids": [] if kind == "TRIG" else [f"REF_{group}"],
+            "measurement": "digital" if kind == "TRIG" else "referential",
+        }
+        if name == "A1-A2":
+            channel.update({
+                "measurement": "recorded_bipolar", "electrode_ids": ["A1", "A2"],
+                "reference_electrode_ids": [],
+                "electrode_weights": {"A1": 1.0, "A2": -1.0},
+                "monopolar_reconstruction_available": False,
+            })
+        channels.append(channel)
+
+    files = {}
+    segments = []
+    for segment_id, start, length in (("seg-0", 1000, 256), ("seg-1", 1500, 192)):
+        # Integer-valued ramps avoid ambiguous FFT or stochastic ground truth.
+        offsets = np.arange(length, dtype=np.int64)
+        uv = np.stack([((offsets + row * 3) % 16 - 8) * (row + 1) for row in range(11)])
+        uv = uv.astype(np.float64)
+        uv[-1] = 0.0
+        if segment_id == "seg-0":
+            uv[0, 16], uv[0, 17] = -25.0, 75.0  # signed, known EEG samples
+            uv[0, 100:116] += 500.0  # only C3, half-open [1100, 1116)
+            uv[-1, 101] = 1.0  # digital event inside the C3-only bad interval
+        volts = uv.copy()
+        volts[:-1] *= 1e-6
+        for representation, array in (("uV", uv), ("V", volts)):
+            filename = f"{segment_id}-{representation}.npy"
+            np.save(root / filename, array, allow_pickle=False)
+            files[filename] = root / filename
+        segments.append({
+            "segment_id": segment_id, "start_sample": start, "stop_sample": start + length,
+            "n_samples": length, "recording_start_seconds": (start - 1000) / 256,
+            "arrays": {"uV": f"{segment_id}-uV.npy", "V": f"{segment_id}-V.npy"},
+        })
+
+    bipolar_filename = "bipolar-only-V.npy"
+    bipolar = np.load(root / "seg-0-V.npy", allow_pickle=False)[4:5].copy()
+    np.save(root / bipolar_filename, bipolar, allow_pickle=False)
+    files[bipolar_filename] = root / bipolar_filename
+
+    spike_ids = np.array(["spike-a0", "spike-b0", "spike-a1", "spike-b1", "spike-a2", "spike-b2"])
+    spike_samples = np.array([31032, 31032, 31256, 31900, 35300, 37999], dtype=np.int64)
+    spike_units = np.array([1, 2, 1, 2, 1, 2], dtype=np.int16)
+    for filename, array in (("spike_ids.npy", spike_ids), ("spike_samples.npy", spike_samples),
+                            ("spike_unit_ids.npy", spike_units)):
+        np.save(root / filename, array, allow_pickle=False)
+        files[filename] = root / filename
+
+    expected = {
+        "physical_value_tolerance_volts": 1e-12,
+        "signed_samples": [
+            {"channel_id": "C3", "sample": 1016, "segment_id": "seg-0", "value_volts": -25e-6},
+            {"channel_id": "C3", "sample": 1017, "segment_id": "seg-0", "value_volts": 75e-6},
+        ],
+        "gap_samples": {"start": 1256, "stop": 1500, "length": 244},
+        "observed_sample_count": 448,
+        "elapsed_sample_span": 692,
+        "missing_recorded_channel_ids": ["Pz", "A2"],
+        "bad_intervals": [{"channel_ids": ["C3"], "stream_id": "mixed-256",
+                           "segment_id": "seg-0", "start_sample": 1100, "stop_sample": 1116,
+                           "reason": "synthetic_offset", "status": "confirmed_example"}],
+        "not_inferred_bipolar_pairs": [["A1", "A3"]],
+        "ecog_grid_positions": {"G1": [0, 0], "G2": [0, 1], "G3": [1, 0], "G4": [1, 1]},
+        "ecog_adjacency": [["G1", "G2"], ["G1", "G3"], ["G2", "G4"], ["G3", "G4"]],
+        "same_sample_distinct_spike_ids": ["spike-a0", "spike-b0"],
+        "no_dedup_retained_spike_ids": spike_ids.tolist(),
+        "explicit_exclusion_example": {
+            "policy": "explicit_spike_id_exclusion", "removed_spike_ids": ["spike-b0"],
+            "retained_spike_ids": [item for item in spike_ids.tolist() if item != "spike-b0"],
+            "audit": [{"spike_id": "spike-b0", "reason": "synthetic_explicit_decision",
+                       "evidence": "fixture-only; same-sample timing alone is not duplicate evidence"}],
+        },
+    }
+    manifest = {
+        "schema": "cortrix.synthetic-ephys-fixture", "schema_version": 1,
+        "synthetic": True, "scope": "signal_contract_baseline_only",
+        "not_validated": ["EDF/BDF/Neuracle decoding", "QC algorithms", "clinical or scientific quality"],
+        "interval_convention": "half_open", "array_axes": ["channel", "sample"],
+        "recording_example": {
+            "recording_id": "fixture-recording-001", "subject_id": "fixture-subject",
+            "label": "mixed-recording", "status": "unreviewed", "metadata_source": "synthetic_manifest",
+        },
+        "streams": [{
+            "stream_id": "mixed-256", "sampling_frequency_hz": 256.0,
+            "clock_id": "mixed-sample-clock", "sample_origin": 1000,
+            "time_origin_seconds": 0.0, "channels": channels, "segments": segments,
+            "representation_units": {
+                "uV": {"voltage": "uV", "digital_state": "1"},
+                "V": {"voltage": "V", "digital_state": "1"},
+            },
+            "expected_acquisition_channel_ids": channel_names + ["Pz", "A2"],
+            "physical_electrodes": [
+                {"electrode_id": name, "group_id": group}
+                for name, group in [("C3", "scalp"), ("C4", "scalp"), ("Pz", "scalp"),
+                                    ("A1", "shaft-A"), ("A2", "shaft-A"), ("A3", "shaft-A"),
+                                    ("G1", "grid-G"), ("G2", "grid-G"), ("G3", "grid-G"),
+                                    ("G4", "grid-G"), ("ECG", "auxiliary"),
+                                    ("REF_scalp", "scalp"), ("REF_shaft-A", "shaft-A"),
+                                    ("REF_grid-G", "grid-G"), ("REF_auxiliary", "auxiliary")]
+            ],
+        }, {
+            "stream_id": "spikes-30000", "sampling_frequency_hz": 30000.0,
+            "clock_id": "probe-sample-clock", "sample_origin": 30000,
+            "time_origin_seconds": None, "clock_alignment_to_mixed": "unknown",
+            "segments": [{"segment_id": "probe-0", "start_sample": 30000, "stop_sample": 34096},
+                         {"segment_id": "probe-1", "start_sample": 35000, "stop_sample": 38000}],
+            "spike_arrays": {"spike_id": "spike_ids.npy", "sample": "spike_samples.npy",
+                             "unit_id": "spike_unit_ids.npy"},
+            "sample_semantics": "absolute_integer_sample_in_probe_stream",
+            "unit_ids": [1, 2], "voltage_available": False,
+        }, {
+            "stream_id": "bipolar-only-256", "sampling_frequency_hz": 256.0,
+            "clock_id": "mixed-sample-clock", "sample_origin": 1000,
+            "time_origin_seconds": 0.0,
+            "channels": [{**channels[4], "array_index": 0}],
+            "segments": [{"segment_id": "bipolar-0", "start_sample": 1000,
+                          "stop_sample": 1256, "n_samples": 256,
+                          "arrays": {"V": bipolar_filename}}],
+            "physical_electrode_ids": ["A1", "A2"],
+            "monopolar_channels_available": False,
+            "representation_units": {"V": {"voltage": "V"}},
+        }],
+        "expected": expected,
+        "files": [{"path": name, "size_bytes": path.stat().st_size,
+                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
+                  for name, path in sorted(files.items())],
+    }
+    manifest_path = root / "manifest.json"
+    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
+    return {"root": root, "manifest_path": manifest_path, "manifest": manifest,
+            "files": files, "expected": expected}
diff --git a/docs/pipelines/workflow.md b/docs/pipelines/workflow.md
index d425eb2d7e8679c8f37cf5e2935af43d7857a506..7ee8f572684244b4ab490cf6749fd78640757bc2 100644
--- a/docs/pipelines/workflow.md
+++ b/docs/pipelines/workflow.md
@@ -1362,2 +1362,5 @@ output/{subject}/
 **历史更新**:
+- v4.7 (2026-10-27): SEEG QC 增加跨窗跳变、漂移、平台与恢复联合事件，以及阶段跳转和事件 CSV；ADC 量程未确认的边界停留只作复核证据。两份真实记录完成分段验证和浏览器复核导出。
+- v4.6 (2026-10-26): SEEG 基础 QC v1 完成：真实电压分窗、通道映射、技术特征、人工复核、评分更新、冻结与 ZIP 导出。入口为电生理 → SEEG → SEEG QC；后台任务为 `ephys_qc_scan`。
+- v4.5 (2026-10-22): 电生理 QC S2 新增独立保真读取路径。`ephys_qc/readers` 按 EDF/BDF 头定标读取数字码或物理值，电压统一为 V；保留辅助通道、重复名称、混合采样率、TAL 分段缺口和事件原字节，不调用旧清理/回放 loader。`GET /api/v1/ephys/artifacts/{id}/signals|events` 提供控制信息，`samples|overview` 返回 `EPHY` + little-endian uint32 JSON 长度 + UTF-8 metadata + 补齐至 8 字节 + `<f8` C-order 数据；样本为 channel × sample，概览为 channel × bin × min/max，均绑定 artifact/hash、stream、segment、半开样本区间、单位和时钟。samples 上限 16 MiB；overview 最多 4096 bins、扫描 256 MiB，内部按小窗口读取。首次验证输入 SHA256，后续用 inode/size/mtime/ctime 守卫，变化即重新核验，读中变化丢弃结果。内部 `publish_voltage` 将有父输入、处理参数、软件版本和显式通道权重的 NPY 电压保存为新的 `derived_voltage` artifact；v2 迁移保留每条 recording 唯一 raw input，允许多派生版本。首版派生映射保留父采样率/时钟，重采样须另扩展明确映射，不作隐式换算。旧上传/回放入口及包络语义不变；新概览不能冒充精确样本。供应商异常 TAL 保留歧义解释，跨文件同步仍未验证；保真读取与产物发布均不代表科学 QC 通过，状态保持 pending。人工决定/实际处理作业及工作台分别属于 S3/S4。
 - v4.4 (2026-07-19，合并前说明修订于 2026-10-19): SEEG **Stage-1** 检测增加按扫描内 cohort 自适应的 seed/axis 强度阈值、饱和平台评分、头部金属簇证据与 seed ladder 累积。空间离群处理采用 `_annotate_spatial_outliers`：候选同时满足最近邻距离 >55mm、seed 强度低于其余候选中位数的 0.6 倍且非 `plan_matched` 时，保留候选并写入 `spatial_outlier_evidence`、`isolated_weak_seed_unvalidated` 和 degraded 信息；不直接剔除，`n_rejected_spatial_outlier` 固定为 0，另报 `n_degraded_spatial_outlier`。配套契约覆盖强 seed 远端电极、簇内弱 seed、双侧/分散植入与已匹配计划等反例。旧的直接剔除规则、单受试者结果及旧测试计数不作为当前实现的验证证据；科学边界与核验工具见 ADR-029。Stage-1 head 检测不经过 Stage-2 的 rigid/cylinder 流程。
diff --git a/frontend/public/templates/viewer/tab-visualization.html b/frontend/public/templates/viewer/tab-visualization.html
index 4874b39bdbc2389ddbb736ba1785c2c7fdd38ca9..9a76308beb86eecfab368cd819c8a808995fcd6b 100644
--- a/frontend/public/templates/viewer/tab-visualization.html
+++ b/frontend/public/templates/viewer/tab-visualization.html
@@ -218,2 +218,3 @@
                                 <button type="button" class="ephys-enter-btn" data-action="seeg-panel" data-i18n="SEEG Analysis Panel">SEEG分析面板</button>
+                                <button type="button" class="ephys-enter-btn" data-action="open-seeg-qc" data-i18n="SEEG QC">SEEG QC</button>
                             </div>
diff --git a/frontend/scripts/test-ephys-binary.mjs b/frontend/scripts/test-ephys-binary.mjs
new file mode 100644
index 0000000000000000000000000000000000000000..a733394fd92f505533c31de3d67dafdfe5b420c1
--- /dev/null
+++ b/frontend/scripts/test-ephys-binary.mjs
@@ -0,0 +1,236 @@
+import assert from 'node:assert/strict';
+import { readFile } from 'node:fs/promises';
+import { test } from 'node:test';
+import { decodeEphysSamples, decodeEphysOverview, prepareEphysSampleRequest } from '../src/core/ephysBinary.js';
+
+const header = {
+    schema_version: 1, recording_id: 'recording-1', artifact_id: 'artifact-1',
+    content_sha256: 'a'.repeat(64), reader_version: 'edf-reader/1',
+    stream_id: 'stream-1', segment_id: 'segment-2',
+    channel_ids: ['channel-4', 'channel-1'], units: ['V', null], shape: [2, 3],
+    dtype: '<f8', order: 'C', start_sample: 11, stop_sample: 14,
+    sampling_frequency_hz: 500, recording_start_seconds: 17.25,
+    clock_id: 'file-clock-1', clock_alignment: 'source_local_unaligned',
+    source_sha256: 'b'.repeat(64), source_relative_path: 'source.edf', interval_convention: 'half_open',
+    sample_index_origin: 0,
+    representation: 'physical', data_kind: 'raw_samples', processing: [],
+};
+const expected = {
+    artifact_id: header.artifact_id, stream_id: header.stream_id,
+    segment_id: header.segment_id, channel_ids: [...header.channel_ids],
+    start_sample: 11, stop_sample: 14, representation: 'physical',
+};
+const values = [-25e-6, 0, 50e-6, -3, 0, 7];
+
+function encode(metadata = header, samples = values) {
+    const text = new TextEncoder().encode(JSON.stringify(metadata));
+    const offset = Math.ceil((8 + text.length) / 8) * 8;
+    const bytes = new Uint8Array(offset + samples.length * 8);
+    bytes.set([69, 80, 72, 89]);
+    new DataView(bytes.buffer).setUint32(4, text.length, true);
+    bytes.set(text, 8);
+    samples.forEach((value, index) => new DataView(bytes.buffer).setFloat64(offset + index * 8, value, true));
+    return bytes;
+}
+
+test('channel-major physical values, sign, order, unknown units and segment time survive', () => {
+    const decoded = decodeEphysSamples(encode().buffer, expected);
+    assert.deepEqual([...decoded.samples], values);
+    assert.deepEqual(decoded.metadata, header);
+    assert.deepEqual([...decoded.samples.subarray(3, 6)], [-3, 0, 7]);
+});
+
+test('unaligned Uint8Array views decode only their own bytes', () => {
+    const frame = encode();
+    const wrapped = new Uint8Array(frame.length + 5);
+    wrapped.set(frame, 3);
+    assert.deepEqual([...decodeEphysSamples(wrapped.subarray(3, 3 + frame.length), expected).samples], values);
+});
+
+test('digital values have an explicit representation and per-channel units', () => {
+    const metadata = { ...header, representation: 'digital', units: ['count', 'count'] };
+    const result = decodeEphysSamples(encode(metadata, [-32768, 0, 32767, -1, 2, 3]), { ...expected, representation: 'digital' });
+    assert.equal(result.samples[0], -32768);
+    assert.deepEqual(result.metadata.units, ['count', 'count']);
+});
+
+test('nonfinite raw samples are retained without silent replacement', () => {
+    const result = decodeEphysSamples(encode(header, [NaN, Infinity, -Infinity, 0, 1, 2]), expected);
+    assert.ok(Number.isNaN(result.samples[0]));
+    assert.equal(result.samples[1], Infinity);
+});
+
+test('missing magic, truncated header, invalid UTF-8 and invalid JSON reject', () => {
+    const magic = encode(); magic[0] = 0;
+    assert.throws(() => decodeEphysSamples(magic, expected), /magic/);
+    assert.throws(() => decodeEphysSamples(new Uint8Array(7), expected), /truncated/);
+    const length = encode(); new DataView(length.buffer).setUint32(4, length.length, true);
+    assert.throws(() => decodeEphysSamples(length, expected), /header/);
+    const utf8 = encode(); utf8[8] = 255;
+    assert.throws(() => decodeEphysSamples(utf8, expected), /header/);
+    const json = encode(); json[8] = 120;
+    assert.throws(() => decodeEphysSamples(json, expected), /header/);
+});
+
+test('inexact payload length and nonzero alignment padding reject', () => {
+    const frame = encode();
+    assert.throws(() => decodeEphysSamples(frame.subarray(0, frame.length - 1), expected), /length/);
+    const extra = new Uint8Array(frame.length + 8); extra.set(frame);
+    assert.throws(() => decodeEphysSamples(extra, expected), /length/);
+    const metadata = { ...header, reader_version: 'x' };
+    while ((8 + new TextEncoder().encode(JSON.stringify(metadata)).length) % 8 === 0) metadata.reader_version += 'x';
+    const padded = encode(metadata);
+    padded[8 + new DataView(padded.buffer).getUint32(4, true)] = 1;
+    assert.throws(() => decodeEphysSamples(padded, expected), /padding/);
+});
+
+for (const [field, value] of [
+    ['schema_version', 2], ['dtype', '>f8'], ['order', 'F'],
+    ['shape', [3, 2]], ['shape', [2, 3.5]], ['shape', [2, Number.MAX_SAFE_INTEGER]],
+    ['units', ['V']], ['units', ['V', 0]], ['channel_ids', ['channel-4', 'channel-4']],
+    ['sampling_frequency_hz', 0], ['recording_start_seconds', null],
+    ['content_sha256', 'unknown'], ['reader_version', ''],
+    ['clock_id', ''], ['clock_alignment', 'assumed_aligned'], ['source_sha256', 'unknown'],
+    ['source_relative_path', ''], ['interval_convention', 'closed'],
+    ['sample_index_origin', -1], ['sample_index_origin', 12],
+    ['data_kind', 'overview'], ['processing', ['average_reference']],
+]) {
+    test(`invalid header ${field}=${JSON.stringify(value)} rejects`, () => {
+        assert.throws(() => decodeEphysSamples(encode({ ...header, [field]: value }), expected), { code: 'invalid_ephys_binary' });
+    });
+}
+
+for (const [field, value] of [
+    ['artifact_id', 'artifact-2'], ['stream_id', 'stream-2'], ['segment_id', 'segment-1'],
+    ['channel_ids', ['channel-1', 'channel-4']], ['start_sample', 10],
+    ['stop_sample', 15], ['representation', 'digital'], ['recording_id', 'recording-2'],
+    ['content_sha256', 'b'.repeat(64)], ['reader_version', 'edf-reader/2'],
+]) {
+    test(`response identity mismatch ${field} rejects`, () => {
+        assert.throws(() => decodeEphysSamples(encode(), { ...expected, [field]: value }), { code: 'ephys_identity_mismatch' });
+    });
+}
+
+test('request serialization preserves channel ordering and separates optional identity constraints', () => {
+    const prepared = prepareEphysSampleRequest('artifact/1', { ...expected, recording_id: 'recording-1' });
+    assert.deepEqual(prepared.query.getAll('channel_id'), header.channel_ids);
+    assert.equal(prepared.query.get('representation'), 'physical');
+    assert.equal(prepared.query.has('recording_id'), false);
+    assert.equal(prepared.expected.artifact_id, 'artifact/1');
+    expected.channel_ids.push('later-mutation');
+    assert.equal(prepared.expected.channel_ids.length, 2);
+    expected.channel_ids.pop();
+});
+
+test('missing/ambiguous request fields cannot trigger guessed units or windows', () => {
+    for (const update of [
+        { channel_ids: [] }, { channel_ids: ['x', 'x'] }, { representation: undefined },
+        { start_sample: -1 }, { start_sample: 1.5 }, { stop_sample: 11 },
+        { stop_sample: Number.MAX_SAFE_INTEGER + 1 }, { segment_id: '' },
+    ]) assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...expected, ...update }));
+    assert.throws(() => decodeEphysSamples(encode(), undefined));
+});
+
+const overviewHeader = {
+    ...header, data_kind: 'minmax_overview', source_data_kind: 'raw_samples', shape: [2, 2, 2], bin_samples: 2,
+    axes: ['channel', 'bin', 'extremum'], extrema: ['min', 'max'], bin_start_samples: [11, 13], bin_stop_samples: [13, 14],
+};
+const overviewValues = [-25e-6, 0, 50e-6, 50e-6, -3, 0, 7, 7];
+const overviewExpected = { ...expected, bin_samples: 2 };
+
+test('overview is explicitly binned, unnormalized and retains the short final bin', () => {
+    const result = decodeEphysOverview(encode(overviewHeader, overviewValues), overviewExpected);
+    assert.deepEqual([...result.samples], overviewValues);
+    assert.deepEqual(result.metadata.bin_stop_samples, [13, 14]);
+    assert.throws(() => decodeEphysSamples(encode(overviewHeader, overviewValues), expected), /raw_samples/);
+    assert.throws(() => decodeEphysOverview(encode(), overviewExpected), /minmax_overview/);
+    assert.equal(prepareEphysSampleRequest('artifact-1', overviewExpected, true).query.get('bin_samples'), '2');
+});
+
+const derivedHeader = {
+    ...header, data_kind: 'derived_samples', units: ['V', 'V'], sample_index_origin: 10,
+    processing: [{ operation: 'explicit_reference', version: 1 }],
+    parent_artifact_id: 'raw-artifact-1', parent_content_sha256: 'c'.repeat(64),
+    software_versions: { numpy: '2.0.2' },
+};
+const derivedExpected = { ...expected, data_kind: 'derived_samples' };
+
+test('derived windows require explicit kind, physical volts and complete parent provenance', () => {
+    const result = decodeEphysSamples(encode(derivedHeader), derivedExpected);
+    assert.deepEqual([...result.samples], values);
+    assert.equal(result.metadata.sample_index_origin, 10);
+    assert.equal(result.metadata.parent_artifact_id, 'raw-artifact-1');
+    assert.throws(() => decodeEphysSamples(encode(derivedHeader), expected), /raw_samples/);
+    assert.throws(() => decodeEphysSamples(encode(), derivedExpected), /derived_samples/);
+    for (const update of [
+        { units: ['V', null] }, { units: ['mV', 'V'] }, { representation: 'digital' },
+        { processing: [] }, { parent_artifact_id: '' }, { parent_content_sha256: 'unknown' }, { parent_content_sha256: ['c'.repeat(64)] },
+        { software_versions: {} }, { software_versions: { numpy: null } },
+    ]) assert.throws(() => decodeEphysSamples(encode({ ...derivedHeader, ...update }), derivedExpected));
+    assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...derivedExpected, representation: 'digital' }));
+    assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...expected, data_kind: 'filtered_samples' }));
+});
+
+test('derived overview retains source kind and cannot masquerade as a raw overview', () => {
+    const derivedOverview = { ...derivedHeader, ...overviewHeader, ...derivedHeader,
+        data_kind: 'minmax_overview', source_data_kind: 'derived_samples', shape: [2, 2, 2] };
+    const result = decodeEphysOverview(encode(derivedOverview, overviewValues), { ...derivedExpected, bin_samples: 2 });
+    assert.equal(result.metadata.source_data_kind, 'derived_samples');
+    assert.throws(() => decodeEphysOverview(encode(derivedOverview, overviewValues), overviewExpected));
+    assert.throws(() => decodeEphysOverview(encode({ ...overviewHeader, source_data_kind: undefined }, overviewValues), overviewExpected));
+});
+
+test('overview rejects reshaped axes, wrong bin bounds and changed bin size', () => {
+    for (const update of [
+        { axes: ['bin', 'channel', 'extremum'] }, { axes: [['channel'], 'bin', 'extremum'] },
+        { extrema: ['max', 'min'] }, { extrema: [['min'], 'max'] }, { bin_start_samples: [11, 12] },
+        { bin_stop_samples: [13, 15] }, { shape: [2, 4] }, { bin_samples: 0 },
+    ]) assert.throws(() => decodeEphysOverview(encode({ ...overviewHeader, ...update }, overviewValues), overviewExpected));
+    assert.throws(() => decodeEphysOverview(encode(overviewHeader, overviewValues), { ...overviewExpected, bin_samples: 3 }), { code: 'ephys_identity_mismatch' });
+});
+
+// Only browser configuration/demo selection is replaced for Node. The real
+// ApiClient domain methods, httpClient and response adapter execute below.
+const apiUrl = new URL('../src/core/api.js', import.meta.url);
+const configStub = 'data:text/javascript,' + encodeURIComponent('export const Config = {}; export const DEMO_MODE = false;');
+const demoStub = 'data:text/javascript,' + encodeURIComponent('export function demoResolve(){ throw new Error("unexpected demo"); }');
+const apiSource = (await readFile(apiUrl, 'utf8')).replace(/from ['"](.+?)['"]/g, (match, path) => {
+    const target = path === './config.js' ? configStub : path === './http/demoAdapter.js' ? demoStub : new URL(path, apiUrl).href;
+    return `from ${JSON.stringify(target)}`;
+});
+const { ApiClient } = await import('data:text/javascript;base64,' + Buffer.from(apiSource).toString('base64'));
+
+test('domain methods use real JSON adapter/binary transport and preserve request signal', async () => {
+    const originalFetch = globalThis.fetch;
+    const controller = new AbortController();
+    const calls = [];
+    globalThis.fetch = async (url, options) => {
+        calls.push({ url, options });
+        if (url.includes('/signals')) return new Response(JSON.stringify({ data: { streams: ['s'] } }), { headers: { 'Content-Type': 'application/json' } });
+        if (url.includes('/events')) return new Response(JSON.stringify({ data: { events: [{ event_id: 'event-1' }] } }), { headers: { 'Content-Type': 'application/json' } });
+        if (url.includes('/overview')) return new Response(encode(overviewHeader, overviewValues), { headers: { 'Content-Type': 'application/octet-stream' } });
+        return new Response(encode(), { headers: { 'Content-Type': 'application/octet-stream' } });
+    };
+    try {
+        const api = new ApiClient({ init: async () => {}, getBackendURL: path => `http://isolated.test${path}` });
+        assert.deepEqual(await api.getEphysSignals('artifact/1'), { streams: ['s'] });
+        assert.ok(calls[0].url.endsWith('/artifacts/artifact%2F1/signals'));
+        const result = await api.getEphysSamples('artifact-1', expected, { signal: controller.signal, responseType: 'json' });
+        assert.deepEqual([...result.samples], values);
+        assert.equal(calls[1].options.signal, controller.signal);
+        assert.deepEqual(new URL(calls[1].url).searchParams.getAll('channel_id'), header.channel_ids);
+        assert.equal(calls[1].options.method, 'GET');
+        assert.deepEqual([...(await api.getEphysOverview('artifact-1', overviewExpected)).samples], overviewValues);
+        assert.deepEqual(await api.getEphysEvents('artifact-1', { offset: 10, limit: 2 }), { events: [{ event_id: 'event-1' }] });
+        assert.equal(new URL(calls[3].url).searchParams.get('offset'), '10');
+    } finally { globalThis.fetch = originalFetch; }
+});
+
+test('binary requests still surface structured backend errors through ApiError', async () => {
+    const originalFetch = globalThis.fetch;
+    globalThis.fetch = async () => new Response(JSON.stringify({ error: { code: 'artifact_changed', message: 'Input changed' } }), { status: 409, headers: { 'Content-Type': 'application/json' } });
+    try {
+        const api = new ApiClient({ init: async () => {}, getBackendURL: path => `http://isolated.test${path}` });
+        await assert.rejects(api.getEphysSamples('artifact-1', expected), { name: 'ApiError', code: 'artifact_changed', status: 409 });
+    } finally { globalThis.fetch = originalFetch; }
+});
diff --git a/frontend/scripts/test-seeg-qc-ui.mjs b/frontend/scripts/test-seeg-qc-ui.mjs
new file mode 100644
index 0000000000000000000000000000000000000000..8e85aced66e002640445a134a6138fe280528341
--- /dev/null
+++ b/frontend/scripts/test-seeg-qc-ui.mjs
@@ -0,0 +1,162 @@
+import assert from 'node:assert/strict';
+import { test } from 'node:test';
+import { RequestGeneration, channelMapPayload, mapEditorRow, numberText, percentText, validateDecision, windowRequest, upsertDecision, scanRanges } from '../src/viewer/ephysQc/state.js';
+import { associatedWaveformEvents, eventSelection, eventViewport, reviewWindowEvidence, reviewEventEvidence, boundaryDwellFeatures, waveformStageBands, sampleInputRange, editableEventViewport } from '../src/viewer/ephysQc/waveformEvents.js';
+
+const channel = { channel_id: 'ch-1', raw_name: "A'01" };
+const segment = { segment_id: 'seg-2', start_sample: 100, stop_sample: 500 };
+const stream = { stream_id: 's-1', channels: [channel], segments: [segment], clock_id: 'clock-1' };
+const catalog = { recording_id: 'r-1', artifact_id: 'a-1', streams: [stream] };
+
+test('unconfirmed naming candidates remain suggestions, original name survives', () => {
+    const row = { ...channel, status: 'unreviewed', candidate: { shaft: "A'", contact_index: 1, measurement_kind: 'monopolar' } };
+    assert.equal(mapEditorRow(row).raw_name, "A'01");
+    assert.deepEqual(channelMapPayload([row]), []);
+    const confirmed = { ...mapEditorRow(row), status: 'confirmed', channel_type: 'seeg' };
+    assert.equal(channelMapPayload([confirmed])[0].shaft, "A'");
+    assert.equal(channelMapPayload([confirmed])[0].contact_index, 1);
+});
+
+test('missing metrics never become zero or a passing score', () => {
+    for (const value of [null, undefined, NaN]) assert.equal(numberText(value), '—');
+    assert.equal(numberText(0), '0.00');
+    assert.equal(percentText(5 / 223473), '<0.1%');
+    assert.equal(percentText(0), '0.0%');
+    assert.equal(percentText(null), '—');
+});
+
+test('subject/run/window generation invalidates success and failure callbacks', () => {
+    const gate = new RequestGeneration(), old = gate.next();
+    assert.equal(gate.current(old), true);
+    gate.next();
+    assert.equal(gate.current(old), false);
+});
+
+test('raw requests retain exact source window and cannot cross a gap', () => {
+    assert.equal(windowRequest(catalog, 'ch-1', 'seg-2', 110, 120).representation, 'physical');
+    assert.throws(() => windowRequest(catalog, 'ch-1', 'seg-2', 90, 120));
+    assert.throws(() => windowRequest(catalog, 'ch-1', 'seg-2', 110, 501));
+});
+
+test('manual review requires a valid half-open range and reason', () => {
+    const data = { channel, stream, segment, start: 100, stop: 120, action: 'exclude', reason: 'contact artifact' };
+    const decision = validateDecision(data);
+    assert.equal(decision.stop_sample, 120);
+    assert.equal(decision.scope, 'interval');
+    assert.throws(() => validateDecision({ ...data, start: 120 }));
+    assert.throws(() => validateDecision({ ...data, reason: '' }));
+    assert.equal(upsertDecision([decision], { ...decision, action: 'keep' }).length, 1);
+    assert.equal(upsertDecision([decision], { ...decision, start_sample: 110 }).length, 2);
+});
+
+test('scan seconds split into original source segments without filling a clock gap', () => {
+    const clocked = { streams: [{ stream_id: 's', sampling_frequency_hz: 10, segments: [
+        { segment_id: 'a', start_sample: 0, stop_sample: 100, recording_start_seconds: 0 },
+        { segment_id: 'b', start_sample: 100, stop_sample: 200, recording_start_seconds: 15 },
+    ] }] };
+    assert.deepEqual(scanRanges(clocked, 5, 18), [
+        { stream_id: 's', segment_id: 'a', start_sample: 50, stop_sample: 100 },
+        { stream_id: 's', segment_id: 'b', start_sample: 100, stop_sample: 130 },
+    ]);
+    assert.throws(() => scanRanges(clocked, 11, 14));
+    assert.equal(scanRanges(clocked, 0, null), undefined);
+});
+
+const waveformEvent = {
+    event_id: 'event-1', channel_id: 'ch-1', stream_id: 's-1', segment_id: 'seg-2',
+    start_sample: 180, stop_sample: 360, recovery_status: 'confirmed', score_effect: 'none',
+    stages: [{ kind: 'jump', start_sample: 180, stop_sample: 190 }, { kind: 'plateau', start_sample: 190, stop_sample: 310 },
+        { kind: 'recovery', start_sample: 310, stop_sample: 360 }],
+};
+const eventWindow = { window_id: 'w-1', channel_id: 'ch-1', stream_id: 's-1', segment_id: 'seg-2',
+    start_sample: 100, stop_sample: 200, waveform_event_ids: ['event-1'] };
+
+test('window evidence retains the complete cross-window event, excluding other channels and gaps', () => {
+    const report = { waveform_events: [waveformEvent, { ...waveformEvent, channel_id: 'ch-2' }, { ...waveformEvent, segment_id: 'seg-3' }] };
+    const events = associatedWaveformEvents(report, eventWindow);
+    assert.deepEqual(events, [waveformEvent]);
+    assert.equal(events[0].stop_sample, 360);
+    assert.deepEqual(associatedWaveformEvents({}, eventWindow), []);
+});
+
+test('event and stage selections keep exact original samples without inventing absent stages', () => {
+    assert.equal(eventSelection(waveformEvent).stop_sample, 360);
+    assert.deepEqual(eventSelection(waveformEvent, 1), {
+        channel_id: 'ch-1', stream_id: 's-1', segment_id: 'seg-2', start_sample: 190, stop_sample: 310,
+    });
+    assert.throws(() => eventSelection(waveformEvent, 3));
+    assert.equal(waveformEvent.stages.some(stage => stage.kind === 'drift'), false);
+});
+
+test('paged waveform viewing does not truncate the complete event review range', () => {
+    const selection = { ...eventSelection(waveformEvent), start_sample: 120, stop_sample: 450123 };
+    const pages = [0, 1, 2].map(page => eventViewport(selection, page));
+    assert.deepEqual(pages.map(page => [page.start_sample, page.stop_sample]), [[120, 200120], [200120, 400120], [400120, 450123]]);
+    assert.equal(selection.stop_sample, 450123);
+    assert.throws(() => eventViewport(selection, 3));
+});
+
+test('cross-window event responses carry all overlapping evidence IDs and preserve exact event bounds', () => {
+    const selection = eventSelection(waveformEvent);
+    assert.equal(reviewWindowEvidence(eventWindow, selection), undefined);
+    assert.equal(reviewWindowEvidence(eventWindow, { ...selection, stop_sample: 195 }), 'w-1');
+    const windows = [eventWindow,
+        { ...eventWindow, window_id: 'w-2', start_sample: 200, stop_sample: 300 },
+        { ...eventWindow, window_id: 'w-3', start_sample: 300, stop_sample: 400 },
+        { ...eventWindow, window_id: 'w-4', start_sample: 400, stop_sample: 500 },
+        { ...eventWindow, window_id: 'w-other', channel_id: 'ch-2' },
+    ];
+    assert.deepEqual(reviewEventEvidence(windows, waveformEvent, selection), ['w-1', 'w-2', 'w-3']);
+    const decision = validateDecision({ channel, stream, segment, start: selection.start_sample, stop: selection.stop_sample,
+        action: 'exclude', response: 'confirmed', reason: 'jump and offset until recovery',
+        evidenceWindowIds: reviewEventEvidence(windows, waveformEvent, selection) });
+    assert.equal(decision.stop_sample, 360);
+    assert.deepEqual(decision.evidence_window_ids, ['w-1', 'w-2', 'w-3']);
+    assert.equal(decision.response, 'confirmed');
+    assert.deepEqual(reviewEventEvidence(windows, waveformEvent, { ...selection, stop_sample: 500 }), []);
+});
+
+test('unconfirmed ADC boundary evidence and legacy clipping keep their respective meanings', () => {
+    assert.deepEqual(boundaryDwellFeatures({ boundary_dwell_fraction: .1, longest_boundary_dwell_seconds: .2, adc_range_status: 'unconfirmed' }),
+        { fraction: .1, longest_seconds: .2, is_boundary_evidence: true });
+    assert.equal(boundaryDwellFeatures({ clipping_fraction: .2, longest_clipping_seconds: .3 }).is_boundary_evidence, false);
+    assert.equal(boundaryDwellFeatures({ boundary_dwell_fraction: null, clipping_fraction: .2 }).fraction, null);
+});
+
+test('stage overlays use the visible raw source clock and never annotate another channel or segment', () => {
+    const metadata = { stream_id: 's-1', segment_id: 'seg-2', channel_ids: ['ch-1'],
+        start_sample: 200, stop_sample: 340, recording_start_seconds: 35, sampling_frequency_hz: 10 };
+    assert.deepEqual(waveformStageBands([waveformEvent], metadata), [
+        { kind: 'plateau', start: 35, stop: 46 }, { kind: 'recovery', start: 46, stop: 49 },
+    ]);
+    assert.deepEqual(waveformStageBands([waveformEvent], { ...metadata, channel_ids: ['ch-2'] }), []);
+    assert.deepEqual(waveformStageBands([waveformEvent], { ...metadata, segment_id: 'seg-3' }), []);
+    assert.equal(waveformEvent.stop_sample, 360);
+});
+
+test('editing an event to a narrower valid interval retains candidate response evidence', () => {
+    const windows = [eventWindow, { ...eventWindow, window_id: 'w-2', start_sample: 200, stop_sample: 400 }];
+    const edited = { ...eventSelection(waveformEvent), ...sampleInputRange('190', '280') };
+    const evidenceWindowIds = reviewEventEvidence(windows, waveformEvent, edited);
+    const decision = validateDecision({ channel, stream, segment, start: edited.start_sample, stop: edited.stop_sample,
+        action: 'exclude', response: 'confirmed', reason: 'narrower artifact interval', evidenceWindowIds });
+    assert.deepEqual(decision.evidence_window_ids, ['w-1', 'w-2']);
+    assert.equal(decision.response, 'confirmed');
+    assert.deepEqual([decision.start_sample, decision.stop_sample], [190, 280]);
+    assert.equal(waveformEvent.start_sample, 180);
+    assert.equal(waveformEvent.stop_sample, 360);
+    assert.deepEqual(reviewEventEvidence(windows, waveformEvent, { ...edited, start_sample: 170 }), []);
+    assert.deepEqual(reviewEventEvidence(windows, waveformEvent, { ...edited, stop_sample: 370 }), []);
+});
+
+test('empty, fractional and reversed in-progress inputs neither throw in rendering nor bind evidence', () => {
+    for (const [start, stop] of [['', '280'], ['190', ''], ['300', '200'], ['-1', '280'], ['190.5', '280'], ['NaN', '280']]) {
+        assert.equal(sampleInputRange(start, stop), null);
+        assert.doesNotThrow(() => editableEventViewport(start, stop));
+        assert.equal(editableEventViewport(start, stop), null);
+        const selection = { ...eventSelection(waveformEvent), start_sample: start, stop_sample: stop };
+        assert.equal(reviewWindowEvidence(eventWindow, selection), undefined);
+        assert.deepEqual(reviewEventEvidence([eventWindow], waveformEvent, selection), []);
+    }
+    assert.deepEqual(editableEventViewport('190', '280'), { start_sample: 190, stop_sample: 280, page: 0, pages: 1 });
+});
diff --git a/frontend/src/core/api.js b/frontend/src/core/api.js
index b15fe4f7b305e798365d4c5d9cbb8deedfeebecc..3cf1ad0341a9fcb4e49dc58a1360ede6cfb56fd9 100644
--- a/frontend/src/core/api.js
+++ b/frontend/src/core/api.js
@@ -4,2 +4,3 @@ import { httpRequest } from './http/httpClient.js';
 import { demoResolve } from './http/demoAdapter.js';
+import { decodeEphysSamples, decodeEphysOverview, prepareEphysSampleRequest } from './ephysBinary.js';

@@ -566,2 +567,84 @@ export class ApiClient {

+    // =========================================================================
+    // Immutable electrophysiology input API (independent of legacy playback).
+    // =========================================================================
+
+    getEphysRecordings(subjectId, options = {}) {
+        return this.request(`/api/v1/ephys/recordings?${new URLSearchParams({ subject_id: subjectId })}`, options);
+    }
+
+    importEphysRecording(metadata, files, options = {}) {
+        const body = new FormData();
+        body.append('metadata', JSON.stringify(metadata));
+        for (const file of files) body.append('files', file, file.name);
+        return this.request('/api/v1/ephys/recordings', { ...options, method: 'POST', body });
+    }
+
+    getEphysChannelMap(recordingId, options = {}) {
+        return this.request(`/api/v1/ephys/recordings/${encodeURIComponent(recordingId)}/qc/channel-map`, options);
+    }
+
+    getEphysQcHistory(recordingId, options = {}) {
+        return this.request(`/api/v1/ephys/recordings/${encodeURIComponent(recordingId)}/qc`, options);
+    }
+
+    startEphysQc(recordingId, payload, options = {}) {
+        return this.request(`/api/v1/ephys/recordings/${encodeURIComponent(recordingId)}/qc/runs`, {
+            ...options, method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(payload),
+        });
+    }
+
+    getEphysQcRun(runId, options = {}) {
+        return this.request(`/api/v1/ephys/qc/runs/${encodeURIComponent(runId)}`, options);
+    }
+
+    saveEphysQcReview(runId, payload, options = {}) {
+        return this.request(`/api/v1/ephys/qc/runs/${encodeURIComponent(runId)}/review`, {
+            ...options, method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify(payload),
+        });
+    }
+
+    freezeEphysQcRun(runId, payload, options = {}) {
+        return this.request(`/api/v1/ephys/qc/runs/${encodeURIComponent(runId)}/freeze`, {
+            ...options, method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(payload),
+        });
+    }
+
+    exportEphysQcRun(runId, options = {}) {
+        return this.request(`/api/v1/ephys/qc/runs/${encodeURIComponent(runId)}/export`, {
+            ...options, responseType: 'blob', fastPath: true,
+        });
+    }
+
+    getEphysSignals(artifactId, options = {}) {
+        return this.request(`/api/v1/ephys/artifacts/${encodeURIComponent(artifactId)}/signals`, {
+            ...options, method: 'GET', responseType: 'json', fastPath: false,
+        });
+    }
+
+    getEphysEvents(artifactId, { offset = 0, limit = 100 } = {}, options = {}) {
+        const query = new URLSearchParams({ offset, limit });
+        return this.request(`/api/v1/ephys/artifacts/${encodeURIComponent(artifactId)}/events?${query}`, {
+            ...options, method: 'GET', responseType: 'json', fastPath: false,
+        });
+    }
+
+    /** Explicit channel IDs and half-open samples; derived inputs require data_kind: 'derived_samples'. */
+    async getEphysSamples(artifactId, request, options = {}) {
+        const { query, expected } = prepareEphysSampleRequest(artifactId, request);
+        const binary = await this.request(`/api/v1/ephys/artifacts/${encodeURIComponent(artifactId)}/samples?${query}`, {
+            ...options, method: 'GET', responseType: 'arrayBuffer', fastPath: true,
+        });
+        return decodeEphysSamples(binary, expected);
+    }
+
+    /** Unnormalized min/max bins; never interchangeable with raw samples. */
+    async getEphysOverview(artifactId, request, options = {}) {
+        const { query, expected } = prepareEphysSampleRequest(artifactId, request, true);
+        const binary = await this.request(`/api/v1/ephys/artifacts/${encodeURIComponent(artifactId)}/overview?${query}`, {
+            ...options, method: 'GET', responseType: 'arrayBuffer', fastPath: true,
+        });
+        return decodeEphysOverview(binary, expected);
+    }
+
     // =========================================================================
diff --git a/frontend/src/core/ephysBinary.js b/frontend/src/core/ephysBinary.js
new file mode 100644
index 0000000000000000000000000000000000000000..f183697080b467d45d86faa6544c67a74e98d0fc
--- /dev/null
+++ b/frontend/src/core/ephysBinary.js
@@ -0,0 +1,160 @@
+import { ApiError } from './http/apiError.js';
+
+const MAX_HEADER_BYTES = 4 * 1024 * 1024;
+const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
+const IDENTITY_FIELDS = ['artifact_id', 'stream_id', 'segment_id', 'start_sample', 'stop_sample', 'representation'];
+const OPTIONAL_IDENTITY_FIELDS = ['recording_id', 'content_sha256', 'reader_version', 'clock_id', 'source_sha256', 'source_relative_path', 'sample_index_origin', 'parent_artifact_id', 'parent_content_sha256'];
+
+function requireContract(condition, message, code = 'invalid_ephys_binary') {
+    if (!condition) throw new ApiError({ code, message: `Ephys: ${message}` });
+}
+
+function isId(value) {
+    return typeof value === 'string' && value.trim().length > 0;
+}
+
+function isSample(value) {
+    return Number.isSafeInteger(value) && value >= 0;
+}
+
+function validateWindow(request, code) {
+    for (const key of ['artifact_id', 'stream_id', 'segment_id']) {
+        requireContract(isId(request[key]), `missing ${key}`, code);
+    }
+    requireContract(Array.isArray(request.channel_ids) && request.channel_ids.length > 0
+        && request.channel_ids.every(isId)
+        && new Set(request.channel_ids).size === request.channel_ids.length, 'invalid channel_ids', code);
+    requireContract(isSample(request.start_sample) && isSample(request.stop_sample)
+        && request.stop_sample > request.start_sample, 'invalid half-open sample window', code);
+    requireContract(['physical', 'digital'].includes(request.representation), 'representation must be explicit', code);
+}
+
+/** Serialize a source-sample window without guessing channels, units or bounds. */
+export function prepareEphysSampleRequest(artifactId, request, overview = false) {
+    requireContract(request && typeof request === 'object', 'missing sample request', 'invalid_ephys_request');
+    const expected = { ...request, artifact_id: artifactId, channel_ids: request.channel_ids?.slice(), data_kind: request.data_kind ?? 'raw_samples' };
+    validateWindow(expected, 'invalid_ephys_request');
+    requireContract(['raw_samples', 'derived_samples'].includes(expected.data_kind), 'unsupported data_kind', 'invalid_ephys_request');
+    if (expected.data_kind === 'derived_samples') requireContract(expected.representation === 'physical', 'derived samples require physical representation', 'invalid_ephys_request');
+    const query = new URLSearchParams();
+    for (const key of IDENTITY_FIELDS.filter(key => key !== 'artifact_id')) query.set(key, String(expected[key]));
+    for (const channelId of expected.channel_ids) query.append('channel_id', channelId);
+    if (overview) {
+        requireContract(Number.isSafeInteger(expected.bin_samples) && expected.bin_samples > 0, 'invalid bin_samples', 'invalid_ephys_request');
+        query.set('bin_samples', String(expected.bin_samples));
+    }
+    return { query, expected };
+}
+
+function parseFrame(input) {
+    const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
+        : input instanceof Uint8Array ? input : null;
+    requireContract(bytes && bytes.byteLength >= 8, 'truncated binary frame');
+    requireContract(bytes[0] === 69 && bytes[1] === 80 && bytes[2] === 72 && bytes[3] === 89, 'invalid EPHY magic');
+    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
+    const headerLength = view.getUint32(4, true);
+    requireContract(headerLength > 0 && headerLength <= MAX_HEADER_BYTES
+        && headerLength <= bytes.byteLength - 8, 'invalid header length');
+    let metadata;
+    try {
+        metadata = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(8, 8 + headerLength)));
+    } catch {
+        requireContract(false, 'invalid UTF-8/JSON header');
+    }
+    requireContract(metadata && typeof metadata === 'object' && !Array.isArray(metadata), 'invalid header object');
+    const offset = Math.ceil((8 + headerLength) / 8) * 8;
+    requireContract(offset <= bytes.byteLength, 'truncated alignment padding');
+    for (let index = 8 + headerLength; index < offset; index++) requireContract(bytes[index] === 0, 'nonzero alignment padding');
+    return { bytes, view, metadata, offset };
+}
+
+function validateMetadata(metadata, expected, kind) {
+    requireContract(metadata.schema_version === 1, 'unsupported schema_version');
+    validateWindow(metadata, 'invalid_ephys_binary');
+    for (const key of ['recording_id', 'reader_version', 'clock_id']) requireContract(isId(metadata[key]), `missing ${key}`);
+    requireContract(typeof metadata.content_sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.content_sha256), 'invalid content_sha256');
+    requireContract(typeof metadata.source_sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.source_sha256), 'invalid source_sha256');
+    requireContract(isId(metadata.source_relative_path), 'missing source_relative_path');
+    requireContract(metadata.interval_convention === 'half_open', 'unsupported interval_convention');
+    requireContract(isSample(metadata.sample_index_origin) && metadata.sample_index_origin <= metadata.start_sample, 'invalid sample_index_origin');
+    requireContract(metadata.dtype === '<f8' && metadata.order === 'C', 'unsupported dtype/order');
+    requireContract(metadata.data_kind === kind, `expected ${kind}`);
+    requireContract(Array.isArray(metadata.units) && metadata.units.length === metadata.channel_ids.length
+        && metadata.units.every(unit => unit === null || typeof unit === 'string'), 'invalid units');
+    if (metadata.representation === 'digital') requireContract(metadata.units.every(unit => unit === 'count'), 'digital units must be count');
+    requireContract(Number.isFinite(metadata.sampling_frequency_hz) && metadata.sampling_frequency_hz > 0, 'invalid sampling_frequency_hz');
+    requireContract(Number.isFinite(metadata.recording_start_seconds), 'invalid recording_start_seconds');
+    requireContract(metadata.clock_alignment === 'source_local_unaligned', 'unsupported clock_alignment');
+    requireContract(expected && typeof expected === 'object', 'missing expected identity');
+    const sourceKind = expected.data_kind ?? 'raw_samples';
+    requireContract(['raw_samples', 'derived_samples'].includes(sourceKind), 'unsupported expected data_kind');
+    if (kind === 'minmax_overview') requireContract(metadata.source_data_kind === sourceKind, 'overview source_data_kind differs from request', 'ephys_identity_mismatch');
+    requireContract(Array.isArray(metadata.processing), 'missing processing history');
+    if (sourceKind === 'raw_samples') requireContract(metadata.processing.length === 0, 'raw-source response contains processing');
+    else {
+        requireContract(metadata.representation === 'physical' && metadata.units.every(unit => unit === 'V'), 'derived samples require physical volts');
+        requireContract(metadata.processing.length > 0, 'derived samples require processing history');
+        requireContract(isId(metadata.parent_artifact_id) && typeof metadata.parent_content_sha256 === 'string'
+            && /^[a-f0-9]{64}$/.test(metadata.parent_content_sha256), 'invalid derived parent provenance');
+        requireContract(metadata.software_versions && typeof metadata.software_versions === 'object'
+            && !Array.isArray(metadata.software_versions) && Object.keys(metadata.software_versions).length > 0
+            && Object.entries(metadata.software_versions).every(([name, version]) => isId(name) && isId(version)), 'invalid derived software_versions');
+    }
+    for (const key of IDENTITY_FIELDS) {
+        requireContract(metadata[key] === expected[key], `response ${key} differs from request`, 'ephys_identity_mismatch');
+    }
+    requireContract(Array.isArray(expected.channel_ids) && expected.channel_ids.length === metadata.channel_ids.length
+        && metadata.channel_ids.every((id, index) => id === expected.channel_ids[index]), 'response channel order differs from request', 'ephys_identity_mismatch');
+    for (const key of OPTIONAL_IDENTITY_FIELDS) {
+        if (expected[key] !== undefined) requireContract(metadata[key] === expected[key], `response ${key} differs from catalog`, 'ephys_identity_mismatch');
+    }
+}
+
+function decodeValues(frame, shape) {
+    const { bytes, view, metadata, offset } = frame;
+    requireContract(Array.isArray(metadata.shape) && metadata.shape.length === shape.length
+        && metadata.shape.every((size, index) => Number.isSafeInteger(size) && size > 0 && size === shape[index]), 'invalid shape');
+    const count = shape.reduce((total, size) => total * size, 1);
+    requireContract(Number.isSafeInteger(count * 8) && bytes.byteLength - offset === count * 8, 'payload length differs from shape');
+    // Network ArrayBuffers are aligned, so the common path does not copy samples.
+    // DataView keeps little-endian decoding correct on other hosts/views too.
+    let samples;
+    if (LITTLE_ENDIAN && (bytes.byteOffset + offset) % 8 === 0) samples = new Float64Array(bytes.buffer, bytes.byteOffset + offset, count);
+    else {
+        samples = new Float64Array(count);
+        for (let index = 0; index < count; index++) samples[index] = view.getFloat64(offset + index * 8, true);
+    }
+    return { metadata, samples };
+}
+
+/**
+ * Returns channel-major Float64Array samples plus their exact metadata.
+ * recording_start_seconds is window time in clock_id's local source clock;
+ * it does not assert alignment between separate files or streams.
+ */
+export function decodeEphysSamples(input, expected) {
+    const frame = parseFrame(input);
+    validateMetadata(frame.metadata, expected, expected?.data_kind ?? 'raw_samples');
+    return decodeValues(frame, [frame.metadata.channel_ids.length, frame.metadata.stop_sample - frame.metadata.start_sample]);
+}
+
+/** Min/max bins with explicit raw/derived source kind, separate from exact samples. */
+export function decodeEphysOverview(input, expected) {
+    const frame = parseFrame(input);
+    const metadata = frame.metadata;
+    validateMetadata(metadata, expected, 'minmax_overview');
+    requireContract(Number.isSafeInteger(metadata.bin_samples) && metadata.bin_samples > 0, 'invalid bin_samples');
+    requireContract(metadata.bin_samples === expected.bin_samples, 'response bin_samples differs from request', 'ephys_identity_mismatch');
+    requireContract(Array.isArray(metadata.axes) && metadata.axes.length === 3
+        && metadata.axes.every((axis, index) => axis === ['channel', 'bin', 'extremum'][index]), 'invalid overview axes');
+    requireContract(Array.isArray(metadata.extrema) && metadata.extrema.length === 2
+        && metadata.extrema.every((extremum, index) => extremum === ['min', 'max'][index]), 'invalid overview extrema');
+    const bins = Math.ceil((metadata.stop_sample - metadata.start_sample) / metadata.bin_samples);
+    for (const key of ['bin_start_samples', 'bin_stop_samples']) requireContract(Array.isArray(metadata[key]) && metadata[key].length === bins, `invalid ${key}`);
+    for (let index = 0; index < bins; index++) {
+        const start = metadata.start_sample + index * metadata.bin_samples;
+        requireContract(metadata.bin_start_samples[index] === start
+            && metadata.bin_stop_samples[index] === Math.min(start + metadata.bin_samples, metadata.stop_sample), 'overview bin sample bounds mismatch');
+    }
+    return decodeValues(frame, [metadata.channel_ids.length, bins, 2]);
+}
diff --git a/frontend/src/core/i18n/locales/en.js b/frontend/src/core/i18n/locales/en.js
index 33ffa271c3fe945545a0c82864dcc8ecbcbf04f8..651398e357edf3a68a2c643b8e0ee9a076dcc400 100644
--- a/frontend/src/core/i18n/locales/en.js
+++ b/frontend/src/core/i18n/locales/en.js
@@ -1,2 +1,177 @@
 export const en = {
+    "QC selection outside candidate": "Selection is outside this candidate; adjust it or use a manual decision",
+    "QC waveform events": "Waveform event candidates",
+    "QC evidence without automatic penalty": "Review evidence · no automatic penalty",
+    "QC view complete event": "View complete event",
+    "QC event recovered": "Recovery confirmed",
+    "QC event recovery unconfirmed": "Recovery not confirmed by this point",
+    "QC event stage jump": "Jump",
+    "QC event stage drift": "Drift",
+    "QC event stage plateau": "Plateau",
+    "QC event stage recovery": "Recovery",
+    "QC event reason sustained_level_shift": "Sustained level shift",
+    "QC event reason linear_drift_with_plateau": "Drift followed by plateau",
+    "QC event reason sustained_linear_drift": "Sustained drift",
+    "QC event reason low_variability_plateau": "Low-variability plateau",
+    "QC event reason baseline_recovery_confirmed": "Baseline recovery confirmed",
+    "QC event reason recovery_not_observed": "Recovery not observed",
+    "QC event end recovered": "Recovered",
+    "QC event end segment_end": "End of continuous segment",
+    "QC event end range_end": "End of scan range",
+    "QC event end nonfinite_gap": "Nonfinite sample gap",
+    "QC waveform view": "Waveform view",
+    "QC previous waveform page": "Previous waveform page",
+    "QC next waveform page": "Next waveform page",
+    "QC invalid event range": "Invalid event interval",
+    "QC invalid event page": "Invalid waveform page",
+    "QC boundary dwell share": "File-boundary dwell share",
+    "QC longest boundary dwell": "Longest boundary dwell",
+    "QC longest clipping": "Longest clipping",
+    "QC flag waveform_anomaly": "Waveform anomaly candidate",
+    "QC flag boundary_dwell": "File-boundary dwell candidate",
+    "QC next scan settings": "Settings for the next scan",
+    "QC actual scan ranges": "Actual ranges in this run",
+    "QC next issue": "next issue",
+    "QC assessment pending": "assessment pending",
+    "QC assessment degraded": "assessment degraded",
+    "QC assessment reviewed": "assessment reviewed",
+    "QC issue": "issue",
+    "QC penalty": "penalty",
+    "QC windows": "windows",
+    "QC channel seconds": "channel seconds",
+    "QC mapping pending": "mapping pending",
+    "QC unknown units": "unknown units",
+    "QC gaps": "gaps",
+    "QC scanned coverage": "scanned coverage",
+    "QC evidence coverage": "evidence coverage",
+    "QC source": "source",
+    "QC sample rate": "sample rate",
+    "QC channels": "channels",
+    "QC segments": "segments",
+    "QC channel relations": "channel relations",
+    "QC pair": "pair",
+    "QC relation": "relation",
+    "QC correlation": "correlation",
+    "QC samples": "samples",
+    "QC total seconds": "total seconds",
+    "QC kept seconds": "kept seconds",
+    "QC excluded seconds": "excluded seconds",
+    "QC unknown seconds": "unknown seconds",
+    "QC common usable time": "common usable time",
+    "QC flag nonfinite": "flag nonfinite",
+    "QC flag flat": "flag flat",
+    "QC flag clipping": "flag clipping",
+    "QC flag line_noise": "flag line_noise",
+    "QC relation duplicate_signal_candidate": "relation duplicate_signal_candidate",
+    "QC relation high_correlation_candidate": "relation high_correlation_candidate",
+    "QC relation same_shaft_correlation_evidence": "relation same_shaft_correlation_evidence",
+    "QC RMS": "RMS",
+    "QC MAD": "MAD",
+    "QC amplitude range": "amplitude range",
+    "QC offset": "offset",
+    "QC drift": "drift",
+    "QC largest jump": "largest jump",
+    "QC missing samples": "missing samples",
+    "QC flat share": "flat share",
+    "QC longest flat": "longest flat",
+    "QC clipping share": "clipping share",
+    "QC line prominence": "line prominence",
+    "QC line power share": "line power share",
+    "QC high frequency share": "high frequency share",
+    "QC kurtosis": "kurtosis",
+    "QC mapping applies to next scan": "Mapping changes apply to the next scan",
+    "QC retained share": "Retained share of reviewed time",
+    "SEEG QC": "SEEG QC",
+    "QC recording": "recording",
+    "QC refresh": "refresh",
+    "QC import EDF/BDF": "import EDF/BDF",
+    "QC run": "run",
+    "QC channel mapping": "channel mapping",
+    "QC original name": "original name",
+    "QC type": "type",
+    "QC shaft": "shaft",
+    "QC contact": "contact",
+    "QC measurement": "measurement",
+    "QC endpoints": "endpoints",
+    "QC reference": "reference",
+    "QC confirmed": "confirmed",
+    "QC line frequency": "line frequency",
+    "QC window seconds": "window seconds",
+    "QC scan start": "scan start",
+    "QC scan stop": "scan stop",
+    "QC start scan": "start scan",
+    "QC channels and issues": "channels and issues",
+    "QC segment": "segment",
+    "QC start sample": "start sample",
+    "QC stop sample": "stop sample",
+    "QC load window": "load window",
+    "QC raw waveform": "raw waveform",
+    "QC scan PSD": "scan PSD",
+    "QC time windows": "time windows",
+    "QC previous": "previous",
+    "QC next": "next",
+    "QC review": "review",
+    "QC reviewer": "reviewer",
+    "QC reason": "reason",
+    "QC confirm issue": "confirm issue",
+    "QC false alarm": "false alarm",
+    "QC pending": "pending",
+    "QC manual exclude": "manual exclude",
+    "QC manual keep": "manual keep",
+    "QC report details": "report details",
+    "QC save review": "save review",
+    "QC reload saved": "reload saved",
+    "QC freeze": "freeze",
+    "QC download ZIP": "download ZIP",
+    "QC select channel and segment": "select channel and segment",
+    "QC invalid sample range": "invalid sample range",
+    "QC invalid action": "invalid action",
+    "QC reason required": "reason required",
+    "QC waveform range too large": "waveform range too large",
+    "QC invalid scan range": "invalid scan range",
+    "QC empty scan range": "empty scan range",
+    "QC not evaluated": "not evaluated",
+    "QC select window": "select window",
+    "QC unknown unit": "unknown unit",
+    "QC source time seconds": "source time seconds",
+    "QC ungrouped": "ungrouped",
+    "QC initial score": "initial score",
+    "QC reviewed score": "reviewed score",
+    "QC reviewed coverage": "reviewed coverage",
+    "QC excluded time": "excluded time",
+    "QC unknown time": "unknown time",
+    "QC score components": "score components",
+    "QC input and reference": "input and reference",
+    "QC final channel summary": "final channel summary",
+    "QC raw": "raw",
+    "QC no automatic flags": "no automatic flags",
+    "QC no scan yet": "no scan yet",
+    "QC action keep": "action keep",
+    "QC action exclude": "action exclude",
+    "QC action pending": "action pending",
+    "QC whole channel": "whole channel",
+    "QC remove decision": "remove decision",
+    "QC unsaved changes": "unsaved changes",
+    "QC frozen": "frozen",
+    "QC saved": "saved",
+    "QC status pending": "status pending",
+    "QC status running": "status running",
+    "QC status succeeded": "status succeeded",
+    "QC status failed": "status failed",
+    "QC status cancelled": "status cancelled",
+    "QC request failed": "request failed",
+    "QC select subject": "select subject",
+    "QC draft revision conflict": "draft revision conflict",
+    "QC choose recording": "choose recording",
+    "QC import a recording": "import a recording",
+    "QC new scan": "new scan",
+    "QC discard local draft?": "discard local draft?",
+    "QC importing": "importing",
+    "QC scanning": "scanning",
+    "QC scan ready": "scan ready",
+    "QC scan failed": "scan failed",
+    "QC loading samples": "loading samples",
+    "QC samples unavailable": "samples unavailable",
+    "QC reviewer required": "reviewer required",
+    "QC review saved": "review saved",
     "SUBJECT NAVIGATOR": "SUBJECT NAVIGATOR",
diff --git a/frontend/src/core/i18n/locales/zh.js b/frontend/src/core/i18n/locales/zh.js
index dc853e8b05dfaada816d84508954f21e53712865..6fc8f538ad8811cb9983e968ff5dde6d20a1a09b 100644
--- a/frontend/src/core/i18n/locales/zh.js
+++ b/frontend/src/core/i18n/locales/zh.js
@@ -1,2 +1,177 @@
 export const zh = {
+    "QC selection outside candidate": "所选范围超出当前候选，请调整范围或使用手工决定",
+    "QC waveform events": "联合波形候选",
+    "QC evidence without automatic penalty": "复核证据 · 不自动扣分",
+    "QC view complete event": "查看完整事件",
+    "QC event recovered": "已确认恢复",
+    "QC event recovery unconfirmed": "截至此处仍未确认恢复",
+    "QC event stage jump": "跳变",
+    "QC event stage drift": "漂移",
+    "QC event stage plateau": "平台",
+    "QC event stage recovery": "恢复",
+    "QC event reason sustained_level_shift": "持续基线偏移",
+    "QC event reason linear_drift_with_plateau": "漂移后平台",
+    "QC event reason sustained_linear_drift": "持续漂移",
+    "QC event reason low_variability_plateau": "低波动平台",
+    "QC event reason baseline_recovery_confirmed": "基线已恢复",
+    "QC event reason recovery_not_observed": "尚未观察到恢复",
+    "QC event end recovered": "已恢复",
+    "QC event end segment_end": "连续段结束",
+    "QC event end range_end": "扫描范围结束",
+    "QC event end nonfinite_gap": "遇到缺失样本",
+    "QC waveform view": "波形视口",
+    "QC previous waveform page": "波形前段",
+    "QC next waveform page": "波形后段",
+    "QC invalid event range": "事件范围无效",
+    "QC invalid event page": "波形页码无效",
+    "QC boundary dwell share": "文件边界停留占比",
+    "QC longest boundary dwell": "最长边界停留",
+    "QC longest clipping": "最长削顶",
+    "QC flag waveform_anomaly": "联合波形异常候选",
+    "QC flag boundary_dwell": "文件边界停留候选",
+    "QC next scan settings": "下一次扫描配置",
+    "QC actual scan ranges": "当前版本实际扫描范围",
+    "QC next issue": "下一候选",
+    "QC assessment pending": "待评价",
+    "QC assessment degraded": "有限可用",
+    "QC assessment reviewed": "已复核",
+    "QC issue": "问题",
+    "QC penalty": "单窗扣分上限",
+    "QC windows": "窗口数",
+    "QC channel seconds": "通道秒",
+    "QC mapping pending": "待确认映射",
+    "QC unknown units": "未知单位",
+    "QC gaps": "时钟缺口",
+    "QC scanned coverage": "自动扫描覆盖",
+    "QC evidence coverage": "证据裁决覆盖",
+    "QC source": "源文件",
+    "QC sample rate": "采样率",
+    "QC channels": "通道数",
+    "QC segments": "连续段数",
+    "QC channel relations": "当前通道的关联证据（前 60 项）",
+    "QC pair": "通道对",
+    "QC relation": "关系",
+    "QC correlation": "相关系数",
+    "QC samples": "样本范围",
+    "QC total seconds": "总时长 s",
+    "QC kept seconds": "保留 s",
+    "QC excluded seconds": "排除 s",
+    "QC unknown seconds": "待定 s",
+    "QC common usable time": "同一信号流全通道共同可用时长",
+    "QC flag nonfinite": "缺失/非有限值",
+    "QC flag flat": "平直",
+    "QC flag clipping": "削顶/饱和",
+    "QC flag line_noise": "工频",
+    "QC relation duplicate_signal_candidate": "重复信号候选",
+    "QC relation high_correlation_candidate": "高相关候选",
+    "QC relation same_shaft_correlation_evidence": "同杆相关证据",
+    "QC RMS": "RMS",
+    "QC MAD": "MAD",
+    "QC amplitude range": "稳健幅度范围",
+    "QC offset": "偏移",
+    "QC drift": "漂移",
+    "QC largest jump": "最大突跳",
+    "QC missing samples": "缺失比例",
+    "QC flat share": "平直比例",
+    "QC longest flat": "最长平直",
+    "QC clipping share": "削顶比例",
+    "QC line prominence": "工频突出度",
+    "QC line power share": "工频功率占比",
+    "QC high frequency share": "高频占比",
+    "QC kurtosis": "峰度",
+    "QC mapping applies to next scan": "映射修改将在下一次扫描生效",
+    "QC retained share": "已复核中的保留比例",
+    "SEEG QC": "SEEG QC",
+    "QC recording": "记录",
+    "QC refresh": "刷新",
+    "QC import EDF/BDF": "导入 EDF/BDF",
+    "QC run": "扫描版本",
+    "QC channel mapping": "通道映射",
+    "QC original name": "原始名称",
+    "QC type": "类型",
+    "QC shaft": "电极杆",
+    "QC contact": "触点号",
+    "QC measurement": "测量类型",
+    "QC endpoints": "触点（正端, 负端）",
+    "QC reference": "采集参考",
+    "QC confirmed": "已确认",
+    "QC line frequency": "工频 Hz",
+    "QC window seconds": "窗口 s",
+    "QC scan start": "起始 s",
+    "QC scan stop": "截止 s（留空至末尾）",
+    "QC start scan": "开始扫描",
+    "QC channels and issues": "通道与候选问题",
+    "QC segment": "连续段",
+    "QC start sample": "起始样本",
+    "QC stop sample": "截止样本",
+    "QC load window": "读取波形",
+    "QC raw waveform": "原始波形",
+    "QC scan PSD": "扫描范围频谱",
+    "QC time windows": "时间窗",
+    "QC previous": "上一页",
+    "QC next": "下一页",
+    "QC review": "人工复核",
+    "QC reviewer": "审核者",
+    "QC reason": "理由",
+    "QC confirm issue": "确认问题",
+    "QC false alarm": "误报",
+    "QC pending": "待定",
+    "QC manual exclude": "手工排除",
+    "QC manual keep": "手工保留",
+    "QC report details": "详细报告",
+    "QC save review": "保存复评",
+    "QC reload saved": "重载已保存版本",
+    "QC freeze": "冻结版本",
+    "QC download ZIP": "下载 ZIP",
+    "QC select channel and segment": "请选择通道和连续段",
+    "QC invalid sample range": "样本区间无效，请保持在当前连续段内",
+    "QC invalid action": "复核动作无效",
+    "QC reason required": "请填写判断理由",
+    "QC waveform range too large": "波形窗口最多读取 200000 个样本，请缩小范围",
+    "QC invalid scan range": "扫描时间范围无效",
+    "QC empty scan range": "扫描范围内没有样本",
+    "QC not evaluated": "未评估",
+    "QC select window": "选择通道或时间窗查看原始证据",
+    "QC unknown unit": "单位未知",
+    "QC source time seconds": "源文件时间 (s)",
+    "QC ungrouped": "未分组",
+    "QC initial score": "自动初评",
+    "QC reviewed score": "复评分",
+    "QC reviewed coverage": "人工复核覆盖",
+    "QC excluded time": "已排除时间",
+    "QC unknown time": "未定时间",
+    "QC score components": "评分分项",
+    "QC input and reference": "输入与参考",
+    "QC final channel summary": "逐通道最终统计",
+    "QC raw": "原始电压",
+    "QC no automatic flags": "未触发自动报警",
+    "QC no scan yet": "尚未扫描",
+    "QC action keep": "保留",
+    "QC action exclude": "排除",
+    "QC action pending": "待定",
+    "QC whole channel": "整条通道",
+    "QC remove decision": "移除决定",
+    "QC unsaved changes": "有未保存的决定",
+    "QC frozen": "已冻结",
+    "QC saved": "已保存",
+    "QC status pending": "排队中",
+    "QC status running": "扫描中",
+    "QC status succeeded": "扫描完成",
+    "QC status failed": "扫描失败",
+    "QC status cancelled": "已取消",
+    "QC request failed": "请求失败",
+    "QC select subject": "请先选择受试者",
+    "QC draft revision conflict": "本地草稿对应旧修订，请核对已保存版本",
+    "QC choose recording": "选择记录",
+    "QC import a recording": "导入 EDF/BDF 开始 QC",
+    "QC new scan": "新建扫描",
+    "QC discard local draft?": "丢弃当前未保存的决定并重载？",
+    "QC importing": "正在导入…",
+    "QC scanning": "正在提交扫描…",
+    "QC scan ready": "扫描完成，可以开始复核",
+    "QC scan failed": "扫描失败，请查看任务错误",
+    "QC loading samples": "正在读取原始样本…",
+    "QC samples unavailable": "原始样本读取失败",
+    "QC reviewer required": "请填写审核者",
+    "QC review saved": "复核决定与复评分已保存",
     "Status Unknown": "状态未知",
diff --git a/frontend/src/core/i18n/runtime.js b/frontend/src/core/i18n/runtime.js
index 6576f8b2ead67ef06768d7469b813d6b8695081b..5740a54360d6d3e674d40bd25b55c0b257bbb5f0 100644
--- a/frontend/src/core/i18n/runtime.js
+++ b/frontend/src/core/i18n/runtime.js
@@ -55,2 +55,3 @@ export class I18n {
         this.listeners.push(callback);
+        return () => { this.listeners = this.listeners.filter(listener => listener !== callback); };
     }
diff --git a/frontend/src/viewer/ephysQc/charts.js b/frontend/src/viewer/ephysQc/charts.js
new file mode 100644
index 0000000000000000000000000000000000000000..76ac20185746940ef2b3cc005408e8df4b247992
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/charts.js
@@ -0,0 +1,49 @@
+import { i18n } from '../../core/i18n.js';
+import { esc } from './template.js';
+import { numberText } from './state.js';
+import { waveformStageBands } from './waveformEvents.js';
+
+function graph(x, y, { xLabel, yLabel, color = '#38bdf8', regions = [] }) {
+    const pairs = x.map((value, index) => [value, y[index]]).filter(pair => pair.every(Number.isFinite));
+    if (!pairs.length) return esc(i18n.t('QC not evaluated'));
+    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
+    for (const [xv, yv] of pairs) { xMin = Math.min(xMin, xv); xMax = Math.max(xMax, xv); yMin = Math.min(yMin, yv); yMax = Math.max(yMax, yv); }
+    if (yMax === yMin) { const pad = Math.max(Math.abs(yMin) * .05, .1); yMin -= pad; yMax += pad; }
+    const xSpan = xMax - xMin || 1, ySpan = yMax - yMin;
+    let path = '', connected = false;
+    for (let index = 0; index < x.length; index++) {
+        if (!Number.isFinite(x[index]) || !Number.isFinite(y[index])) { connected = false; continue; }
+        path += `${connected ? 'L' : 'M'}${(52 + (x[index] - xMin) / xSpan * 720).toFixed(2)},${(18 + (yMax - y[index]) / ySpan * 132).toFixed(2)} `;
+        connected = true;
+    }
+    const colors = { jump: '#fbbf24', drift: '#a78bfa', plateau: '#fb923c', recovery: '#34d399' };
+    const bands = regions.map(region => {
+        const start = Math.max(xMin, region.start), stop = Math.min(xMax, region.stop);
+        if (!Number.isFinite(start) || !Number.isFinite(stop) || stop <= start || !colors[region.kind]) return '';
+        return `<rect x="${52 + (start - xMin) / xSpan * 720}" y="18" width="${(stop - start) / xSpan * 720}" height="132" fill="${colors[region.kind]}" opacity=".13"><title>${esc(i18n.t(`QC event stage ${region.kind}`))}</title></rect>`;
+    }).join('');
+    return `<svg viewBox="0 0 800 185" preserveAspectRatio="none" role="img" aria-label="${esc(yLabel)}">
+        ${bands}
+        <path d="M52 15V150H775" stroke="#64748b" fill="none"/>
+        <path d="${path}" stroke="${color}" stroke-width="1.2" fill="none" vector-effect="non-scaling-stroke"/>
+        <g fill="currentColor" font-size="11"><text x="4" y="23">${esc(numberText(yMax))}</text><text x="4" y="150">${esc(numberText(yMin))}</text>
+        <text x="52" y="168">${esc(numberText(xMin))}</text><text x="735" y="168">${esc(numberText(xMax))}</text>
+        <text x="70" y="13">${esc(yLabel)}</text><text x="395" y="180">${esc(xLabel)}</text></g></svg>`;
+}
+
+export function rawWaveform(frame, events = []) {
+    if (!frame) return esc(i18n.t('QC select window'));
+    const { metadata, samples } = frame, factor = metadata.units[0] === 'V' ? 1e6 : 1;
+    const unit = metadata.units[0] === 'V' ? 'µV' : metadata.units[0] || i18n.t('QC unknown unit');
+    const time = Array.from(samples, (_, index) => metadata.recording_start_seconds + index / metadata.sampling_frequency_hz);
+    return graph(time, Array.from(samples, value => value * factor), {
+        xLabel: i18n.t('QC source time seconds'), yLabel: unit, regions: waveformStageBands(events, metadata),
+    });
+}
+
+export function spectrumGraph(spectrum) {
+    if (spectrum?.status !== 'computed') return esc(i18n.t('QC not evaluated'));
+    return graph(spectrum.frequency_hz, spectrum.power_v2_hz.map(value => value > 0 ? Math.log10(value) : NaN), {
+        xLabel: 'Hz', yLabel: 'log₁₀(V²/Hz)', color: '#a78bfa',
+    });
+}
diff --git a/frontend/src/viewer/ephysQc/eventRender.js b/frontend/src/viewer/ephysQc/eventRender.js
new file mode 100644
index 0000000000000000000000000000000000000000..c6a343f285d28a4329fd19ad8d8d9c42000dfdf1
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/eventRender.js
@@ -0,0 +1,28 @@
+import { i18n } from '../../core/i18n.js';
+import { esc, label } from './template.js';
+import { numberText } from './state.js';
+import { associatedWaveformEvents, editableEventViewport } from './waveformEvents.js';
+
+export function visibleWaveformEvents(panel) {
+    const selection = panel.activeWindow || panel.activeEvent || { channel_id: panel.channelId,
+        stream_id: panel.catalog?.streams.find(stream => stream.channels.some(channel => channel.channel_id === panel.channelId))?.stream_id,
+        segment_id: panel.el.segment.value, start_sample: Number(panel.el.start.value), stop_sample: Number(panel.el.stop.value) };
+    return associatedWaveformEvents(panel.run?.report, selection);
+}
+
+export function renderWaveformEvents(panel) {
+    const events = visibleWaveformEvents(panel);
+    panel.el['waveform-events'].hidden = !events.length;
+    panel.el['waveform-events'].innerHTML = events.length ? `<h3>${label('QC waveform events')} <small>${label('QC evidence without automatic penalty')}</small></h3>`
+        + events.map(event => `<article class="qc-event-card ${panel.activeEvent?.event_id === event.event_id ? 'selected' : ''}">
+            <div class="qc-event-title"><button type="button" data-event="${esc(event.event_id)}">${label('QC view complete event')} · ${numberText(event.recording_start_seconds, 3)}–${numberText(event.recording_stop_seconds, 3)} s</button>
+                <span>${label(event.recovery_status === 'confirmed' ? 'QC event recovered' : 'QC event recovery unconfirmed')}</span></div>
+            <div class="qc-muted">[${event.start_sample}, ${event.stop_sample}) · ${esc((event.reasons || []).filter(reason => !['baseline_recovery_confirmed', 'recovery_not_observed'].includes(reason)).map(reason => i18n.t(`QC event reason ${reason}`)).join(' · '))}${event.recovery_status === 'right_censored' ? ` · ${esc(i18n.t(`QC event end ${event.termination_reason}`))}` : ''}</div>
+            <div class="qc-event-stages">${(event.stages || []).map((stage, index) => `<button type="button" data-event="${esc(event.event_id)}" data-stage="${index}" class="qc-stage-${esc(stage.kind)}" title="[${stage.start_sample}, ${stage.stop_sample})">${label(`QC event stage ${stage.kind}`)} · ${numberText(stage.recording_start_seconds, 3)}–${numberText(stage.recording_stop_seconds, 3)} s</button>`).join('')}</div>
+        </article>`).join('') : '';
+    const viewport = panel.activeEvent ? editableEventViewport(panel.el.start.value, panel.el.stop.value, panel.wavePage || 0) : null;
+    panel.el['wave-pages'].hidden = !viewport || viewport.pages <= 1;
+    panel.el['wave-previous'].disabled = !viewport || viewport.page === 0;
+    panel.el['wave-next'].disabled = !viewport || viewport.page + 1 >= viewport.pages;
+    panel.el['wave-page'].textContent = viewport ? `${i18n.t('QC waveform view')} ${viewport.page + 1}/${viewport.pages} · [${viewport.start_sample}, ${viewport.stop_sample})` : '';
+}
diff --git a/frontend/src/viewer/ephysQc/io.js b/frontend/src/viewer/ephysQc/io.js
new file mode 100644
index 0000000000000000000000000000000000000000..8d8cb7098f5f5b6f1a40832689f3d359316008cb
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/io.js
@@ -0,0 +1,171 @@
+import { apiClient } from '../../core/api.js';
+import { i18n } from '../../core/i18n.js';
+import { channelMapPayload, windowRequest, scanRanges } from './state.js';
+import { renderQc, renderMap, renderEvidence, renderReview } from './render.js';
+import { eventViewport } from './waveformEvents.js';
+
+export const qcIoMethods = {
+    trackMutation(promise) {
+        this.pendingMutation = promise;
+        promise.finally(() => { if (this.pendingMutation === promise) this.pendingMutation = null; }).catch(() => {});
+        return promise;
+    },
+
+    async loadRecords(preferredId) {
+        this.cacheDraft();
+        const token = this.recordGeneration.next(); this.runGeneration.next(); this.windowGeneration.next();
+        clearTimeout(this.pollTimer); this.status(); this.busy = true; renderReview(this);
+        if (!this.subjectId) { this.busy = false; this.status('QC select subject'); return; }
+        try {
+            const rows = await apiClient.getEphysRecordings(this.subjectId);
+            if (!this.contextCurrent(token)) return;
+            this.records = rows;
+            this.el.recording.replaceChildren(new Option(i18n.t('QC choose recording'), ''), ...rows.map(row => new Option(row.label, row.recording_id)));
+            const id = preferredId || (rows.some(row => row.recording_id === this.recordingId) ? this.recordingId : rows[0]?.recording_id);
+            this.busy = false;
+            if (id) await this.selectRecording(id);
+            else { this.catalog = null; this.run = null; this.mapRows = []; this.decisions = []; this.dirty = false; renderMap(this); renderQc(this); this.status('QC import a recording'); }
+        } catch (error) { if (this.contextCurrent(token)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    async selectRecording(recordingId) {
+        this.cacheDraft();
+        const token = this.recordGeneration.next(); this.runGeneration.next(); this.windowGeneration.next();
+        clearTimeout(this.pollTimer); this.recordingId = recordingId; this.el.recording.value = recordingId;
+        this.catalog = null; this.run = null; this.frame = null; this.activeWindow = null; this.activeEvent = null; this.wavePage = 0; this.channelId = null;
+        this.decisions = []; this.mapRows = []; this.dirty = false; this.mapDirty = false; this.busy = true;
+        this.el.history.replaceChildren(); this.el.segment.replaceChildren(); this.el['catalog-summary'].textContent = '';
+        this.status(); renderMap(this); renderQc(this);
+        const recording = this.records.find(row => row.recording_id === recordingId);
+        if (!recording) { this.busy = false; renderReview(this); return; }
+        try {
+            const [catalog, history, mapping] = await Promise.all([
+                apiClient.getEphysSignals(recording.input_artifact_id), apiClient.getEphysQcHistory(recordingId), apiClient.getEphysChannelMap(recordingId),
+            ]);
+            if (!this.contextCurrent(token)) return;
+            this.catalog = catalog; this.mapRows = mapping.channel_map;
+            this.el['catalog-summary'].textContent = `${catalog.streams.length} streams · ${this.mapRows.length} ch · ${catalog.status}`;
+            this.setHistory(history); renderMap(this); this.busy = false;
+            if (history.current) this.applyRun(history.current);
+            const firstId = this.mapRows.find(row => (row.channel_type === 'seeg' || row.candidate?.channel_type === 'seeg'))?.channel_id || this.mapRows[0]?.channel_id;
+            if (firstId) this.selectChannel(firstId); else renderQc(this);
+            if (['pending', 'running'].includes(this.run?.status)) this.pollRun(this.run.run_id);
+        } catch (error) { if (this.contextCurrent(token)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    setHistory(history) {
+        this.el.history.replaceChildren(new Option(i18n.t('QC new scan'), ''), ...history.runs.map(run => new Option(`${run.created_at} · ${run.status}`, run.run_id)));
+        this.el.history.value = history.current?.run_id || '';
+    },
+
+    async selectRun(runId, discardDraft = false) {
+        if (!runId) return;
+        if (discardDraft && this.dirty && !window.confirm(i18n.t('QC discard local draft?'))) return;
+        if (!discardDraft) this.cacheDraft(); else this.drafts.delete(runId);
+        const token = this.runGeneration.next(); this.windowGeneration.next(); clearTimeout(this.pollTimer);
+        this.busy = true; this.status(); this.frame = null; this.activeWindow = null; this.activeEvent = null; this.wavePage = 0; renderReview(this);
+        try {
+            const run = await apiClient.getEphysQcRun(runId);
+            if (!this.contextCurrent(token, this.runGeneration)) return;
+            this.busy = false; this.applyRun(run, !discardDraft); this.el.history.value = runId;
+            if (this.channelId) this.loadWindow();
+            if (['pending', 'running'].includes(run.status)) this.pollRun(runId);
+        } catch (error) { if (this.contextCurrent(token, this.runGeneration)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    async importFiles() {
+        const files = [...this.el.files.files]; if (!files.length || !this.subjectId) return;
+        const token = this.recordGeneration.value; this.busy = true; this.status('QC importing'); renderReview(this);
+        try {
+            const record = await this.trackMutation(apiClient.importEphysRecording({ subject_id: this.subjectId, label: files[0].name,
+                modality: 'seeg', metadata: {}, import_key: crypto.randomUUID() }, files));
+            if (!this.contextCurrent(token)) return;
+            this.busy = false; this.el.files.value = ''; await this.loadRecords(record.recording_id);
+        } catch (error) { if (this.contextCurrent(token)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    async startScan() {
+        if (!this.catalog || this.busy || this.dirty) return;
+        const token = this.runGeneration.next(); this.windowGeneration.next(); clearTimeout(this.pollTimer);
+        try {
+            const config = { line_frequency: Number(this.el.line.value), window_seconds: Number(this.el.window.value) };
+            const ranges = scanRanges(this.catalog, Number(this.el['scan-start'].value), this.el['scan-stop'].value === '' ? null : Number(this.el['scan-stop'].value));
+            if (ranges) config.ranges = ranges;
+            this.busy = true; this.status('QC scanning'); renderReview(this);
+            const run = await this.trackMutation(apiClient.startEphysQc(this.recordingId, { artifact_id: this.catalog.artifact_id, channel_map: channelMapPayload(this.mapRows), config }));
+            if (!this.contextCurrent(token, this.runGeneration)) return;
+            this.run = run; this.frame = null; this.activeWindow = null; this.activeEvent = null; this.wavePage = 0;
+            this.el.history.add(new Option(`${run.created_at} · ${run.status}`, run.run_id)); this.el.history.value = run.run_id;
+            this.busy = false; renderQc(this); this.pollRun(run.run_id);
+        } catch (error) { if (this.contextCurrent(token, this.runGeneration)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    pollRun(runId) {
+        clearTimeout(this.pollTimer);
+        const token = this.runGeneration.value;
+        const poll = async () => {
+            try {
+                const run = await apiClient.getEphysQcRun(runId);
+                if (!this.contextCurrent(token, this.runGeneration) || this.run?.run_id !== runId) return;
+                this.applyRun(run);
+                if (['pending', 'running'].includes(run.status)) this.pollTimer = setTimeout(poll, 1500);
+                else { this.status(run.status === 'succeeded' ? 'QC scan ready' : run.error?.message || run.error || 'QC scan failed'); }
+            } catch (error) { if (this.contextCurrent(token, this.runGeneration)) this.error(error); }
+        };
+        poll();
+    },
+
+    async loadWindow() {
+        const token = this.windowGeneration.next(); this.frame = null; renderEvidence(this);
+        if (!this.catalog || !this.channelId) return;
+        try {
+            const selection = this.selectedInterval();
+            if (!selection) throw new Error('QC invalid sample range');
+            const viewport = this.activeEvent ? eventViewport(selection, this.wavePage || 0) : selection;
+            const request = windowRequest(this.catalog, this.channelId, this.el.segment.value, viewport.start_sample, viewport.stop_sample);
+            this.el.wave.textContent = i18n.t('QC loading samples');
+            const frame = await apiClient.getEphysSamples(this.catalog.artifact_id, request);
+            if (!this.contextCurrent(token, this.windowGeneration)) return;
+            this.frame = frame; renderEvidence(this);
+        } catch (error) { if (this.contextCurrent(token, this.windowGeneration)) { this.error(error); this.el.wave.textContent = i18n.t('QC samples unavailable'); } }
+    },
+
+    async saveReview() {
+        if (!this.run || this.run.frozen || !this.dirty || this.busy) return;
+        const token = this.runGeneration.value, runId = this.run.run_id;
+        if (!this.el.reviewer.value.trim()) { this.status('QC reviewer required'); return; }
+        this.busy = true; this.status(); renderReview(this);
+        try {
+            const run = await this.trackMutation(apiClient.saveEphysQcReview(runId, { base_revision: this.run.review.revision,
+                reviewer: this.el.reviewer.value.trim(), decisions: this.decisions }));
+            this.drafts.delete(runId);
+            // A completed write remains saved even if its panel was closed in flight.
+            if (this.run?.run_id === runId) this.dirty = false;
+            if (!this.contextCurrent(token, this.runGeneration)) return;
+            this.busy = false; this.applyRun(run, false); this.status('QC review saved');
+        } catch (error) { if (this.contextCurrent(token, this.runGeneration)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    async freezeRun() {
+        if (!this.run || this.run.frozen || this.dirty || this.busy) return;
+        if (!this.el.reviewer.value.trim()) { this.status('QC reviewer required'); return; }
+        const token = this.runGeneration.value; this.busy = true; renderReview(this);
+        try {
+            const run = await this.trackMutation(apiClient.freezeEphysQcRun(this.run.run_id, { base_revision: this.run.review.revision, reviewer: this.el.reviewer.value.trim() }));
+            if (!this.contextCurrent(token, this.runGeneration)) return;
+            this.busy = false; this.applyRun(run, false); this.status('QC frozen');
+        } catch (error) { if (this.contextCurrent(token, this.runGeneration)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+
+    async exportRun() {
+        if (!this.run?.frozen || this.busy) return;
+        const token = this.runGeneration.value, runId = this.run.run_id; this.busy = true; renderReview(this);
+        try {
+            const blob = await apiClient.exportEphysQcRun(runId);
+            if (!this.contextCurrent(token, this.runGeneration)) return;
+            const url = URL.createObjectURL(blob), link = document.createElement('a');
+            link.href = url; link.download = `seeg-qc-${runId}.zip`; document.body.appendChild(link); link.click(); link.remove();
+            setTimeout(() => URL.revokeObjectURL(url), 10000); this.busy = false; renderReview(this);
+        } catch (error) { if (this.contextCurrent(token, this.runGeneration)) { this.busy = false; this.error(error); renderReview(this); } }
+    },
+};
diff --git a/frontend/src/viewer/ephysQc/panel.css b/frontend/src/viewer/ephysQc/panel.css
new file mode 100644
index 0000000000000000000000000000000000000000..107b593d20d28073b932eb534c7aa6e90f87ec80
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/panel.css
@@ -0,0 +1,44 @@
+.seeg-qc-panel { position: fixed; inset: 64px 18px 18px; z-index: 80; display: flex; flex-direction: column; background: var(--qc-bg, #151c26); color: var(--qc-text, #e2e8f0); border: 1px solid #64748b66; border-radius: 10px; box-shadow: 0 12px 60px #0006; font-size: 12px; }
+.seeg-qc-panel.hidden { display: none; }
+.seeg-qc-panel * { box-sizing: border-box; }
+.seeg-qc-panel button, .seeg-qc-panel select, .seeg-qc-panel input, .seeg-qc-panel textarea { border: 1px solid #64748b66; border-radius: 5px; background: var(--qc-input, #1e293b); color: inherit; font: inherit; padding: 5px 7px; }
+.seeg-qc-panel button { cursor: pointer; white-space: nowrap; }
+.seeg-qc-panel button:hover:not(:disabled) { border-color: #38bdf8; }
+.seeg-qc-panel button:disabled { opacity: .4; cursor: default; }
+.seeg-qc-panel input[type=checkbox] { accent-color: #38bdf8; }
+.seeg-qc-panel h3 { font-size: 12px; margin: 8px 0; }
+.qc-header, .qc-toolbar, .qc-footer, .qc-window-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
+.qc-header { padding: 12px 16px; border-bottom: 1px solid #64748b44; }
+.qc-header strong { font-size: 17px; }.qc-header [data-qc=identity] { flex: 1; opacity: .7; }
+.qc-toolbar { padding: 8px 14px; }.qc-toolbar label { display: flex; gap: 6px; align-items: center; }
+.qc-toolbar input[type=number] { width: 86px; }.qc-toolbar select { max-width: 260px; }
+.qc-status { padding: 0 14px 5px; min-height: 20px; color: #fbbf24; }.qc-status:empty { min-height: 0; padding-bottom: 0; }
+.qc-scroll { flex: 1; overflow: auto; min-height: 0; padding: 0 14px 10px; }
+.qc-map, .seeg-qc-panel details { border-top: 1px solid #64748b44; padding: 7px 0; }.seeg-qc-panel summary { cursor: pointer; }
+.qc-table-scroll { overflow: auto; max-height: 280px; margin-top: 8px; }.seeg-qc-panel table { border-collapse: collapse; width: 100%; }
+.seeg-qc-panel th, .seeg-qc-panel td { text-align: left; padding: 4px; border-bottom: 1px solid #64748b33; }.seeg-qc-panel th { color: #94a3b8; font-weight: 500; }
+.qc-map td:first-child { white-space: nowrap; }.qc-map input:not([type=checkbox]) { width: 100px; }.qc-map select { width: 102px; }
+.qc-map [data-field=contact_names] { width: 130px; }.qc-map input[type=number] { width: 66px; }
+.qc-map tr[data-unconfirmed=true] td:first-child::after { content: ' · ?'; color: #fbbf24; }
+.qc-summary { display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr)); gap: 8px; margin: 8px 0; }
+.qc-stat { border: 1px solid #64748b44; border-radius: 6px; padding: 8px 10px; }.qc-stat strong { display: block; font-size: 18px; margin: 4px 0; }.qc-stat span { color: #94a3b8; }
+.qc-workspace { display: grid; grid-template-columns: 185px minmax(340px, 1fr) 228px; min-height: 530px; border-top: 1px solid #64748b44; }
+.qc-channel-pane { padding: 4px 10px 10px 0; max-height: 670px; overflow: auto; }.qc-channel-pane h4 { font-size: 11px; color: #94a3b8; margin: 13px 0 5px; }
+.qc-channel { display: flex; width: 100%; gap: 8px; margin: 3px 0; text-align: left; }.qc-channel span { flex: 1; overflow: hidden; text-overflow: ellipsis; }.qc-channel.selected { border-color: #38bdf8; background: #38bdf81a; }
+.qc-evidence { padding: 4px 12px 10px; border-left: 1px solid #64748b44; border-right: 1px solid #64748b44; min-width: 0; }.qc-evidence .qc-toolbar { padding: 5px 0; gap: 6px; }
+.qc-evidence .qc-toolbar select { max-width: 130px; }.qc-evidence h3 small { color: #94a3b8; font-weight: 400; }
+.qc-chart { height: 175px; background: var(--qc-input, #1e293b); border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; color: #94a3b8; }.qc-chart svg { width: 100%; height: 100%; }.qc-psd { height: 135px; }
+.qc-window-toolbar { margin: 10px 0 6px; gap: 6px; }.qc-window-toolbar strong { flex: 1; }.qc-timeline { display: grid; grid-template-columns: repeat(12, minmax(25px, 1fr)); gap: 4px; }
+.qc-timeline button { padding: 4px 1px; font-size: 10px; }.qc-timeline .candidate { background: #f59e0b33; border-color: #f59e0b99; }.qc-timeline .unavailable { background: #64748b33; }.qc-timeline .selected { outline: 2px solid #38bdf8; }
+.qc-timeline .reviewed { border-bottom: 3px solid #38bdf8; }.qc-metrics { margin-top: 8px; font-size: 11px; display: flex; flex-wrap: wrap; gap: 4px 12px; color: #94a3b8; }
+.qc-waveform-events { margin: 8px 0; }.qc-waveform-events h3 small { display: inline-block; margin-left: 6px; font-size: 10px; }
+.qc-event-card { border: 1px solid #64748b55; border-radius: 5px; padding: 7px; margin: 5px 0; }.qc-event-card.selected { border-color: #38bdf8; }
+.qc-event-title { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }.qc-event-title > span { font-size: 11px; }.qc-event-stages { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
+.qc-event-stages .qc-stage-jump { border-left: 3px solid #fbbf24; }.qc-event-stages .qc-stage-drift { border-left: 3px solid #a78bfa; }.qc-event-stages .qc-stage-plateau { border-left: 3px solid #fb923c; }.qc-event-stages .qc-stage-recovery { border-left: 3px solid #34d399; }
+.qc-review { padding: 4px 0 10px 12px; }.qc-review label { display: flex; flex-direction: column; gap: 5px; margin-bottom: 9px; }.qc-decisions { display: flex; flex-wrap: wrap; gap: 5px; }
+.qc-muted { color: #94a3b8; font-size: 11px; line-height: 1.5; }.qc-decision-list { max-height: 280px; overflow: auto; }.qc-decision-item { border-top: 1px solid #64748b44; padding: 7px 0; display: flex; gap: 5px; }.qc-decision-item div { flex: 1; min-width: 0; overflow-wrap: anywhere; }.qc-decision-item button { align-self: start; }
+.qc-report-details { padding: 8px 0; overflow-wrap: anywhere; }.qc-report-details pre { white-space: pre-wrap; font-size: 11px; max-height: 200px; overflow: auto; }
+.qc-footer { border-top: 1px solid #64748b44; padding: 10px 14px; }.qc-footer [data-qc=saved] { flex: 1; color: #94a3b8; }
+body.light-theme .seeg-qc-panel { --qc-bg: #f8fafc; --qc-text: #1e293b; --qc-input: #fff; }.seeg-qc-panel [hidden] { display: none; }
+@media (max-width: 1050px) { .seeg-qc-panel { inset: 55px 6px 6px; }.qc-workspace { grid-template-columns: 140px minmax(300px, 1fr); }.qc-review { grid-column: 1 / -1; border-top: 1px solid #64748b44; }.qc-review label { display: inline-flex; width: 45%; margin-right: 12px; }.qc-summary { grid-template-columns: repeat(3, 1fr); } }
+@media (max-width: 640px) { .qc-workspace { display: flex; flex-direction: column; }.qc-channel-pane { max-height: 180px; }.qc-evidence { border: 0; padding: 0; }.qc-summary { grid-template-columns: repeat(2, 1fr); }.qc-toolbar { padding: 6px; gap: 5px; } }
diff --git a/frontend/src/viewer/ephysQc/panel.js b/frontend/src/viewer/ephysQc/panel.js
new file mode 100644
index 0000000000000000000000000000000000000000..05384a592e9b90e158727452d8d1bed3ce470cbe
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/panel.js
@@ -0,0 +1,223 @@
+import { i18n } from '../../core/i18n.js';
+import { createQcPanel } from './template.js';
+import { RequestGeneration, mapEditorRow, sampleSelection, validateDecision, upsertDecision } from './state.js';
+import { renderQc, renderMap } from './render.js';
+import { qcIoMethods } from './io.js';
+import { eventSelection, reviewWindowEvidence, reviewEventEvidence, sampleInputRange } from './waveformEvents.js';
+import './panel.css';
+
+export class EphysQcPanel {
+    constructor({ viewer, onVisibleChange }) {
+        this.viewer = viewer; this.onVisibleChange = onVisibleChange;
+        this.visible = false; this.records = []; this.mapRows = []; this.decisions = [];
+        this.recordGeneration = new RequestGeneration(); this.runGeneration = new RequestGeneration();
+        this.windowGeneration = new RequestGeneration(); this.drafts = new Map(); this.page = 0;
+        this.panel = createQcPanel(); document.body.appendChild(this.panel);
+        this.el = Object.fromEntries([...this.panel.querySelectorAll('[data-qc]')].map(el => [el.dataset.qc, el]));
+        this.listeners = new AbortController();
+        const on = (node, event, fn) => node.addEventListener(event, fn, { signal: this.listeners.signal });
+        on(this.el.close, 'click', () => this.close());
+        on(this.el.refresh, 'click', () => this.loadRecords());
+        on(this.el.recording, 'change', () => this.selectRecording(this.el.recording.value));
+        on(this.el.history, 'change', () => this.selectRun(this.el.history.value));
+        on(this.el.import, 'click', () => this.el.files.click());
+        on(this.el.files, 'change', () => this.importFiles());
+        on(this.el.scan, 'click', () => this.startScan());
+        on(this.el.load, 'click', () => this.loadWindow());
+        for (const name of ['start', 'stop']) on(this.el[name], 'input', () => {
+            this.wavePage = 0;
+            this.frame = null; this.windowGeneration.next(); renderQc(this);
+        });
+        on(this.el.segment, 'change', () => { this.setWindowBounds(); renderQc(this); this.loadWindow(); });
+        on(this.el.save, 'click', () => this.saveReview());
+        on(this.el.reload, 'click', () => this.selectRun(this.run?.run_id, true));
+        on(this.el.freeze, 'click', () => this.freezeRun());
+        on(this.el.export, 'click', () => this.exportRun());
+        on(this.el.previous, 'click', () => { this.page--; renderQc(this); });
+        on(this.el.next, 'click', () => { this.page++; renderQc(this); });
+        on(this.el.issue, 'click', () => {
+            const candidates = this.run?.report?.windows.filter(window => window.flags?.length) || [];
+            const index = candidates.findIndex(window => window.window_id === this.activeWindow?.window_id);
+            if (candidates.length) this.selectWindow(candidates[(index + 1) % candidates.length]);
+        });
+        on(this.el.whole, 'change', () => renderQc(this));
+        on(this.el.map, 'change', event => this.updateMap(event));
+        on(this.el.channels, 'click', event => { const el = event.target.closest('[data-channel]'); if (el) this.selectChannel(el.dataset.channel); });
+        on(this.el.timeline, 'click', event => {
+            const id = event.target.closest('[data-window]')?.dataset.window;
+            const window = this.run?.report?.windows.find(item => item.window_id === id);
+            if (window) this.selectWindow(window);
+        });
+        on(this.el['waveform-events'], 'click', event => {
+            const target = event.target.closest('[data-event]');
+            const candidate = this.run?.report?.waveform_events?.find(item => item.event_id === target?.dataset.event);
+            if (candidate) this.selectWaveformEvent(candidate, target.dataset.stage === undefined ? null : Number(target.dataset.stage));
+        });
+        for (const [name, direction] of [['wave-previous', -1], ['wave-next', 1]]) on(this.el[name], 'click', () => {
+            this.wavePage += direction; this.loadWindow();
+        });
+        on(this.el.decisions, 'click', event => {
+            const index = event.target.closest('[data-remove]')?.dataset.remove;
+            if (index !== undefined && !this.run?.frozen && !this.busy) { this.decisions.splice(Number(index), 1); this.markDirty(); }
+        });
+        for (const [key, action, response] of [['confirm', 'exclude', 'confirmed'], ['reject', 'keep', 'rejected'], ['pending', 'pending', 'pending'], ['exclude', 'exclude'], ['keep', 'keep']]) {
+            on(this.el[key], 'click', () => this.addDecision(action, response));
+        }
+        this.unsubscribeLanguage = i18n.onLanguageChange(() => {
+            this.panel.setAttribute('aria-label', i18n.t('SEEG QC'));
+            renderMap(this); renderQc(this);
+        });
+        renderQc(this);
+    }
+
+    isOpen() { return this.visible; }
+    contextCurrent(token, gate = this.recordGeneration) {
+        return this.visible && gate.current(token) && this.viewer.currentSubject === this.subjectId;
+    }
+    status(message = '') { this.el.status.textContent = i18n.t(message); }
+    error(error) { console.error('[SEEG QC]', error); this.status(error.message || 'QC request failed'); }
+    channelName() { return this.mapRows.find(row => row.channel_id === this.channelId)?.raw_name || ''; }
+
+    open() {
+        if (this.visible) return;
+        this.visible = true; this.panel.classList.remove('hidden'); this.onVisibleChange?.(true);
+        const changed = this.subjectId !== this.viewer.currentSubject;
+        this.subjectId = this.viewer.currentSubject;
+        this.el.identity.textContent = this.subjectId || i18n.t('QC select subject');
+        if (changed || !this.catalog) this.loadRecords();
+        else {
+            const token = this.recordGeneration.value, recordingId = this.recordingId;
+            this.busy = Boolean(this.pendingMutation);
+            renderQc(this);
+            Promise.resolve(this.pendingMutation).catch(() => {}).then(() => {
+                if (this.contextCurrent(token)) this.loadRecords(recordingId);
+            });
+        }
+    }
+
+    close() {
+        this.cacheDraft(); this.visible = false; this.panel.classList.add('hidden');
+        this.recordGeneration.next(); this.runGeneration.next(); this.windowGeneration.next();
+        clearTimeout(this.pollTimer); this.busy = false; this.onVisibleChange?.(false);
+    }
+
+    cacheDraft() {
+        if (this.dirty && this.run) this.drafts.set(this.run.run_id, { baseRevision: this.run.review.revision,
+            decisions: structuredClone(this.decisions), reviewer: this.el.reviewer.value });
+    }
+
+    markDirty() { this.dirty = true; this.cacheDraft(); renderQc(this); }
+
+    updateMap(event) {
+        const input = event.target.closest('[data-field]'), tr = input?.closest('[data-map-index]');
+        if (!tr || !input) return;
+        const index = Number(tr.dataset.mapIndex), row = mapEditorRow(this.mapRows[index]);
+        for (const field of tr.querySelectorAll('[data-field]')) {
+            const key = field.dataset.field;
+            if (key === 'status') row.status = field.checked ? 'confirmed' : 'unreviewed';
+            else if (key === 'contact_names') row[key] = field.value.split(',').map(value => value.trim()).filter(Boolean);
+            else if (key === 'contact_index') row[key] = field.value === '' ? null : Number(field.value);
+            else row[key] = field.value || null;
+        }
+        // Keep edited suggestions separate until the user confirms this row.
+        if (row.status !== 'confirmed') row.candidate = { ...row, candidate: undefined };
+        this.mapRows[index] = row;
+        this.mapDirty = true;
+        this.status('QC mapping applies to next scan');
+        tr.dataset.unconfirmed = String(row.status !== 'confirmed');
+        this.el['map-summary'].textContent = `${this.mapRows.filter(item => item.status === 'confirmed').length} / ${this.mapRows.length}`;
+        renderQc(this);
+    }
+
+    selectChannel(channelId) {
+        this.channelId = channelId; this.activeWindow = null; this.activeEvent = null; this.wavePage = 0; this.frame = null; this.page = 0;
+        this.el.whole.checked = false;
+        const { stream } = sampleSelection(this.catalog, channelId);
+        this.el.segment.replaceChildren(...(stream?.segments || []).map((segment, index) => new Option(`${index + 1} · ${segment.recording_start_seconds}s`, segment.segment_id)));
+        this.setWindowBounds(); renderQc(this); this.loadWindow();
+    }
+
+    setWindowBounds() {
+        const { stream, segment } = sampleSelection(this.catalog, this.channelId, this.el.segment.value);
+        if (!segment) return;
+        this.el.start.value = segment.start_sample;
+        this.el.stop.value = Math.min(segment.stop_sample, segment.start_sample + Math.round(stream.sampling_frequency_hz * 5));
+        this.activeWindow = null; this.activeEvent = null; this.wavePage = 0;
+    }
+
+    selectWindow(window) {
+        if (this.channelId !== window.channel_id) this.selectChannel(window.channel_id);
+        this.activeWindow = window; this.activeEvent = null; this.wavePage = 0; this.el.segment.value = window.segment_id;
+        this.el.whole.checked = false;
+        const windows = this.run.report.windows.filter(item => item.channel_id === window.channel_id);
+        this.page = Math.floor(windows.findIndex(item => item.window_id === window.window_id) / 60);
+        this.el.start.value = window.start_sample; this.el.stop.value = window.stop_sample;
+        const decision = this.decisions.find(item => item.evidence_window_ids?.includes(window.window_id));
+        this.el.reason.value = decision?.reason || '';
+        renderQc(this); this.loadWindow();
+    }
+
+    selectWaveformEvent(event, stageIndex = null) {
+        try {
+            const selection = eventSelection(event, stageIndex);
+            if (this.channelId !== selection.channel_id) this.selectChannel(selection.channel_id);
+            this.activeEvent = event; this.activeWindow = null; this.wavePage = 0; this.el.whole.checked = false;
+            this.el.segment.value = selection.segment_id;
+            this.el.start.value = selection.start_sample; this.el.stop.value = selection.stop_sample;
+            const windows = this.run.report.windows.filter(window => window.channel_id === selection.channel_id);
+            const index = windows.findIndex(window => window.segment_id === selection.segment_id
+                && window.start_sample <= selection.start_sample && window.stop_sample > selection.start_sample);
+            if (index >= 0) this.page = Math.floor(index / 60);
+            this.el.reason.value = this.decisions.find(item => item.channel_id === selection.channel_id
+                && item.segment_id === selection.segment_id && item.start_sample === selection.start_sample
+                && item.stop_sample === selection.stop_sample)?.reason || '';
+            renderQc(this); this.loadWindow();
+        } catch (error) { this.error(error); }
+    }
+
+    selectedInterval() { return sampleInputRange(this.el.start.value, this.el.stop.value); }
+
+    reviewEvidenceIds() {
+        const interval = this.selectedInterval();
+        if (!interval) return [];
+        const { stream } = sampleSelection(this.catalog, this.channelId, this.el.segment.value);
+        const selection = { channel_id: this.channelId, stream_id: stream?.stream_id, segment_id: this.el.segment.value,
+            ...interval };
+        if (this.activeEvent) return reviewEventEvidence(this.run?.report?.windows, this.activeEvent, selection);
+        const id = reviewWindowEvidence(this.activeWindow, selection);
+        return id ? [id] : [];
+    }
+
+    addDecision(action, response) {
+        if (this.run?.frozen || this.run?.status !== 'succeeded') return;
+        try {
+            const selected = sampleSelection(this.catalog, this.channelId, this.el.segment.value);
+            const interval = this.selectedInterval();
+            if (!interval) throw new Error('QC invalid sample range');
+            const evidenceWindowIds = this.reviewEvidenceIds();
+            if (response && response !== 'pending' && !evidenceWindowIds.length) throw new Error('QC selection outside candidate');
+            const decision = validateDecision({ ...selected, start: interval.start_sample, stop: interval.stop_sample,
+                action, response: evidenceWindowIds.length ? response : undefined, reason: this.el.reason.value, evidenceWindowIds });
+            if (this.el.whole.checked) {
+                decision.scope = 'channel';
+                for (const key of ['segment_id', 'start_sample', 'stop_sample', 'response', 'evidence_window_ids']) delete decision[key];
+            }
+            this.decisions = upsertDecision(this.decisions, decision); this.status(); this.markDirty();
+        } catch (error) { this.error(error); }
+    }
+
+    applyRun(run, restoreDraft = true) {
+        this.run = run; this.decisions = structuredClone(run.review?.decisions || []); this.dirty = false; this.mapDirty = false;
+        if (run.report?.channel_map) { this.mapRows = structuredClone(run.report.channel_map); renderMap(this); }
+        const draft = restoreDraft ? this.drafts.get(run.run_id) : null;
+        if (draft && draft.baseRevision === run.review?.revision && !run.frozen) {
+            this.decisions = draft.decisions; this.el.reviewer.value = draft.reviewer; this.dirty = true;
+        } else if (draft) this.status('QC draft revision conflict');
+        if (run.error) this.status(typeof run.error === 'string' ? run.error : JSON.stringify(run.error));
+        renderQc(this);
+    }
+
+    dispose() { this.close(); this.listeners.abort(); this.unsubscribeLanguage?.(); this.panel.remove(); }
+}
+
+Object.assign(EphysQcPanel.prototype, qcIoMethods);
diff --git a/frontend/src/viewer/ephysQc/render.js b/frontend/src/viewer/ephysQc/render.js
new file mode 100644
index 0000000000000000000000000000000000000000..770307bae80c775c74babc8ecd3daad9d96b0e71
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/render.js
@@ -0,0 +1,132 @@
+import { i18n } from '../../core/i18n.js';
+import { esc, label } from './template.js';
+import { numberText, percentText, mapEditorRow } from './state.js';
+import { rawWaveform, spectrumGraph } from './charts.js';
+import { visibleWaveformEvents, renderWaveformEvents } from './eventRender.js';
+import { boundaryDwellFeatures } from './waveformEvents.js';
+
+const option = (value, selected, title = value) => `<option value="${esc(value)}" ${value === selected ? 'selected' : ''}>${esc(title)}</option>`;
+const types = ['unknown', 'seeg', 'eeg', 'ecog', 'ecg', 'eog', 'emg', 'stim', 'misc'];
+const select = (field, value, options) => `<select data-field="${field}">${options.map(item => option(item, value)).join('')}</select>`;
+const field = (name, value, type = 'text') => `<input data-field="${name}" type="${type}" value="${esc(value)}">`;
+
+export function renderMap(panel) {
+    panel.el.map.innerHTML = panel.mapRows.map((raw, index) => {
+        const row = mapEditorRow(raw);
+        return `<tr data-map-index="${index}" data-unconfirmed="${row.status !== 'confirmed'}"><td title="${esc(row.channel_id)}">${esc(row.raw_name)}</td>
+            <td>${select('channel_type', row.channel_type, types)}</td><td>${field('shaft', row.shaft)}</td>
+            <td>${field('contact_index', row.contact_index, 'number')}</td><td>${select('measurement_kind', row.measurement_kind || 'unknown', ['unknown', 'monopolar', 'bipolar', 'linear'])}</td>
+            <td>${field('contact_names', (row.contact_names || []).join(', '))}</td><td>${field('reference', row.reference)}</td>
+            <td><input type="checkbox" data-field="status" ${row.status === 'confirmed' ? 'checked' : ''} aria-label="${esc(i18n.t('QC confirmed'))}"><small>${esc((row.issues || []).join(', '))}</small></td></tr>`;
+    }).join('');
+    panel.el['map-summary'].textContent = `${panel.mapRows.filter(row => row.status === 'confirmed').length} / ${panel.mapRows.length}`;
+}
+
+export function renderChannels(panel) {
+    const report = panel.run?.report, windows = report?.windows || [];
+    const groups = new Map();
+    for (const row of panel.mapRows) {
+        const key = row.shaft || row.candidate?.shaft || i18n.t('QC ungrouped');
+        if (!groups.has(key)) groups.set(key, []);
+        groups.get(key).push(row);
+    }
+    panel.el.channels.innerHTML = [...groups].map(([shaft, rows]) => `<h4>${esc(shaft)}</h4>${rows.map(row => {
+        const count = windows.filter(window => window.channel_id === row.channel_id && window.flags?.length).length;
+        return `<button class="qc-channel ${panel.channelId === row.channel_id ? 'selected' : ''}" data-channel="${esc(row.channel_id)}"><span>${esc(row.raw_name)}</span><small>${count ? `⚑ ${count}` : '—'}</small></button>`;
+    }).join('')}`).join('');
+}
+
+export function renderSummary(panel) {
+    const score = panel.run?.report?.initial_score, summary = panel.run?.review?.summary;
+    const stat = (key, value, detail = '') => `<div class="qc-stat"><span>${label(key)}</span><strong>${esc(value)}</strong><small>${esc(detail)}</small></div>`;
+    const scanned = panel.run?.report?.coverage?.scanned_fraction;
+    panel.el.summary.innerHTML = [stat('QC initial score', numberText(score?.total, 0), Number.isFinite(scanned) ? `${i18n.t('QC scanned coverage')} ${percentText(scanned)}` : ''),
+        stat('QC reviewed score', numberText(summary?.reviewed_score, 0), Number.isFinite(summary?.evidence_reviewed_ratio) ? `${i18n.t('QC evidence coverage')} ${percentText(summary.evidence_reviewed_ratio)}` : ''),
+        stat('QC retained share', Number.isFinite(summary?.retained_percent) ? `${numberText(summary.retained_percent, 1)}%` : '—'),
+        stat('QC reviewed coverage', percentText(summary?.reviewed_ratio)),
+        stat('QC excluded time', Number.isFinite(summary?.excluded_seconds) ? `${numberText(summary.excluded_seconds, 1)} ch·s` : '—'),
+        stat('QC unknown time', Number.isFinite(summary?.unknown_seconds) ? `${numberText(summary.unknown_seconds, 1)} ch·s` : '—')].join('');
+    const report = panel.run?.report, integrity = report?.integrity;
+    const table = (headers, rows) => `<table><thead><tr>${headers.map(key => `<th>${label(key)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${esc(value)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
+    const name = id => panel.mapRows.find(row => row.channel_id === id)?.raw_name || id;
+    const sourceName = id => panel.catalog?.streams.find(stream => stream.stream_id === id)?.source_relative_path || id;
+    panel.el['report-details'].innerHTML = report ? `<h3>${label('QC score components')}</h3>${table(['QC issue', 'QC penalty', 'QC windows', 'QC channel seconds'],
+        (score.components || []).map(item => [i18n.t(`QC flag ${item.flag}`), item.penalty, item.window_count, numberText(item.channel_seconds, 1)]))}
+        <h3>${label('QC actual scan ranges')}</h3>${table(['QC source', 'QC source time seconds', 'QC samples'], (report.coverage.ranges || []).map(range => {
+            const stream = panel.catalog?.streams.find(item => item.stream_id === range.stream_id);
+            const segment = stream?.segments.find(item => item.segment_id === range.segment_id);
+            const time = sample => segment ? segment.recording_start_seconds + (sample - segment.start_sample) / stream.sampling_frequency_hz : null;
+            return [sourceName(range.stream_id), `${numberText(time(range.start_sample), 2)} – ${numberText(time(range.stop_sample), 2)}`, `[${range.start_sample}, ${range.stop_sample})`];
+        }))}
+        <h3>${label('QC input and reference')}</h3><p>${i18n.t('QC mapping pending')}: ${integrity?.mapping_pending_channel_ids?.length ?? '—'} · ${i18n.t('QC unknown units')}: ${integrity?.unknown_voltage_unit_channel_ids?.length ?? '—'} · ${i18n.t('QC gaps')}: ${integrity?.gaps?.length ?? '—'} · ${i18n.t('QC scanned coverage')}: ${percentText(report.coverage.scanned_fraction)}</p>
+        ${table(['QC source', 'QC sample rate', 'QC channels', 'QC segments', 'QC reference'], (integrity?.streams || []).map(item => [sourceName(item.stream_id), `${item.sampling_frequency_hz} Hz`, item.channel_count, item.segment_count, item.reference]))}
+        <h3>${label('QC channel relations')}</h3>${table(['QC pair', 'QC relation', 'QC correlation', 'QC samples'], (report.relations || []).filter(item => item.channel_ids.includes(panel.channelId)).slice(0, 60).map(item => [item.channel_ids.map(name).join(' / '), i18n.t(`QC relation ${item.kind}`), numberText(item.correlation, 4), `[${item.start_sample}, ${item.stop_sample})`]))}
+        <h3>${label('QC final channel summary')}</h3>${table(['QC original name', 'QC total seconds', 'QC kept seconds', 'QC excluded seconds', 'QC unknown seconds'], (summary?.channels || []).map(item => [item.raw_name, numberText(item.total_seconds, 1), numberText(item.kept_seconds, 1), numberText(item.excluded_seconds, 1), numberText(item.unknown_seconds, 1)]))}
+        <h3>${label('QC common usable time')}</h3>${table(['QC source', 'QC kept seconds'], (summary?.common_usable || []).map(item => [sourceName(item.stream_id), numberText(item.duration_seconds, 1)]))}
+        ${(panel.run.assessment?.limitations || []).map(value => `<p>${esc(value)}</p>`).join('')}` : esc(i18n.t('QC no scan yet'));
+}
+
+export function renderEvidence(panel) {
+    panel.el.wave.innerHTML = rawWaveform(panel.frame, visibleWaveformEvents(panel));
+    const metadata = panel.frame?.metadata;
+    panel.el['wave-meta'].textContent = metadata ? `${metadata.sampling_frequency_hz} Hz · [${metadata.start_sample}, ${metadata.stop_sample}) · ${i18n.t('QC raw')}` : '';
+    panel.el.psd.innerHTML = spectrumGraph(panel.run?.report?.spectra?.find(item => item.channel_id === panel.channelId));
+    const window = panel.activeWindow;
+    const boundary = boundaryDwellFeatures(window?.metrics);
+    const features = [
+        ['rms_v', 'QC RMS', 1e6, 'µV'], ['mad_v', 'QC MAD', 1e6, 'µV'], ['robust_range_v', 'QC amplitude range', 1e6, 'µV'],
+        ['offset_v', 'QC offset', 1e6, 'µV'], ['drift_v_per_second', 'QC drift', 1e6, 'µV/s'], ['max_jump_v', 'QC largest jump', 1e6, 'µV'],
+        ['missing_fraction', 'QC missing samples', 100, '%'], ['flat_fraction', 'QC flat share', 100, '%'],
+        ['longest_flat_seconds', 'QC longest flat', 1, 's'],
+        ['line_noise_db', 'QC line prominence', 1, 'dB'], ['line_noise_fraction', 'QC line power share', 100, '%'],
+        ['high_frequency_ratio', 'QC high frequency share', 100, '%'], ['kurtosis', 'QC kurtosis', 1, ''],
+    ];
+    panel.el.metrics.innerHTML = window ? `<strong>${esc((window.flags || []).map(flag => i18n.t(`QC flag ${flag}`)).join(' · ') || i18n.t('QC no automatic flags'))}</strong> `
+        + features.map(([key, title, scale, unit]) => `<span>${label(title)}: ${Number.isFinite(window.metrics?.[key]) ? `${numberText(window.metrics[key] * scale)} ${unit}` : '—'}</span>`).join('')
+        + `<span>${label(boundary.is_boundary_evidence ? 'QC boundary dwell share' : 'QC clipping share')}: ${Number.isFinite(boundary.fraction) ? `${numberText(boundary.fraction * 100)}%` : '—'}</span>
+           <span>${label(boundary.is_boundary_evidence ? 'QC longest boundary dwell' : 'QC longest clipping')}: ${Number.isFinite(boundary.longest_seconds) ? `${numberText(boundary.longest_seconds)} s` : '—'}</span>
+           ${boundary.is_boundary_evidence && window.flags?.includes('boundary_dwell') ? `<span>${label('QC evidence without automatic penalty')}</span>` : ''}` : '';
+    renderWaveformEvents(panel);
+    panel.el.scope.textContent = panel.channelId ? `${panel.channelName()} · ${panel.el.whole.checked ? i18n.t('QC whole channel') : `[${panel.el.start.value}, ${panel.el.stop.value})`}` : '';
+}
+
+export function renderTimeline(panel) {
+    const windows = panel.run?.report?.windows?.filter(item => item.channel_id === panel.channelId) || [];
+    const pages = Math.max(1, Math.ceil(windows.length / 60));
+    panel.page = Math.min(Math.max(0, panel.page), pages - 1);
+    panel.el.page.textContent = `${panel.page + 1} / ${pages}`;
+    panel.el.previous.disabled = panel.page === 0; panel.el.next.disabled = panel.page + 1 >= pages;
+    panel.el.timeline.innerHTML = windows.slice(panel.page * 60, (panel.page + 1) * 60).map((window, index) => {
+        const reviewed = panel.decisions.some(item => item.evidence_window_ids?.includes(window.window_id));
+        return `<button data-window="${esc(window.window_id)}" class="${window.flags?.length ? 'candidate' : ''} ${window.status !== 'computed' ? 'unavailable' : ''} ${reviewed ? 'reviewed' : ''} ${panel.activeWindow?.window_id === window.window_id ? 'selected' : ''}" title="${esc(`[${window.start_sample}, ${window.stop_sample}) · ${(window.flags || []).join(', ') || window.status}`)}">${panel.page * 60 + index + 1}</button>`;
+    }).join('') || esc(i18n.t('QC no scan yet'));
+}
+
+export function renderReview(panel) {
+    panel.el.decisions.innerHTML = panel.decisions.map((item, index) => `<div class="qc-decision-item"><div><strong>${esc(i18n.t(`QC action ${item.action}`))}</strong> ${esc(panel.mapRows.find(row => row.channel_id === item.channel_id)?.raw_name || item.channel_id)}<br>
+        <small>${esc(item.scope === 'channel' ? i18n.t('QC whole channel') : `[${item.start_sample}, ${item.stop_sample})`)}</small><br>${esc(item.reason)}</div>
+        <button data-remove="${index}" ${panel.run?.frozen || panel.busy ? 'disabled' : ''} aria-label="${esc(i18n.t('QC remove decision'))}">×</button></div>`).join('');
+    const mutable = panel.run?.status === 'succeeded' && !panel.run.frozen && !panel.busy;
+    for (const key of ['confirm', 'reject', 'pending', 'exclude', 'keep']) panel.el[key].disabled = !mutable || !panel.channelId || !panel.selectedInterval();
+    panel.el.confirm.disabled ||= !panel.reviewEvidenceIds().length || panel.el.whole.checked;
+    panel.el.reject.disabled ||= !panel.reviewEvidenceIds().length || panel.el.whole.checked;
+    panel.el.issue.disabled = !panel.run?.report?.windows.some(window => window.flags?.length);
+    panel.el.save.disabled = !mutable || !panel.dirty;
+    panel.el.freeze.disabled = !mutable || panel.dirty || panel.mapDirty;
+    panel.el.export.disabled = !panel.run?.frozen || panel.busy;
+    panel.el.reload.disabled = !panel.run || panel.busy;
+    panel.el.recording.disabled = panel.busy;
+    panel.el.history.disabled = panel.busy || !panel.catalog;
+    panel.el.refresh.disabled = panel.busy;
+    panel.el.import.disabled = panel.busy;
+    const scanning = ['pending', 'running'].includes(panel.run?.status);
+    panel.el.scan.disabled = !panel.catalog || panel.busy || panel.dirty || scanning;
+    for (const input of panel.el.map.querySelectorAll('input, select')) input.disabled = panel.busy || scanning;
+    const version = panel.run?.review?.revision;
+    panel.el.saved.textContent = panel.dirty ? i18n.t('QC unsaved changes') : panel.run ? `${i18n.t(panel.run.frozen ? 'QC frozen' : 'QC saved')} · r${version ?? 0}${panel.run.assessment ? ` · ${i18n.t(`QC assessment ${panel.run.assessment.status}`)}` : ''}` : '';
+    panel.el['run-status'].textContent = panel.run ? i18n.t(`QC status ${panel.run.status}`) : '';
+    const option = [...panel.el.history.options].find(item => item.value === panel.run?.run_id);
+    if (option) option.textContent = `${panel.run.created_at} · ${panel.run.status}`;
+}
+
+export function renderQc(panel) { renderChannels(panel); renderSummary(panel); renderEvidence(panel); renderTimeline(panel); renderReview(panel); }
diff --git a/frontend/src/viewer/ephysQc/state.js b/frontend/src/viewer/ephysQc/state.js
new file mode 100644
index 0000000000000000000000000000000000000000..b7ed6cb55f08e0ec2e385ff54366d939f77aed8b
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/state.js
@@ -0,0 +1,86 @@
+/** UI contracts only: scientific naming and metrics come from the scan service. */
+export function numberText(value, digits = 2) {
+    return Number.isFinite(value) ? value.toFixed(digits) : '—';
+}
+
+export function percentText(ratio) {
+    if (!Number.isFinite(ratio)) return '—';
+    return ratio > 0 && ratio < .001 ? '<0.1%' : `${(ratio * 100).toFixed(1)}%`;
+}
+
+export class RequestGeneration {
+    constructor() { this.value = 0; }
+    next() { return ++this.value; }
+    current(value) { return this.value === value; }
+}
+
+export function channelMapPayload(rows) {
+    return rows.filter(row => row.status === 'confirmed').map(row => ({
+        channel_id: row.channel_id, channel_type: row.channel_type,
+        shaft: row.shaft || null, contact_index: row.contact_index === '' ? null : row.contact_index,
+        measurement_kind: row.measurement_kind, contact_names: [...(row.contact_names || [])],
+        reference: row.reference || null, source_channels: [...(row.source_channels || [])], status: 'confirmed',
+    }));
+}
+
+export function mapEditorRow(row) {
+    const fields = row.status === 'confirmed' ? row : row.candidate || {};
+    return { ...row, ...fields, raw_name: row.raw_name, channel_id: row.channel_id, status: row.status };
+}
+
+export function validateDecision({ channel, stream, segment, start, stop, action, reason, response, evidenceWindowId, evidenceWindowIds }) {
+    if (!channel || !stream || !segment) throw new Error('QC select channel and segment');
+    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(stop) || start < segment.start_sample
+        || stop > segment.stop_sample || stop <= start) throw new Error('QC invalid sample range');
+    if (!['keep', 'exclude', 'pending'].includes(action)) throw new Error('QC invalid action');
+    if (!reason?.trim()) throw new Error('QC reason required');
+    const result = { channel_id: channel.channel_id, stream_id: stream.stream_id,
+        segment_id: segment.segment_id, start_sample: start, stop_sample: stop,
+        scope: 'interval', action, reason: reason.trim() };
+    if (evidenceWindowIds?.length) result.evidence_window_ids = [...evidenceWindowIds];
+    else if (evidenceWindowId) result.evidence_window_ids = [evidenceWindowId];
+    if (response) result.response = response;
+    return result;
+}
+
+export function sampleSelection(catalog, channelId, segmentId) {
+    const stream = catalog?.streams?.find(item => item.channels.some(channel => channel.channel_id === channelId));
+    const channel = stream?.channels.find(item => item.channel_id === channelId);
+    const segment = stream?.segments.find(item => item.segment_id === segmentId) || stream?.segments[0];
+    return { stream, channel, segment };
+}
+
+export function windowRequest(catalog, channelId, segmentId, start, stop) {
+    const { stream, channel, segment } = sampleSelection(catalog, channelId, segmentId);
+    if (!stream || !channel || !segment || !Number.isSafeInteger(start) || !Number.isSafeInteger(stop)
+        || start < segment.start_sample || stop > segment.stop_sample || start >= stop) throw new Error('QC invalid sample range');
+    if (stop - start > 200000) throw new Error('QC waveform range too large');
+    return { artifact_id: catalog.artifact_id, recording_id: catalog.recording_id,
+        content_sha256: catalog.content_sha256, reader_version: catalog.reader_version,
+        stream_id: stream.stream_id, segment_id: segment.segment_id, channel_ids: [channelId],
+        start_sample: start, stop_sample: stop, representation: 'physical', data_kind: 'raw_samples',
+        clock_id: stream.clock_id, source_sha256: stream.source_sha256, source_relative_path: stream.source_relative_path };
+}
+
+export function upsertDecision(decisions, decision) {
+    // An exact interval replacement is intentional; overlaps remain explicit for server validation.
+    return [...decisions.filter(item => !(item.channel_id === decision.channel_id
+        && item.segment_id === decision.segment_id && item.start_sample === decision.start_sample
+        && item.stop_sample === decision.stop_sample)), decision];
+}
+
+export function scanRanges(catalog, startSeconds, stopSeconds) {
+    if (!Number.isFinite(startSeconds) || startSeconds < 0 || (stopSeconds !== null
+        && (!Number.isFinite(stopSeconds) || stopSeconds <= startSeconds))) throw new Error('QC invalid scan range');
+    if (startSeconds === 0 && stopSeconds === null) return undefined;
+    const ranges = [];
+    for (const stream of catalog.streams) for (const segment of stream.segments) {
+        const rate = stream.sampling_frequency_hz, origin = segment.recording_start_seconds;
+        const start = Math.max(segment.start_sample, segment.start_sample + Math.ceil((startSeconds - origin) * rate));
+        const stop = stopSeconds === null ? segment.stop_sample
+            : Math.min(segment.stop_sample, segment.start_sample + Math.floor((stopSeconds - origin) * rate));
+        if (start < stop) ranges.push({ stream_id: stream.stream_id, segment_id: segment.segment_id, start_sample: start, stop_sample: stop });
+    }
+    if (!ranges.length) throw new Error('QC empty scan range');
+    return ranges;
+}
diff --git a/frontend/src/viewer/ephysQc/template.js b/frontend/src/viewer/ephysQc/template.js
new file mode 100644
index 0000000000000000000000000000000000000000..24c6c7cdee7a3d18c97233be2451224e020ca0af
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/template.js
@@ -0,0 +1,58 @@
+import { i18n } from '../../core/i18n.js';
+
+export const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
+export const label = key => `<span data-i18n="${esc(key)}">${esc(i18n.t(key))}</span>`;
+export const button = (name, key, extra = '') => `<button type="button" data-qc="${name}" ${extra}>${label(key)}</button>`;
+const input = (name, key, props) => `<label>${label(key)}<input data-qc="${name}" ${props}></label>`;
+
+export function createQcPanel() {
+    const panel = document.createElement('section');
+    panel.id = 'seeg-qc-panel';
+    panel.className = 'seeg-qc-panel hidden';
+    panel.setAttribute('role', 'dialog');
+    panel.setAttribute('aria-label', i18n.t('SEEG QC'));
+    panel.innerHTML = `
+        <header class="qc-header"><strong>${label('SEEG QC')}</strong><span data-qc="identity"></span>${button('close', 'Close')}</header>
+        <div class="qc-toolbar"><label>${label('QC recording')}<select data-qc="recording"></select></label>
+            ${button('refresh', 'QC refresh')}${button('import', 'QC import EDF/BDF')}
+            <input data-qc="files" type="file" accept=".edf,.bdf" multiple hidden>
+            <label>${label('QC run')}<select data-qc="history"></select></label><span data-qc="run-status"></span>
+        </div>
+        <div data-qc="status" class="qc-status" role="status"></div>
+        <div class="qc-scroll">
+            <details class="qc-map"><summary>${label('QC channel mapping')} <span data-qc="map-summary"></span></summary>
+                <div class="qc-table-scroll"><table><thead><tr>${['QC original name', 'QC type', 'QC shaft', 'QC contact', 'QC measurement', 'QC endpoints', 'QC reference', 'QC confirmed'].map(key => `<th>${label(key)}</th>`).join('')}</tr></thead><tbody data-qc="map"></tbody></table></div>
+            </details>
+            <div class="qc-toolbar qc-scan"><label>${label('QC line frequency')}<select data-qc="line"><option>50</option><option>60</option></select></label>
+                ${input('window', 'QC window seconds', 'type="number" min="1" max="30" step="1" value="5"')}
+                ${input('scan-start', 'QC scan start', 'type="number" min="0" step="0.1" value="0"')}
+                ${input('scan-stop', 'QC scan stop', 'type="number" min="0" step="0.1" placeholder="—"')}
+                ${button('scan', 'QC start scan')}<small>${label('QC next scan settings')}</small><span data-qc="catalog-summary"></span>
+            </div>
+            <div class="qc-summary" data-qc="summary"></div>
+            <div class="qc-workspace">
+                <aside class="qc-channel-pane"><h3>${label('QC channels and issues')}</h3><div data-qc="channels"></div></aside>
+                <main class="qc-evidence">
+                    <div class="qc-toolbar"><label>${label('QC segment')}<select data-qc="segment"></select></label>
+                        ${input('start', 'QC start sample', 'type="number" min="0" step="1"')}${input('stop', 'QC stop sample', 'type="number" min="1" step="1"')}
+                        ${button('load', 'QC load window')}</div>
+                    <h3>${label('QC raw waveform')} <small data-qc="wave-meta"></small></h3><div data-qc="wave" class="qc-chart"></div>
+                    <div data-qc="wave-pages" class="qc-window-toolbar" hidden>${button('wave-previous', 'QC previous waveform page')}${button('wave-next', 'QC next waveform page')}<span data-qc="wave-page"></span></div>
+                    <section data-qc="waveform-events" class="qc-waveform-events" hidden></section>
+                    <h3>${label('QC scan PSD')} <small>V²/Hz</small></h3><div data-qc="psd" class="qc-chart qc-psd"></div>
+                    <div class="qc-window-toolbar"><strong>${label('QC time windows')}</strong>${button('issue', 'QC next issue')}${button('previous', 'QC previous')}${button('next', 'QC next')}<span data-qc="page"></span></div>
+                    <div data-qc="timeline" class="qc-timeline"></div><div data-qc="metrics" class="qc-metrics"></div>
+                </main>
+                <aside class="qc-review"><h3>${label('QC review')}</h3>
+                    ${input('reviewer', 'QC reviewer', 'type="text" autocomplete="off"')}
+                    <label class="qc-scope-toggle"><span>${label('QC whole channel')}</span><input type="checkbox" data-qc="whole"></label>
+                    <label>${label('QC reason')}<textarea data-qc="reason" rows="3"></textarea></label>
+                    <div class="qc-decisions">${button('confirm', 'QC confirm issue')}${button('reject', 'QC false alarm')}${button('pending', 'QC pending')}${button('exclude', 'QC manual exclude')}${button('keep', 'QC manual keep')}</div>
+                    <p data-qc="scope" class="qc-muted"></p><div data-qc="decisions" class="qc-decision-list"></div>
+                </aside>
+            </div>
+            <details><summary>${label('QC report details')}</summary><div data-qc="report-details" class="qc-report-details"></div></details>
+        </div>
+        <footer class="qc-footer"><span data-qc="saved"></span>${button('save', 'QC save review')}${button('reload', 'QC reload saved')}${button('freeze', 'QC freeze')}${button('export', 'QC download ZIP')}</footer>`;
+    return panel;
+}
diff --git a/frontend/src/viewer/ephysQc/waveformEvents.js b/frontend/src/viewer/ephysQc/waveformEvents.js
new file mode 100644
index 0000000000000000000000000000000000000000..381d60b3fd030c80c3c6ae4ff635a8a24a87e871
--- /dev/null
+++ b/frontend/src/viewer/ephysQc/waveformEvents.js
@@ -0,0 +1,81 @@
+/** Exact source intervals for waveform-event viewing; no waveform classification here. */
+export function sampleInputRange(startText, stopText) {
+    if (String(startText ?? '').trim() === '' || String(stopText ?? '').trim() === '') return null;
+    const start = Number(startText), stop = Number(stopText);
+    return Number.isSafeInteger(start) && Number.isSafeInteger(stop) && start >= 0 && stop > start
+        ? { start_sample: start, stop_sample: stop } : null;
+}
+
+export function editableEventViewport(startText, stopText, page = 0) {
+    const selection = sampleInputRange(startText, stopText);
+    return selection ? eventViewport(selection, page) : null;
+}
+
+export function associatedWaveformEvents(report, selection) {
+    if (!selection) return [];
+    const ids = selection.waveform_event_ids;
+    return (report?.waveform_events || []).filter(event => event.channel_id === selection.channel_id
+        && event.stream_id === selection.stream_id && event.segment_id === selection.segment_id
+        && event.start_sample < selection.stop_sample && event.stop_sample > selection.start_sample
+        && (!Array.isArray(ids) || ids.includes(event.event_id)));
+}
+
+export function eventSelection(event, stageIndex = null) {
+    const interval = stageIndex === null ? event : event?.stages?.[stageIndex];
+    if (!event || !interval || !Number.isSafeInteger(interval.start_sample) || !Number.isSafeInteger(interval.stop_sample)
+        || interval.start_sample < event.start_sample || interval.stop_sample > event.stop_sample
+        || interval.stop_sample <= interval.start_sample) throw new Error('QC invalid event range');
+    return { channel_id: event.channel_id, stream_id: event.stream_id, segment_id: event.segment_id,
+        start_sample: interval.start_sample, stop_sample: interval.stop_sample };
+}
+
+export function eventViewport(selection, page = 0, limit = 200000) {
+    if (!Number.isSafeInteger(selection?.start_sample) || !Number.isSafeInteger(selection?.stop_sample)
+        || selection.stop_sample <= selection.start_sample || !Number.isSafeInteger(page)
+        || !Number.isSafeInteger(limit) || limit <= 0) throw new Error('QC invalid event range');
+    const pages = Math.ceil((selection.stop_sample - selection.start_sample) / limit);
+    if (page < 0 || page >= pages) throw new Error('QC invalid event page');
+    const start = selection.start_sample + page * limit;
+    return { start_sample: start, stop_sample: Math.min(start + limit, selection.stop_sample), page, pages };
+}
+
+export function reviewWindowEvidence(window, selection) {
+    if (!window || !selection || !sampleInputRange(selection.start_sample, selection.stop_sample)
+        || window.channel_id !== selection.channel_id || window.stream_id !== selection.stream_id
+        || window.segment_id !== selection.segment_id || selection.start_sample < window.start_sample
+        || selection.stop_sample > window.stop_sample || selection.start_sample >= selection.stop_sample) return undefined;
+    return window.window_id;
+}
+
+export function reviewEventEvidence(windows, event, selection) {
+    if (!event || !selection || !sampleInputRange(selection.start_sample, selection.stop_sample)
+        || event.channel_id !== selection.channel_id || event.stream_id !== selection.stream_id
+        || event.segment_id !== selection.segment_id || selection.start_sample < event.start_sample
+        || selection.stop_sample > event.stop_sample || selection.start_sample >= selection.stop_sample) return [];
+    return [...new Set((windows || []).filter(window => window.channel_id === event.channel_id
+        && window.stream_id === event.stream_id && window.segment_id === event.segment_id
+        && window.start_sample < selection.stop_sample && window.stop_sample > selection.start_sample)
+        .map(window => window.window_id))];
+}
+
+export function boundaryDwellFeatures(metrics = {}) {
+    const explicit = Object.hasOwn(metrics, 'boundary_dwell_fraction');
+    return {
+        fraction: explicit ? metrics.boundary_dwell_fraction : metrics.clipping_fraction,
+        longest_seconds: explicit ? metrics.longest_boundary_dwell_seconds : metrics.longest_clipping_seconds,
+        is_boundary_evidence: explicit || metrics.adc_range_status === 'unconfirmed',
+    };
+}
+
+export function waveformStageBands(events, metadata) {
+    if (!metadata) return [];
+    return (events || []).filter(event => event.channel_id === metadata.channel_ids[0]
+        && event.stream_id === metadata.stream_id && event.segment_id === metadata.segment_id)
+        .flatMap(event => (event.stages || []).map(stage => {
+            const start = Math.max(stage.start_sample, metadata.start_sample);
+            const stop = Math.min(stage.stop_sample, metadata.stop_sample);
+            if (start >= stop) return null;
+            const seconds = sample => metadata.recording_start_seconds + (sample - metadata.start_sample) / metadata.sampling_frequency_hz;
+            return { kind: stage.kind, start: seconds(start), stop: seconds(stop) };
+        }).filter(Boolean));
+}
diff --git a/frontend/src/viewer/lifecycle/subjectLifecycle.js b/frontend/src/viewer/lifecycle/subjectLifecycle.js
index c44257a34fb92d2206696fc0507c2741e0215e98..1501e24b4901f714beeed85061da3ea0f5d4a842 100644
--- a/frontend/src/viewer/lifecycle/subjectLifecycle.js
+++ b/frontend/src/viewer/lifecycle/subjectLifecycle.js
@@ -9,2 +9,3 @@ export const subjectLifecycleMethods = {
         const changed = subjectId !== this.currentSubject;
+        if (changed) this.ephysQcPanel?.close();
         // Leave figure mode BEFORE currentSubject moves. restore:false because the old
diff --git a/frontend/src/viewer/panels/panelCoordinator.js b/frontend/src/viewer/panels/panelCoordinator.js
index ad1755a6d913ee4292f2e99d382339be206c6375..a42cd988841a6267feb5e042ba1fc01bd08d57fd 100644
--- a/frontend/src/viewer/panels/panelCoordinator.js
+++ b/frontend/src/viewer/panels/panelCoordinator.js
@@ -9,2 +9,3 @@ import { EEGZoomManager } from '../eegZoomManager.js';
 import { scheduleWhenHostVisible } from '../shared/canvasLifecycle.js';
+import { EphysQcPanel } from '../ephysQc/panel.js';

@@ -310,2 +311,19 @@ export const panelCoordinatorMethods = {
         }
+        this.ephysQcPanel = new EphysQcPanel({
+            viewer: this,
+            onVisibleChange: visible => {
+                if (visible) {
+                    this.eegAdvancedPanel?.close();
+                    this.closeEEGPanel();
+                    this.seegPanel?.close();
+                    this.seegLocalizationPanel?.close();
+                    this.ecogPanel?.close();
+                    this.dtiPanel?.close();
+                    this.slicePanel?.setVisible(false);
+                    const toggle = document.getElementById('toggle-slices');
+                    if (toggle) toggle.checked = false;
+                }
+                this.scheduleResize();
+            },
+        });
     },
diff --git a/frontend/src/viewer/ui/controlPanel.js b/frontend/src/viewer/ui/controlPanel.js
index e4d0aa300a053c67e6b7536d204107424e2de9f8..17c19220e911d656597675a433f97e4cbec94be8 100644
--- a/frontend/src/viewer/ui/controlPanel.js
+++ b/frontend/src/viewer/ui/controlPanel.js
@@ -633,2 +633,7 @@ export function setupControlPanel(viewer) {

+            if (action === 'open-seeg-qc') {
+                viewer.ephysQcPanel?.open();
+                return;
+            }
+
             const requestedMode = sensorMode?.value || viewer.sensorMode || 'eeg';
````
