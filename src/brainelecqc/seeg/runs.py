"""Independent SEEG scans, saved review revisions and frozen exports.

Adapted from the Cortrix (VISVB) SEEG quality-control workflow.
"""

from copy import deepcopy
import hashlib
import json
import platform
from uuid import uuid4

from ..db import current_ts, get_db
from ..recordings import get_recording
from ..signals import assert_unchanged, verified_catalog
from ..storage import ImportConflict, canonical_bytes, digest
from .export import EXPORT_FORMAT, build_package
from .review import SCOPE, summarize, validate_review


def _json(value):
    return canonical_bytes(value).decode("utf-8")


def _row(connection, run_id):
    row = connection.execute("SELECT * FROM seeg_qc_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise LookupError("SEEG QC run not found")
    return row


def _audit(connection, run_id, revision, action, snapshot, reviewer=None):
    connection.execute(
        "INSERT INTO seeg_qc_audit(run_id,revision,action,reviewer,snapshot_json,created_at) VALUES(?,?,?,?,?,?)",
        (run_id, revision, action, reviewer, _json(snapshot), current_ts()),
    )


def _verify_input(row):
    entry = verified_catalog(row["artifact_id"])
    provenance = json.loads(row["provenance_json"])
    for key in ("recording_id", "artifact_id", "content_sha256", "manifest_sha256"):
        if entry["catalog"][key] != provenance[key]:
            raise ImportConflict("QC input fingerprint changed; restore input or start a new scan")
    return entry


def run_seeg_qc(artifact_id, *, config=None, channel_map=None):
    """Scan a verified SEEG artifact and persist the resulting evidence at revision zero."""
    import numpy
    try:
        import scipy
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise
        raise ValueError("SEEG QC needs SciPy. Install BrainElecQC with the [qc] extra.") from error
    from .seeg import SEEG_ALGORITHM_VERSION, scan_seeg

    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be an object")
    if channel_map is not None and not isinstance(channel_map, list):
        raise ValueError("channel_map must be a list")
    request = {"config": deepcopy(config or {}), "channel_map": deepcopy(channel_map)}
    entry = verified_catalog(artifact_id)
    catalog = entry["catalog"]
    recording_id = catalog["recording_id"]
    if get_recording(recording_id)["modality"] != "seeg":
        raise ValueError("This scan profile requires an explicitly imported SEEG recording")
    if not catalog["streams"]:
        raise ValueError("No readable signal stream; inspect the recording input")
    provenance = {
        key: catalog[key] for key in ("recording_id", "artifact_id", "content_sha256", "manifest_sha256", "reader_version")
    }
    provenance.update(
        algorithm_version=SEEG_ALGORITHM_VERSION, request_sha256=digest(request),
        software={"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__},
    )
    run_id, now = str(uuid4()), current_ts()
    with get_db() as connection:
        connection.execute(
            "INSERT INTO seeg_qc_runs(run_id,recording_id,artifact_id,status,provenance_json,request_json,"
            "review_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, recording_id, artifact_id, "running", _json(provenance), _json(request),
             _json({"decisions": [], "summary": None}), now, now),
        )
        _audit(connection, run_id, 0, "created", provenance)

    def read_window(stream_id, segment_id, channel_ids, start_sample, stop_sample, representation):
        values = entry["readers"][stream_id].read_window(
            stream_id, segment_id, channel_ids, start_sample, stop_sample, representation=representation,
        )
        assert_unchanged(entry)
        return values

    try:
        report = scan_seeg(deepcopy(catalog), read_window, **request)
        report["catalog"] = deepcopy(catalog)
        review = {"decisions": [], "summary": summarize(report["catalog"], [], report["windows"])}
        report_hash = digest(report)
        with get_db() as connection:
            assert_unchanged(entry)
            connection.execute(
                "UPDATE seeg_qc_runs SET status='succeeded',report_json=?,report_sha256=?,review_json=?,updated_at=? "
                "WHERE run_id=?",
                (_json(report), report_hash, _json(review), current_ts(), run_id),
            )
            _audit(connection, run_id, 0, "scan_completed", {"report_sha256": report_hash})
    except (Exception, KeyboardInterrupt) as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        message = str(error) or "Scan interrupted"
        with get_db() as connection:
            connection.execute(
                "UPDATE seeg_qc_runs SET status=?,error=?,updated_at=? WHERE run_id=?",
                (status, message, current_ts(), run_id),
            )
            _audit(connection, run_id, 0, f"scan_{status}", {"error": message})
        raise
    return get_run(run_id)


def _present_run(row):
    report = json.loads(row["report_json"]) if row["report_json"] else None
    provenance = json.loads(row["provenance_json"])
    request = json.loads(row["request_json"])
    if report is not None and digest(report) != row["report_sha256"]:
        raise ImportConflict("QC report fingerprint mismatch; restore the persisted report")
    if digest(request) != provenance["request_sha256"]:
        raise ImportConflict("QC request fingerprint mismatch; restore the persisted request")
    run = {key: row[key] for key in
           ("run_id", "recording_id", "artifact_id", "status", "error", "created_at", "updated_at")}
    run.update(profile="seeg_basic_v1", scope=SCOPE, provenance=provenance, request=request, report=report,
               frozen=bool(row["frozen_json"]), review={**json.loads(row["review_json"]), "revision": row["revision"]})
    if row["frozen_json"]:
        snapshot = json.loads(row["frozen_json"])
        run.update(frozen_at=snapshot["frozen_at"], frozen_by=snapshot["frozen_by"], assessment=snapshot["assessment"])
    return run


def get_run(run_id):
    """Read the current scan and review revision."""
    with get_db() as connection:
        return _present_run(_row(connection, run_id))


def get_audit(run_id):
    """Read saved revisions and other run actions in transaction order."""
    with get_db() as connection:
        _row(connection, run_id)
        rows = connection.execute(
            "SELECT revision,action,reviewer,snapshot_json,created_at FROM seeg_qc_audit "
            "WHERE run_id=? ORDER BY id", (run_id,),
        ).fetchall()
    for row in rows:
        row["snapshot"] = json.loads(row.pop("snapshot_json"))
    return rows


def _editable(row, revision):
    if row["frozen_json"]:
        raise ImportConflict("QC run is frozen; create a new scan to revise it")
    if row["revision"] != revision:
        raise ImportConflict("QC revision conflict; reload the current review before saving")
    if row["status"] != "succeeded":
        raise ImportConflict("Wait for a successful scan before review or freeze")


def _decision_content(decision):
    return {key: value for key, value in decision.items() if key not in ("reviewer", "reviewed_at")}


def save_review(run_id, payload):
    """Validate and commit a full decision list against its base revision."""
    with get_db() as connection:
        row = _row(connection, run_id)
    run = _present_run(row)
    if run["report"] is None:
        raise ImportConflict("Wait for a successful scan before review")
    decisions = validate_review(payload, run["report"]["catalog"], run["report"]["windows"])
    _editable(row, payload["base_revision"])
    previous = {item["decision_id"]: item for item in run["review"]["decisions"]}
    for decision in decisions:
        old = previous.get(decision["decision_id"])
        if old and _decision_content(old) == _decision_content(decision):
            decision.update(reviewer=old["reviewer"], reviewed_at=old["reviewed_at"])
    review = {
        "decisions": decisions,
        "summary": summarize(run["report"]["catalog"], decisions, run["report"]["windows"]),
    }
    entry = _verify_input(row)
    with get_db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _editable(_row(connection, run_id), payload["base_revision"])
        assert_unchanged(entry)
        revision = row["revision"] + 1
        connection.execute(
            "UPDATE seeg_qc_runs SET revision=?,review_json=?,updated_at=? WHERE run_id=?",
            (revision, _json(review), current_ts(), run_id),
        )
        _audit(connection, run_id, revision, "review_saved", review, payload["reviewer"].strip())
        return _present_run(_row(connection, run_id))


def _assessment(run):
    report = run["report"]
    summary = run["review"]["summary"]
    limitations = ["本次 QC 评估信号技术质量。病理活动、刺激事件和解剖定位需另行评估。"]
    if summary["unknown_seconds"]:
        limitations.append("未复核或待定区间保留为未知。可用时间由明确保留的区间计算。")
    if any(item.get("status") != "confirmed" for item in report["channel_map"]):
        limitations.append("部分通道映射尚待确认。")
    coverage = report["coverage"]
    if coverage["scanned_fraction"] < 1:
        limitations.append(f"自动扫描覆盖 {coverage['scanned_fraction']:.1%} 的通道时间。")
    if report["integrity"].get("source_errors"):
        limitations.append("部分输入文件读取失败。统计范围为可读信号流。")
    if report["integrity"].get("unknown_voltage_unit_channel_ids"):
        limitations.append("部分通道电压单位待确认，幅度指标待评估。")
    status = "pending" if not summary["reviewed_ratio"] else "degraded" if len(limitations) > 1 else "reviewed"
    if not summary["kept_seconds"]:
        availability = "pending" if summary["unknown_seconds"] else "unusable"
    else:
        availability = "limited_usable" if summary["excluded_seconds"] or len(limitations) > 1 else "usable"
    return {"status": status, "availability": availability, "scope": SCOPE, "limitations": limitations}


def freeze_run(run_id, payload):
    """Freeze a successfully scanned revision and verify its export before publication."""
    if not isinstance(payload, dict) or type(payload.get("base_revision")) is not int:
        raise ValueError("base_revision must be an integer")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
        raise ValueError("A reviewer name is required (max 200 characters)")
    with get_db() as connection:
        row = _row(connection, run_id)
    _editable(row, payload["base_revision"])
    entry = _verify_input(row)
    snapshot = _present_run(row)
    snapshot.update(
        frozen=True, frozen_at=current_ts(), frozen_by=reviewer.strip(), assessment=_assessment(snapshot),
        export_validation={"format": EXPORT_FORMAT, "readback": "passed"},
    )
    payload_bytes = build_package(snapshot)
    checksum = hashlib.sha256(payload_bytes).hexdigest()
    with get_db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _editable(_row(connection, run_id), payload["base_revision"])
        assert_unchanged(entry)
        connection.execute(
            "UPDATE seeg_qc_runs SET frozen_json=?,export_sha256=?,updated_at=? WHERE run_id=?",
            (_json(snapshot), checksum, current_ts(), run_id),
        )
        _audit(connection, run_id, row["revision"], "frozen",
               {"export_sha256": checksum, "assessment": snapshot["assessment"]}, reviewer.strip())
    return get_run(run_id)


def export_run(run_id):
    """Return the verified bytes for the frozen review."""
    with get_db() as connection:
        row = _row(connection, run_id)
    if not row["frozen_json"]:
        raise ImportConflict("Freeze the reviewed version before export")
    entry = _verify_input(row)
    payload = build_package(json.loads(row["frozen_json"]))
    if hashlib.sha256(payload).hexdigest() != row["export_sha256"]:
        raise ImportConflict("Frozen export fingerprint mismatch; restore the saved version")
    assert_unchanged(entry)
    return payload
