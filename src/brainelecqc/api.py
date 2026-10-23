"""HTTP metadata and binary adapter; file paths are never accepted from requests."""

import json
import logging
from flask import Blueprint, Response, jsonify, request
from werkzeug.exceptions import BadRequest

from . import recordings
from .signals import describe_artifact, list_events
from .storage import ImportConflict
from .windows import samples, overview, encode_window

ephys_bp = Blueprint("v1_ephys", __name__, url_prefix="/api/v1/ephys")
ephys_samples_bp = Blueprint("v1_ephys_samples", __name__, url_prefix="/api/v1/ephys")
logger = logging.getLogger(__name__)


def ok(data, meta=None):
    return jsonify({"data": data, "meta": meta or {}})


def fail(code, message, status):
    return jsonify({"error": {"code": code, "message": message, "details": {}}}), status


@ephys_bp.errorhandler(ImportConflict)
def conflict(error):
    return fail("EPHYS_IMPORT_CONFLICT", str(error), status=409)


@ephys_bp.errorhandler(ValueError)
@ephys_bp.errorhandler(BadRequest)
def invalid(error):
    return fail("EPHYS_INVALID_INPUT", str(error), status=400)


@ephys_bp.errorhandler(LookupError)
def missing(error):
    return fail("EPHYS_NOT_FOUND", str(error), status=404)


@ephys_bp.errorhandler(OSError)
def storage_error(error):
    logger.exception("QC input storage failed")
    return fail("EPHYS_STORAGE_ERROR", "Input storage failed; inspect server logs and retry", status=500)


@ephys_bp.post("/recordings")
def create_recording():
    if request.mimetype == "multipart/form-data":
        payload = json.loads(request.form.get("metadata", "{}"))
        uploads = request.files.getlist("files")
    else:
        raise ValueError("Use multipart/form-data with metadata and files; server paths are not accepted")
    result, reused = recordings.import_recording(payload, uploads)
    return ok(result, {"reused": reused}), 200 if reused else 201


@ephys_bp.get("/recordings")
def list_recordings():
    return ok(recordings.list_recordings(request.args.get("participant_id")))


@ephys_bp.get("/recordings/<recording_id>")
def get_recording(recording_id):
    return ok(recordings.get_recording(recording_id))


@ephys_bp.post("/recordings/<recording_id>/verify")
def verify_recording(recording_id):
    return ok(recordings.verify_recording(recording_id))


@ephys_samples_bp.errorhandler(ImportConflict)
def samples_conflict(error):
    return fail("EPHYS_INPUT_CHANGED", str(error), status=409)


@ephys_samples_bp.errorhandler(ValueError)
def samples_invalid(error):
    return fail("EPHYS_INVALID_WINDOW", str(error), status=400)


@ephys_samples_bp.errorhandler(LookupError)
def samples_missing(error):
    return fail("EPHYS_NOT_FOUND", str(error), status=404)


@ephys_samples_bp.errorhandler(OSError)
def inaccessible(error):
    logger.exception("Signal input unavailable")
    return fail("EPHYS_READ_FAILED", "Signal input unavailable; inspect logs", status=500)


def _integer(name, default=None):
    value = request.args.get(name, default)
    if value is None or isinstance(value, str) and (not value.isascii() or not value.isdecimal()):
        raise ValueError(f"{name} must be an explicit nonnegative integer")
    return int(value)


def _parameters():
    for key in ("stream_id", "segment_id", "representation"):
        if not request.args.get(key):
            raise ValueError(f"{key} is required")
    return {"stream_id": request.args["stream_id"], "segment_id": request.args["segment_id"],
            "channel_ids": request.args.getlist("channel_id"), "start_sample": _integer("start_sample"),
            "stop_sample": _integer("stop_sample"), "representation": request.args["representation"]}


def _binary(result):
    metadata, array = result
    response = Response(encode_window(metadata, array), content_type="application/octet-stream")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@ephys_samples_bp.get("/artifacts/<artifact_id>/signals")
def signals_route(artifact_id):
    return ok(describe_artifact(artifact_id))


@ephys_samples_bp.get("/artifacts/<artifact_id>/events")
def events_route(artifact_id):
    return ok(list_events(artifact_id, _integer("offset", 0), _integer("limit", 100)))


@ephys_samples_bp.get("/artifacts/<artifact_id>/samples")
def samples_route(artifact_id):
    return _binary(samples(artifact_id, **_parameters()))


@ephys_samples_bp.get("/artifacts/<artifact_id>/overview")
def overview_route(artifact_id):
    return _binary(overview(artifact_id, **_parameters(), bin_samples=_integer("bin_samples")))
