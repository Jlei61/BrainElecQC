"""Input declarations describe acquisition; they are not a QC verdict."""
import json
import math

SCHEMA_VERSION = 1
METRIC_STATES = ("computed", "insufficient_data", "disabled", "failed", "not_applicable")
SCIENTIFIC_VALIDITY = ("pending", "valid", "degraded", "blocking")
MODALITIES = ("eeg", "seeg", "ecog")


def descriptor(payload):
    if not isinstance(payload, dict):
        raise ValueError("Import metadata must be an object")
    allowed = {"participant_id", "label", "modality", "session_id", "metadata", "import_key"}
    if set(payload) - allowed:
        raise ValueError(f"Unknown import fields: {sorted(set(payload) - allowed)}")
    label = payload.get("label")
    if not isinstance(label, str) or not label.strip() or len(label) > 512:
        raise ValueError("A nonempty label of at most 512 characters is required")
    if payload.get("modality") not in MODALITIES:
        raise ValueError(f"modality must explicitly be one of {MODALITIES}")
    for key in ("participant_id", "session_id", "import_key"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or not value or len(value) > 256):
            raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object; unknown units/timing must remain null")
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("metadata exceeds the 4 MiB limit")
    hints = metadata.get("format_hints", {})
    if (not isinstance(hints, dict) or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in hints.items())):
        raise ValueError("metadata.format_hints must map relative file paths to explicit format names")
    channels = metadata.get("channels", [])
    if not isinstance(channels, list) or any(not isinstance(c, dict) for c in channels):
        raise ValueError("metadata.channels must be a list of objects")
    ids = [c.get("channel_id") for c in channels]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Each declared channel needs a unique channel_id independent of its name")
    streams = metadata.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError("metadata.streams must be a list")
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("metadata.streams entries must be objects")
        rate = stream.get("sample_rate_hz")
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                                or not math.isfinite(rate) or rate <= 0):
            raise ValueError("sample_rate_hz must be positive or null when unknown")
    return {"participant_id": payload.get("participant_id"), "label": label,
            "modality": payload["modality"], "session_id": payload.get("session_id"),
            "metadata": json.loads(encoded)}
