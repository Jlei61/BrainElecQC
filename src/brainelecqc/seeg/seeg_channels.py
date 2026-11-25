"""Preserved channel identities and reviewable SEEG naming candidates.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
from collections import defaultdict
from copy import deepcopy
import re


_CONTACT = re.compile(r"^([^\d\s\-]+)(\d+)$")
_SCALP = re.compile(r"^(?:FP|AF|F|FT|FC|C|T|TP|CP|P|PO|O)(?:Z|\d+)$", re.I)
_AUX = re.compile(r"^(ECG|EKG|EOG|EMG|TRIG|TRIGGER|STIM|STATUS)(?:\b|\d|[_-])", re.I)
_TYPES = {"unknown", "seeg", "eeg", "ecog", "ecg", "eog", "emg", "stim", "misc"}
_FIELDS = {"channel_id", "channel_type", "status", "shaft", "contact_index", "measurement_kind",
           "contact_names", "reference", "source_channels"}


def _contact(name):
    match = _CONTACT.fullmatch(name)
    return (match.group(1), int(match.group(2))) if match else None


def _candidate(name):
    result = {"shaft": None, "contact_index": None, "measurement_kind": "unknown",
              "contact_names": [], "channel_type": "unknown", "reference": None, "name_adapter": None}
    text = name.strip()
    # These vendor labels describe acquisition slots, not biological modality.
    stripped = re.sub(r"^(?:EEG|POL)\s+", "", text, flags=re.I)
    if stripped != text:
        result["name_adapter"] = "vendor_prefix_candidate_only"
        text = stripped
    if text.lower().endswith("-ref"):
        result["reference"] = text[-3:]
        text = text[:-4]
    aux = _AUX.match(text)
    if aux:
        kind = aux.group(1).lower()
        result["channel_type"] = {"ekg": "ecg", "trig": "stim", "trigger": "stim", "status": "stim"}.get(kind, kind)
        return result
    if _SCALP.fullmatch(text):
        result["channel_type"] = "eeg"
        return result
    pair = text.split("-")
    if len(pair) == 2 and _contact(pair[0]):
        first = _contact(pair[0])
        second_name = first[0] + pair[1] if pair[1].isdigit() else pair[1]
        second = _contact(second_name)
        if second:
            result.update(shaft=first[0] if first[0] == second[0] else None,
                          measurement_kind="bipolar", contact_names=[pair[0], second_name], channel_type="seeg")
            return result
    contact = _contact(text)
    if contact:
        result.update(shaft=contact[0], contact_index=contact[1], measurement_kind="monopolar",
                      contact_names=[text], channel_type="seeg")
    return result


def resolve_seeg_channels(catalog, channel_map=None):
    """Return candidates; only an explicit confirmed map establishes identities.

    Names are never identifiers. In particular A1/A01 remain separate inputs,
    and candidate bipolar contacts never cause a numerical re-reference.
    """
    if channel_map is None:
        channel_map = []
    if not isinstance(channel_map, list):
        raise ValueError("channel_map must be a list")
    overrides = {}
    for item in channel_map:
        if not isinstance(item, dict) or set(item) - _FIELDS or not isinstance(item.get("channel_id"), str):
            raise ValueError("channel_map entries require channel_id and supported mapping fields")
        if item["channel_id"] in overrides:
            raise ValueError("Duplicate channel_id in channel_map")
        overrides[item["channel_id"]] = item
    output, seen = [], set()
    for stream in catalog.get("streams", []):
        stream_ids = {c["channel_id"] for c in stream["channels"]}
        for channel in stream["channels"]:
            cid = channel["channel_id"]
            if cid in seen:
                raise ValueError("Channel IDs must be globally unique")
            seen.add(cid)
            raw_name = channel["raw_name"]
            if not isinstance(raw_name, str):
                raise ValueError("raw_name must preserve the original string")
            item = {"channel_id": cid, "raw_name": raw_name, "stream_id": stream["stream_id"],
                    "source_id": stream.get("source_id"), "unit": channel.get("unit"),
                    "channel_type": channel.get("channel_type", "unknown").lower(), "status": "unreviewed",
                    "shaft": None, "contact_index": None, "measurement_kind": "unknown", "contact_names": [],
                    "reference": None, "source_channels": [], "candidate": _candidate(raw_name), "issues": []}
            override = overrides.get(cid, {})
            status = override.get("status", "unreviewed")
            if status not in ("unreviewed", "confirmed"):
                raise ValueError("Mapping status must be unreviewed or confirmed")
            if override and status != "confirmed":
                raise ValueError("Explicit channel mapping must carry status=confirmed")
            item.update(deepcopy(override))
            _validate_mapping(item, stream_ids)
            if item["status"] != "confirmed":
                item["issues"].append("mapping_pending")
            if item["measurement_kind"] == "bipolar":
                ends = [_contact(name) for name in item["contact_names"]]
                if all(ends) and (ends[0][0] != ends[1][0] or abs(ends[1][1] - ends[0][1]) != 1):
                    item["issues"].append("nonadjacent_bipolar_contacts")
            output.append(item)
    if set(overrides) - seen:
        raise ValueError("channel_map contains unknown channel_id")
    _annotate_names(output)
    return output


def _validate_mapping(item, stream_ids):
    if item["channel_type"] not in _TYPES:
        raise ValueError("Unsupported channel_type")
    kind = item["measurement_kind"]
    if kind not in ("unknown", "monopolar", "bipolar", "linear"):
        raise ValueError("Unsupported measurement_kind")
    names = item["contact_names"]
    if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("contact_names must contain explicit nonempty contact names")
    if kind in ("monopolar", "bipolar") and len(names) != (1 if kind == "monopolar" else 2):
        raise ValueError("Monopolar and bipolar maps require one and two contact names respectively")
    if len(names) != len(set(names)):
        raise ValueError("A measurement cannot repeat the same contact")
    sources = item["source_channels"]
    if (not isinstance(sources, list) or any(not isinstance(c, str) or c not in stream_ids for c in sources)
            or len(sources) != len(set(sources))):
        raise ValueError("source_channels must be distinct channel IDs from the same stream")
    if item["reference"] is not None and not isinstance(item["reference"], str):
        raise ValueError("reference must be a string or null")
    index = item["contact_index"]
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
        raise ValueError("contact_index must be a nonnegative integer or null")
    if item["shaft"] is not None and (not isinstance(item["shaft"], str) or not item["shaft"]):
        raise ValueError("shaft must be a nonempty string or null")


def _annotate_names(channels):
    aliases, shafts, raw_names = defaultdict(list), defaultdict(list), defaultdict(list)
    for item in channels:
        candidate = item["candidate"]
        raw_names[(item["stream_id"], item["raw_name"])].append(item)
        if candidate["contact_index"] is not None:
            key = (item["stream_id"], candidate["shaft"])
            shafts[key].append(item)
            aliases[key + (candidate["contact_index"],)].append(item)
    for group in raw_names.values():
        if len(group) > 1:
            for item in group:
                item["issues"].append("duplicate_raw_name")
    for group in aliases.values():
        if len({c["raw_name"] for c in group}) > 1:
            for item in group:
                item["issues"].append("possible_alias")
    for group in shafts.values():
        numbers = sorted({c["candidate"]["contact_index"] for c in group})
        if any(b - a > 1 for a, b in zip(numbers, numbers[1:])):
            for item in group:
                item["issues"].append("missing_contact_numbers")
