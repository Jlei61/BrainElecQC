"""Sample-accurate human review: unknown is distinct from accepted signal.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
from collections import defaultdict
from uuid import uuid4

from ..db import current_ts

SCOPE = "seeg_basic_signal_only"


def validate_review(payload, catalog, windows):
    if not isinstance(payload, dict) or type(payload.get("base_revision")) is not int:
        raise ValueError("base_revision must be an integer")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
        raise ValueError("A reviewer name is required (max 200 characters)")
    raw = payload.get("decisions")
    if not isinstance(raw, list) or len(raw) > 20000:
        raise ValueError("decisions must be a list with at most 20000 entries")
    channels = {c["channel_id"]: s for s in catalog["streams"] for c in s["channels"]}
    evidence = {w["window_id"]: w for w in windows}
    result, ids = [], set()
    for item in raw:
        if not isinstance(item, dict) or item.get("channel_id") not in channels:
            raise ValueError("Each decision needs a channel_id in this input")
        stream = channels[item["channel_id"]]
        if item.get("stream_id", stream["stream_id"]) != stream["stream_id"]:
            raise ValueError("Decision stream and channel disagree")
        scope = item.get("scope", "interval")
        action, reason = item.get("action"), item.get("reason")
        if scope not in ("interval", "channel") or action not in ("keep", "exclude", "pending"):
            raise ValueError("Use scope channel/interval and action keep/exclude/pending")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("Every decision needs a reason (max 2000 characters)")
        response = item.get("response")
        if response not in (None, "confirmed", "rejected", "pending"):
            raise ValueError("response must be confirmed, rejected or pending")
        if (response is not None and ((response == "pending") != (action == "pending")) or
                response == "confirmed" and action != "exclude" or
                response == "rejected" and action != "keep"):
            raise ValueError("response must match the technical issue decision")
        decision_id = item.get("decision_id") or str(uuid4())
        if not isinstance(decision_id, str) or len(decision_id) > 100 or decision_id in ids:
            raise ValueError("decision_id must be distinct within a revision")
        ids.add(decision_id)
        decision = {"decision_id": decision_id, "channel_id": item["channel_id"],
                    "stream_id": stream["stream_id"], "scope": scope, "action": action,
                    "response": response, "reason": reason.strip(), "reviewer": reviewer.strip(),
                    "reviewed_at": current_ts(), "applicability": SCOPE,
                    "evidence_window_ids": item.get("evidence_window_ids", [])}
        if not isinstance(decision["evidence_window_ids"], list) or any(
                not isinstance(wid, str) or wid not in evidence or
                evidence[wid]["channel_id"] != item["channel_id"] for wid in decision["evidence_window_ids"]):
            raise ValueError("Evidence windows must belong to the reviewed channel")
        if scope == "interval":
            segment = next((s for s in stream["segments"] if s["segment_id"] == item.get("segment_id")), None)
            a, b = item.get("start_sample"), item.get("stop_sample")
            if (segment is None or type(a) is not int or type(b) is not int or
                    not segment["start_sample"] <= a < b <= segment["stop_sample"]):
                raise ValueError("Use a half-open integer sample interval inside one segment; split at gaps")
            decision.update(segment_id=segment["segment_id"], start_sample=a, stop_sample=b)
            if any(evidence[wid]["segment_id"] != segment["segment_id"] or
                   evidence[wid]["start_sample"] >= b or evidence[wid]["stop_sample"] <= a
                   for wid in decision["evidence_window_ids"]):
                raise ValueError("Evidence must overlap the reviewed interval in the same segment")
        result.append(decision)
    # Reject contradictory entries at the same specificity; local intervals may
    # override whole-channel decisions deliberately, independent of list order.
    groups = defaultdict(list)
    for item in result:
        key = item["channel_id"], item["scope"], item.get("segment_id")
        for previous in groups[key]:
            overlaps = item["scope"] == "channel" or (item["start_sample"] < previous["stop_sample"] and
                                                        previous["start_sample"] < item["stop_sample"])
            if overlaps and item["action"] != previous["action"]:
                raise ValueError("Overlapping decisions conflict; edit or remove the existing interval")
        groups[key].append(item)
    return result


def _pieces(channel_id, segment, decisions):
    whole = [d for d in decisions if d["channel_id"] == channel_id and d["scope"] == "channel"]
    local = [d for d in decisions if d["channel_id"] == channel_id and
             d["scope"] == "interval" and d["segment_id"] == segment["segment_id"]]
    boundaries = sorted({segment["start_sample"], segment["stop_sample"]} |
                        {d[k] for d in local for k in ("start_sample", "stop_sample")})
    base = whole[0]["action"] if whole else "pending"
    return [(a, b, next((d["action"] for d in local if d["start_sample"] <= a < d["stop_sample"]), base))
            for a, b in zip(boundaries, boundaries[1:])]


def _technical_score(windows, decisions):
    """Score only explicitly adjudicated evidence, at exact interval boundaries."""
    from .seeg_scoring import compute_quality_score
    slices, responses = [], {}
    by_window = defaultdict(list)
    for decision in decisions:
        if decision.get("response"):
            for window_id in decision["evidence_window_ids"]:
                by_window[window_id].append(decision)
    for window in windows:
        relevant = by_window[window["window_id"]]
        a, b = window["start_sample"], window["stop_sample"]
        boundaries = {a, b}
        for decision in relevant:
            if decision["scope"] == "interval":
                boundaries.update((max(a, decision["start_sample"]), min(b, decision["stop_sample"])))
        boundaries = sorted(boundaries)
        for first, last in zip(boundaries, boundaries[1:]):
            candidates = [d for d in relevant if d["scope"] == "channel" or d["start_sample"] <= first < d["stop_sample"]]
            candidates.sort(key=lambda d: d["scope"] == "interval", reverse=True)
            response = candidates[0]["response"] if candidates else "pending"
            slice_id = f"{window['window_id']}:slice:{first}:{last}"
            slices.append({**window, "window_id": slice_id, "start_sample": first, "stop_sample": last,
                           "duration_seconds": window["duration_seconds"] * (last-first) / (b-a)})
            responses[slice_id] = "accepted" if response == "confirmed" else response
    result = compute_quality_score(slices, responses)
    result["aggregation"] = "duration_weighted_sample_fragments_max_penalty"
    return result


def summarize(catalog, decisions, windows=None):
    rows, common = [], []
    for stream in catalog["streams"]:
        fs = stream["sampling_frequency_hz"]
        channel_pieces = {}
        for channel in stream["channels"]:
            times = {"keep": 0.0, "exclude": 0.0, "pending": 0.0}
            by_segment = {}
            for segment in stream["segments"]:
                parts = _pieces(channel["channel_id"], segment, decisions)
                by_segment[segment["segment_id"]] = parts
                for a, b, action in parts:
                    times[action] += (b - a) / fs
            channel_pieces[channel["channel_id"]] = by_segment
            total = sum(times.values())
            rows.append({"channel_id": channel["channel_id"], "stream_id": stream["stream_id"],
                         "raw_name": channel.get("raw_name"), "total_seconds": total,
                         "kept_seconds": times["keep"], "excluded_seconds": times["exclude"],
                         "unknown_seconds": times["pending"],
                         "reviewed_ratio": (total - times["pending"]) / total if total else 0})
        shared = 0.0
        for segment in stream["segments"]:
            pieces = [by_segment[segment["segment_id"]] for by_segment in channel_pieces.values()]
            bounds = sorted({point for parts in pieces for a, b, _ in parts for point in (a, b)})
            for a, b in zip(bounds, bounds[1:]):
                if pieces and all(any(x <= a < y and action == "keep" for x, y, action in parts) for parts in pieces):
                    shared += (b - a) / fs
        common.append({"stream_id": stream["stream_id"], "channel_ids": list(channel_pieces),
                       "duration_seconds": shared, "scope": "within_stream_only"})
    total = sum(r["total_seconds"] for r in rows)
    kept = sum(r["kept_seconds"] for r in rows)
    excluded = sum(r["excluded_seconds"] for r in rows)
    unknown = sum(r["unknown_seconds"] for r in rows)
    reviewed = kept + excluded
    technical = _technical_score(windows or [], decisions)
    return {"channels": rows, "total_channel_seconds": total, "kept_seconds": kept,
            "excluded_seconds": excluded, "unknown_seconds": unknown,
            "reviewed_ratio": reviewed / total if total else 0,
            "unknown_ratio": unknown / total if total else 1,
            "retained_percent": round(100 * kept / reviewed) if reviewed else None,
            "reviewed_score": technical["total"], "score": technical["total"], "technical_score": technical,
            "score_basis": "technical_risk_with_explicit_evidence_responses",
            "evidence_reviewed_ratio": technical["reviewed_channel_seconds"] / technical["assessed_channel_seconds"]
                if technical["assessed_channel_seconds"] else 0,
            "common_usable": common, "interval_convention": "half_open"}
