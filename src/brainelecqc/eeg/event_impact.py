"""Localize repeatable EEG coupling in individual, held-out event windows.

The existing split-repeatability gate selects channels. Individual windows are
compared to a median template made from nonoverlapping events in the other half
of the record. Window coverage is a review measure at the configured epoch scale.
"""
import copy
import numpy as np

from .event_score import _correlation, _integer
from .impact import percent, time_burden, time_points


POLICY = {"version": "event-impact-windows-1.0", "shape_correlation_min": .50,
          "projected_amplitude_ratio_min": .10,
          "template": "median_of_nonoverlapping_training_events_in_opposite_temporal_half",
          "window": "saved_complete_detector_epoch", "minimum_training_events_per_half": 10}


def score_event_impact(result, kind, coupling):
    """Keep the coupling gate, then score the union of individually supported windows."""
    outcome = copy.deepcopy(coupling)
    if outcome["score"] is None:
        return outcome
    f = result["features"]
    origin, stop = f["start_sample"], f["stop_sample"]
    data = np.asarray(result["raw_data_volts"])
    block = result["method_results"]["events"]["by_kind"][kind]
    metrics = outcome["metrics"]
    metrics["coupling_score"] = outcome["score"]
    metrics["impact_policy"] = copy.deepcopy(POLICY)
    training_ids = set(metrics["event_ids"])
    complete = []
    ids = set()
    for event in block["event_table"]:
        if event["event_id"] in ids:
            raise ValueError("Event IDs must be unique for impact localization")
        ids.add(event["event_id"])
        if not event["complete_epoch"]:
            continue
        a, b, sample = (_integer(event[k], k) for k in ("epoch_start_sample", "epoch_stop_sample", "sample"))
        if not origin <= a <= sample < b <= stop or b-a < 2:
            raise ValueError("Complete event windows must match the input sample clock")
        for key, position in (("time_seconds", sample), ("epoch_start_seconds", a), ("epoch_stop_seconds", b)):
            expected = f["start_seconds"]+(position-origin)/f["sampling_rate_hz"]
            if key in event and not np.isclose(event[key], expected, rtol=0, atol=1e-9):
                raise ValueError("Event impact time must match original samples")
        complete.append(event)
    complete.sort(key=lambda e: (e["sample"], e["event_id"]))
    candidates = {c["channel_id"] for c in metrics["channels"] if c["repeatability_pass"]
                  and c["ratio"] >= metrics["config"]["ratio_edges"][0]}
    if not candidates:
        metrics.update(time_burden=time_burden(f, []), impact_observations=[], impact_affected_event_count=0,
                       impact_complete_event_count=len(complete), impact_unassessed_time=time_burden(f, []))
        outcome.update(score=10, evidence=[], components=[{"id": "time_burden", "deduction": 0}],
                       reason=f"{kind.upper()} 未检出通过锁时重复性及振幅门槛的影响，异常事件窗占比 0%")
        return outcome
    training = [e for e in complete if e["event_id"] in training_ids]
    if len(training) < 2*POLICY["minimum_training_events_per_half"]:
        outcome.update(score=None, reason="事件窗定位需要每半段至少 10 个训练事件")
        return outcome
    offsets = {(e["epoch_start_sample"]-e["sample"], e["epoch_stop_sample"]-e["sample"]) for e in complete}
    if len(offsets) != 1:
        raise ValueError("All complete event windows must use identical sample offsets")
    half = len(training)//2
    split = (training[half-1]["sample"]+training[half]["sample"])/2
    halves = [training[:half], training[half:]]
    # Remove training windows crossing the split; no target can share samples
    # with the opposite-half template to which it is compared.
    halves = [[e for e in halves[0] if e["epoch_stop_sample"] <= split],
              [e for e in halves[1] if e["epoch_start_sample"] >= split]]
    if any(len(events) < POLICY["minimum_training_events_per_half"] for events in halves):
        outcome.update(score=None, reason="跨时段模板的独立训练事件不足，每半段需要至少 10 个")
        return outcome
    metrics["impact_training_event_ids"] = [[e["event_id"] for e in group] for group in halves]
    metrics["impact_boundary_excluded_event_ids"] = []
    inspected, evidence, invalid = [], [], []
    for ci, cid in enumerate(f["channel_ids"]):
        if cid not in candidates:
            continue
        templates = [np.median([data[ci, e["epoch_start_sample"]-origin:e["epoch_stop_sample"]-origin]
                                for e in group], axis=0) for group in halves]
        for event in complete:
            a, b = event["epoch_start_sample"], event["epoch_stop_sample"]
            target_half = 0 if b <= split else 1 if a >= split else None
            if target_half is None:
                metrics["impact_boundary_excluded_event_ids"].append(event["event_id"])
                continue
            epoch = np.asarray(data[ci, a-origin:b-origin], dtype=float)
            template = np.asarray(templates[1-target_half], dtype=float)
            if not np.isfinite(epoch).all() or not np.isfinite(template).all() or np.ptp(epoch) <= 0:
                invalid.append({"event_id": event["event_id"], "channel_id": cid})
                continue
            centered = template-template.mean()
            energy = np.dot(centered, centered)
            if energy == 0:
                correlation, ratio = None, 0.
            else:
                coefficient = np.dot(epoch-epoch.mean(), centered)/energy
                ratio = max(0., float(coefficient))*float(np.ptp(template)/np.ptp(epoch))
                correlation = _correlation(epoch, template)
            flagged = correlation is not None and correlation >= POLICY["shape_correlation_min"] and ratio >= POLICY["projected_amplitude_ratio_min"]
            item = {"kind": f"{kind}_impact_window", "channel_id": cid, "raw_name": f["channel_names"][ci],
                    "event_id": event["event_id"], "sample": event["sample"],
                    "time_seconds": f["start_seconds"]+(event["sample"]-origin)/f["sampling_rate_hz"],
                    "start_sample": a, "stop_sample": b,
                    "start_seconds": f["start_seconds"]+(a-origin)/f["sampling_rate_hz"],
                    "stop_seconds": f["start_seconds"]+(b-origin)/f["sampling_rate_hz"],
                    "template_half": 1-target_half, "shape_correlation": correlation,
                    "projected_amplitude_ratio": ratio, "flagged": bool(flagged),
                    "reason": f"{kind.upper()} 锁时影响窗"}
            inspected.append(item)
            if flagged:
                evidence.append(item)
    metrics.update(impact_observations=inspected, impact_invalid_windows=invalid,
                   impact_boundary_excluded_event_ids=sorted(set(metrics["impact_boundary_excluded_event_ids"])))
    if invalid or metrics["impact_boundary_excluded_event_ids"]:
        # These windows have not been classified; their duration stays explicit.
        excluded_ids = {x["event_id"] for x in invalid} | set(metrics["impact_boundary_excluded_event_ids"])
        unassessed = time_burden(f, [(e["epoch_start_sample"], e["epoch_stop_sample"]) for e in complete if e["event_id"] in excluded_ids])
    else:
        unassessed = time_burden(f, [])
    metrics["impact_unassessed_time"] = unassessed
    if unassessed["fraction"] > .05:
        outcome.update(score=None, reason="逐事件波形可评覆盖不足 95%", evidence=evidence)
        return outcome
    burden = time_burden(f, [(e["start_sample"], e["stop_sample"]) for e in evidence])
    metrics["time_burden"] = burden
    metrics["impact_affected_event_count"] = len({e["event_id"] for e in evidence})
    metrics["impact_complete_event_count"] = len(complete)
    metrics["impact_epoch_offsets_samples"] = list(next(iter(offsets)))
    points = time_points(burden["fraction"])
    outcome.update(score=points, evidence=evidence, components=[{"id": "time_burden", "deduction": 10-points}],
                   reason=f"{kind.upper()} 异常事件窗合并 {burden['affected_seconds']:g} / {burden['total_seconds']:g} s，"
                          f"时间占比 {percent(burden['fraction'])}；时长档扣 {10-points} 分")
    return outcome
