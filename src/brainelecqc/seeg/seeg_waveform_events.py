"""Joint shape candidates from fixed-time summaries, with explicit recovery.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import hashlib
import math

import numpy as np


WAVEFORM_VERSION = "seeg-waveform-1"
WAVEFORM_DEFAULTS = {
    "waveform_bin_seconds": .02, "waveform_baseline_seconds": 1.0,
    "waveform_min_event_seconds": .3, "waveform_jump_min_v": 150e-6,
    "waveform_relative_multiplier": 8.0, "waveform_scale_floor_v": 5e-6,
    "waveform_drift_min_v": 300e-6, "waveform_drift_seconds": 1.0,
    "waveform_long_drift_min_v": .001,
    "waveform_plateau_seconds": .15, "waveform_recovery_seconds": 1.0,
    "waveform_recovery_ratio": .2, "waveform_recovery_abs_v": 30e-6,
    "waveform_recovery_scale_multiplier": 2.0, "max_waveform_bins": 1000000,
    "max_waveform_events": 10000,
}
# Summary columns: half-open sample interval, median, within-bin MAD, strongest
# signed adjacent-sample jump and its actual later-sample position.
START, STOP, LEVEL, MAD, JUMP, JUMP_AT, FIRST, LAST, PREFIX, SUFFIX = range(10)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _line(values):
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    y = values - values.mean()
    denominator = float(np.dot(x, x))
    slope = float(np.dot(x, y) / denominator) if denominator else 0.0
    variation = float(np.dot(y, y))
    r2 = 1 - float(np.sum((y - slope * x) ** 2)) / variation if variation else 0.0
    return slope, r2


def _seed(summary, i, cfg, nb, ne, nd, np_):
    level = summary[:, LEVEL]
    before = level[i - nb:i]
    base = float(np.median(before))
    scale = max(cfg["waveform_scale_floor_v"], 1.4826 * float(np.median(np.abs(before - base))),
                1.4826 * float(np.median(summary[i - nb:i, MAD])))
    threshold = max(cfg["waveform_jump_min_v"], cfg["waveform_relative_multiplier"] * scale)
    after = level[i:i + ne]
    shift = float(np.median(after) - base)
    short_change = float(np.median(level[i:i + 2]) - np.median(level[max(0, i - 2):i]))
    jump_index = i + int(np.argmax(np.abs(summary[i:min(len(summary), i + 2), JUMP])))
    actual_jump = float(summary[jump_index, JUMP])
    sustained = len(after) == ne and np.mean(np.sign(shift) * (after - base) >= abs(shift) * .6) >= .9
    if (sustained and abs(shift) >= threshold and abs(short_change) >= threshold * .7
            and abs(actual_jump) >= threshold * .5):
        return {"i": i, "sample": int(summary[jump_index, JUMP_AT]), "baseline": base, "scale": scale,
                "jump": actual_jump, "kind": "jump", "reason": "sustained_level_shift", "initial_end": i + ne}
    if i + nd + np_ > len(level):
        return None
    ramp, plateau = level[i:i + nd], level[i + nd:i + nd + np_]
    excursion = float(np.median(plateau) - base)
    if abs(excursion) < max(cfg["waveform_drift_min_v"], cfg["waveform_relative_multiplier"] * scale):
        return None
    slope, r2 = _line(ramp)
    tail_slope, _ = _line(ramp[-max(3, nd // 5):])
    plateau_slope, _ = _line(plateau)
    stable_plateau = np.ptp(plateau) <= max(4 * scale, .08 * abs(excursion))
    if (r2 >= .98 and abs(slope) > 0 and np.sign(slope) == np.sign(excursion)
            and abs(tail_slope) >= .65 * abs(slope) and abs(plateau_slope) <= .1 * abs(slope)
            and stable_plateau):
        return {"i": i, "sample": int(summary[i, START]), "baseline": base, "scale": scale, "jump": 0.0,
                "kind": "drift", "reason": "linear_drift_with_plateau", "initial_end": i + nd + np_}
    # A long, nearly linear unresolved displacement may end before a plateau.
    # Smooth oscillatory turning points fail this shape/persistence conjunction.
    long_stop = i + 2 * nd
    if long_stop <= len(level):
        long = level[i:long_stop]
        long_slope, long_r2 = _line(long)
        if (long_r2 >= .997 and abs(long[-1] - long[0]) >= max(cfg["waveform_long_drift_min_v"], 2 * cfg["waveform_relative_multiplier"] * scale)
                and np.sign(long_slope) == np.sign(excursion)):
            return {"i": i, "sample": int(summary[i, START]), "baseline": base, "scale": scale,
                    "jump": 0.0, "kind": "drift", "reason": "sustained_linear_drift", "initial_end": long_stop}
    return None


def detect_events(summary, frequency, context, config, termination):
    if not len(summary):
        return []
    dt = max(1, round(frequency * config["waveform_bin_seconds"])) / frequency
    count = lambda key: max(2, int(math.ceil(config[key] / dt)))
    nb, ne, nd, np_, nr = [count(k) for k in ("waveform_baseline_seconds", "waveform_min_event_seconds",
                                            "waveform_drift_seconds", "waveform_plateau_seconds", "waveform_recovery_seconds")]
    level = summary[:, LEVEL]
    # Cheap absolute gates keep detailed fits away from normal bins. The final
    # decision still requires local robust scale, persistence and joint shape.
    jump_candidates = np.flatnonzero(np.abs(summary[:, JUMP]) >= config["waveform_jump_min_v"] * .5)
    drift_candidates = np.flatnonzero(np.abs(level[nd:] - level[:-nd]) >= config["waveform_drift_min_v"]) if len(level) > nd else []
    candidate_indices = sorted(set(int(i) for j in jump_candidates for i in (j - 1, j)) | set(map(int, drift_candidates)))
    events, cursor = [], nb
    for i in candidate_indices:
        if i < cursor or i < nb or i + ne > len(level):
            continue
        seed = _seed(summary, i, config, nb, ne, nd, np_)
        if seed is None:
            continue
        if seed["kind"] == "jump":
            # A sustained event can begin with a smooth displacement before its
            # largest edge. Trace only an uninterrupted same-direction offset;
            # a preceding spike that returned to baseline remains separate.
            original = seed["i"]
            direction = np.sign(np.median(level[i:seed["initial_end"]]) - seed["baseline"])
            threshold = max(config["waveform_jump_min_v"] * .5, 3 * seed["scale"])
            while i > max(cursor, original - nb) and direction * (level[i - 1] - seed["baseline"]) >= threshold:
                i -= 1
            if i < original:
                seed.update(i=i, jump_sample=seed["sample"], sample=int(summary[i, START]))
        # Recovery is relative to the pre-event baseline; following a shifted
        # level with a rolling baseline would falsely declare recovery.
        initial = level[i:seed["initial_end"]]
        peak = max(abs(float(np.max(initial) - seed["baseline"])), abs(float(np.min(initial) - seed["baseline"])))
        tolerance = _recovery_tolerance(peak, seed["scale"], config)
        near = np.abs(level - seed["baseline"]) <= tolerance
        recovered = None
        for a, b in _runs(near):
            if a < seed["initial_end"] or b - a < nr:
                continue
            target = summary[a, START] + math.ceil(config["waveform_recovery_seconds"] * frequency)
            end = int(np.searchsorted(summary[:, STOP], target, side="left")) + 1
            if end <= b:
                recovered = (a, end)
                break
        end = recovered[1] if recovered else len(level)
        event = _event(summary, seed, end, recovered, frequency, context, config, termination, np_)
        events.append(event)
        cursor = end + nb
        if len(events) > config["max_waveform_events"]:
            raise ValueError("Waveform event limit exceeded; select a shorter scan range")
    return events


def _plateaus(summary, first, end, base, scale, peak, frequency, config, np_):
    exact, pending = [], None
    for row in summary[first:end]:
        start, stop = int(row[START]), int(row[STOP])
        prefix, suffix = int(row[PREFIX]), int(row[SUFFIX])
        pieces = [(start, start + prefix, row[FIRST])]
        if prefix < stop - start:
            pieces.append((stop - suffix, stop, row[LAST]))
        for a, b, value in pieces:
            if pending and pending[1] == a and pending[2] == value:
                pending = (pending[0], b, value)
            else:
                if pending:
                    exact.append(pending)
                pending = (a, b, value)
    if pending:
        exact.append(pending)
    result = [{"start": a, "stop": b, "level": float(value), "range": 0.0, "method": "exact_constant_run"}
              for a, b, value in exact if (b - a) / frequency >= config["waveform_plateau_seconds"]
              and abs(value - base) >= max(3 * scale, peak * .25)]
    # Approximate plateaus retain one fixed level anchor. Moving-window unions
    # otherwise turn a long smooth recovery slope into a fictitious plateau.
    a = first
    limit = max(2 * scale, peak * .002)
    while a + np_ <= end:
        values = summary[a:a + np_, LEVEL]
        slope, _ = _line(values)
        if (np.ptp(values) > limit or np.min(np.abs(values - base)) < max(3 * scale, peak * .25)
                or abs(slope) * np_ > scale * .25):
            a += 1
            continue
        anchor = float(np.median(values))
        b = a + np_
        while b < end and abs(summary[b, LEVEL] - anchor) <= limit * .5:
            b += 1
        start, stop = int(summary[a, START]), int(summary[b - 1, STOP])
        overlaps_exact = any(item["start"] < stop and item["stop"] > start for item in result)
        if not overlaps_exact:
            result.append({"start": start, "stop": stop, "level": anchor,
                           "range": float(np.ptp(summary[a:b, LEVEL])), "method": "anchored_low_variability"})
        a = b
    return sorted(result, key=lambda item: item["start"])


def _event(summary, seed, end, recovered, frequency, context, config, termination, np_):
    i, base, scale = seed["i"], seed["baseline"], seed["scale"]
    values = summary[i:end, LEVEL]
    peak = float(np.max(np.abs(values - base)))
    plateaus = _plateaus(summary, i, end, base, scale, peak, frequency, config, np_)
    stages = []
    if seed["kind"] == "jump":
        jump_sample = seed.get("jump_sample", seed["sample"])
        if jump_sample > seed["sample"]:
            stages.append(_stage("drift", seed["sample"], jump_sample, context, frequency,
                                 {"direction": "away_from_baseline", "method": "contiguous_offset_backtrace"}))
        stages.append(_stage("jump", jump_sample, jump_sample + 1, context, frequency,
                             {"jump_v": seed["jump"]}))
    else:
        drift_stop = plateaus[0]["start"] if plateaus else int(summary[end - 1, STOP])
        stages.append(_stage("drift", seed["sample"], max(seed["sample"] + 1, drift_stop), context, frequency,
                             {"net_change_v": float(summary[min(seed["initial_end"], end) - 1, LEVEL] - base)}))
    for plateau in plateaus:
        stages.append(_stage("plateau", max(seed["sample"], plateau["start"]), plateau["stop"], context, frequency,
                             {"level_v": plateau["level"], "range_v": plateau["range"], "method": plateau["method"]}))
    if recovered:
        recovery_start = int(summary[recovered[0], START])
        if plateaus and plateaus[-1]["stop"] < recovery_start:
            stages.append(_stage("drift", plateaus[-1]["stop"], recovery_start, context, frequency,
                                 {"direction": "toward_baseline"}))
        stages.append(_stage("recovery", recovery_start,
                             int(summary[recovered[1] - 1, STOP]), context, frequency,
                             {"confirmation_start_sample": int(summary[recovered[0], START]),
                              "confirmed_seconds": config["waveform_recovery_seconds"]}))
    elif plateaus:
        drift_start = plateaus[-1]["stop"]
        stop = int(summary[end - 1, STOP])
        if stop - drift_start >= max(2, round(config["waveform_plateau_seconds"] * frequency)):
            stages.append(_stage("drift", drift_start, stop, context, frequency,
                                 {"direction": "unresolved_baseline_shift", "right_censored": True}))
    start, stop = seed["sample"], int(summary[end - 1, STOP])
    identity = "{}:{}:{}:{}:{}".format(context["stream_id"], context["segment_id"], context["channel_id"], start, stop)
    return {"event_id": "wave-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            **{k: context[k] for k in ("channel_id", "stream_id", "segment_id")},
            "start_sample": start, "stop_sample": stop, "recording_start_seconds": _time(start, context, frequency),
            "recording_stop_seconds": _time(stop, context, frequency), "duration_seconds": (stop - start) / frequency,
            "stages": stages, "recovery_status": "confirmed" if recovered else "right_censored",
            "termination_reason": "recovered" if recovered else termination, "score_effect": "none", "status": "candidate",
            "features": {"baseline_v": base, "robust_scale_v": scale, "peak_offset_v": peak, "jump_v": seed["jump"],
                         "plateau_seconds": sum(stage["duration_seconds"] for stage in stages if stage["kind"] == "plateau"),
                         "recovery_confirmation_seconds": config["waveform_recovery_seconds"],
                         "recovery_tolerance_v": _recovery_tolerance(peak, scale, config),
                         "detection_bin_samples": max(1, round(frequency * config["waveform_bin_seconds"]))},
            "reasons": [seed["reason"]] + (["low_variability_plateau"] if plateaus else [])
                       + ["baseline_recovery_confirmed" if recovered else "recovery_not_observed"]}


def _time(sample, context, frequency):
    return context["recording_start_seconds"] + (sample - context["segment_start_sample"]) / frequency


def _recovery_tolerance(peak, scale, config):
    baseline = max(config["waveform_recovery_abs_v"], config["waveform_recovery_scale_multiplier"] * scale)
    return max(config["waveform_scale_floor_v"], min(baseline, config["waveform_recovery_ratio"] * peak))


def _stage(kind, start, stop, context, frequency, features):
    return {"kind": kind, "start_sample": start, "stop_sample": stop,
            "recording_start_seconds": _time(start, context, frequency), "recording_stop_seconds": _time(stop, context, frequency),
            "duration_seconds": (stop - start) / frequency, "features": features}
