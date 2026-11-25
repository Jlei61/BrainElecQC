"""Transparent, uncalibrated SEEG technical-risk rules; no physiology penalty.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import math


RULE_VERSION = "seeg_basic_v2"
PENALTIES = {"nonfinite": 100, "flat": 100, "clipping": 80, "line_noise": 35}


def score_window(flags, eligible=True):
    """Related observations share a window: only the strongest deduction applies."""
    return 100 - max((PENALTIES.get(flag, 0) for flag in flags), default=0) if eligible else None


def compute_quality_score(windows, decisions=None):
    """Duration-weighted channel-window score; missing indicators stay missing.

    Review decisions are optional: accepted confirms technical exclusion (0),
    rejected dismisses the automatic candidate (100), pending keeps initial risk.
    This aggregate does not replace the service's exact reviewed interval masks.
    """
    decisions = decisions or {}
    if not isinstance(decisions, dict):
        raise ValueError("decisions must map window_id to a decision")
    known = {w["window_id"] for w in windows}
    if set(decisions) - known:
        raise ValueError("Decision references unknown window_id")
    weighted, assessed, unassessed, reviewed = 0.0, 0.0, 0.0, 0.0
    components = {flag: {"flag": flag, "penalty": penalty, "window_count": 0, "channel_seconds": 0.0,
                         "evaluated_channel_seconds": 0.0, "unevaluated_channel_seconds": 0.0}
                  for flag, penalty in PENALTIES.items()}
    for window in windows:
        duration = window["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Window duration must be finite and positive")
        for flag in set(window.get("flags", [])) & set(components):
            components[flag]["window_count"] += 1
            components[flag]["channel_seconds"] += duration
        metrics = window.get("metrics", {})
        evaluated = {"nonfinite": metrics.get("finite_fraction") is not None,
                     "flat": metrics.get("flat_fraction") is not None,
                     "clipping": (metrics.get("clipping_status") == "computed"
                                  and metrics.get("adc_range_status") == "confirmed"),
                     "line_noise": metrics.get("line_noise_db") is not None}
        for flag, component in components.items():
            key = "evaluated_channel_seconds" if evaluated[flag] else "unevaluated_channel_seconds"
            component[key] += duration
        score = window.get("score") if window.get("score_eligible", True) else None
        decision = decisions.get(window["window_id"], "pending")
        if isinstance(decision, dict):
            decision = decision.get("decision", decision.get("status"))
        if decision not in ("accepted", "rejected", "pending"):
            raise ValueError("Unsupported review decision")
        if score is None:
            unassessed += duration
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError("Score must be in [0,100] or null")
        if decision != "pending":
            score = 0 if decision == "accepted" else 100
            reviewed += duration
        assessed += duration
        weighted += score * duration
    for component in components.values():
        component["status"] = "computed" if component["evaluated_channel_seconds"] else "insufficient_data"
    return {"total": int(math.floor(weighted / assessed + .5)) if assessed else None,
            "components": list(components.values()), "rule_version": RULE_VERSION, "calibrated": False,
            "aggregation": "duration_weighted_channel_windows_max_penalty",
            "assessed_channel_seconds": assessed, "unassessed_channel_seconds": unassessed,
            "reviewed_channel_seconds": reviewed}
