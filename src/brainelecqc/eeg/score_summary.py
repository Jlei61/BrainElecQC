"""Integer percentage score with an explicit evaluated-feature profile."""


def attach_score_summary(review):
    """Normalize evaluated weights, keeping unmeasured features explicitly empty.

    The per-feature rubric is unchanged. Half-up integer rounding uses exact
    integer arithmetic. Human review adjusts evaluated feature scores under the
    same rubric; new measurements create a new automatic assessment.
    """
    rows = review["rows"]
    evaluated = [r for r in rows if r["auto_points"] is not None]
    weight = sum(r["max_points"] for r in evaluated)
    points = sum(r["auto_points"] for r in evaluated)
    total_weight = sum(r["max_points"] for r in rows)
    for row in rows:
        available = row["auto_points"] is not None
        row.update(assessment_status="scored" if available else "unavailable",
                   human_adjustable=available)
    review.update(
        score_100=(200*points+weight)//(2*weight) if weight else None,
        score_max=100,
        assessment_coverage={
            "evaluated_weight": weight, "total_weight": total_weight,
            "evaluated_dimensions": [r["dimension_id"] for r in evaluated],
            "unavailable_dimensions": [r["dimension_id"] for r in rows if r["auto_points"] is None],
        },
        aggregation_policy={
            "version": "eeg-evaluated-percentage-1.0",
            "formula": "round_half_up(100 * sum(evaluated points) / sum(evaluated weights))",
            "unavailable_features": "excluded from numerator and denominator; retained in coverage",
            "comparison": "match evaluated dimensions, rubric, detector configuration and recording conditions",
            "human_review": "verify or adjust existing automatic feature scores; rerun for new measurements",
        },
        main_deductions=[{
            "dimension_id": r["dimension_id"], "label": r["label"],
            "points": r["auto_points"], "weight": r["max_points"],
            "deduction": r["auto_deduction"], "reason": r["score_reason"],
        } for r in sorted(evaluated, key=lambda r: -r["auto_deduction"]) if r["auto_deduction"]],
    )
    return review
