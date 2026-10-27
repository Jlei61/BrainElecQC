"""Channel prevalence and original-clock interval burden for review scoring."""


TIME_POLICY = {"fraction_edges": [.02, .05, .10, .25, .50], "points": [10, 9, 8, 7, 5, 3, 0],
               "basis": "union_across_channels_in_original_samples"}
PROPORTION_POINTS = {5: [5, 4, 4, 3, 2, 1, 0], 10: TIME_POLICY["points"],
                     20: [20, 19, 18, 16, 12, 6, 0]}


def proportion_points(fraction, maximum):
    return PROPORTION_POINTS[maximum][0 if fraction == 0 else
        1+sum(fraction >= e for e in TIME_POLICY["fraction_edges"])]


def union_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def time_burden(features, intervals):
    start, stop = features["start_sample"], features["stop_sample"]
    intervals = list(intervals)
    if any(not start <= a < b <= stop for a, b in intervals):
        raise ValueError("Impact intervals must lie within the original sample range")
    merged = union_intervals(intervals)
    samples = sum(b-a for a, b in merged)
    fs = features["sampling_rate_hz"]
    return {"affected_samples": samples, "total_samples": stop-start,
            "affected_seconds": samples/fs, "total_seconds": (stop-start)/fs,
            "fraction": samples/(stop-start), "intervals": merged,
            "basis": TIME_POLICY["basis"]}


def time_points(fraction):
    return proportion_points(fraction, 10)


def percent(fraction):
    return "<0.1%" if 0 < fraction < .001 else f"{fraction*100:.1f}%"
