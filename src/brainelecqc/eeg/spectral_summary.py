"""Choose traceable display channels and export measured alpha power."""
import csv
import json
from pathlib import Path
import re
from .display import display_channel_names


def occipital_name(raw_name):
    """Recognize named occipital leads, preserving the raw name elsewhere."""
    match = re.match(r"^(?:EEG[\s_-]*)?(O1|Oz|O2)(?=$|[ ._-])", raw_name, re.I)
    return match.group(1).lower() if match else None


def select_spectral_channels(result, count=6):
    """Occipital leads, strongest flagged noise/deviation, then source order."""
    names = result["features"]["channel_names"]
    selected = {}

    def add(index, reason):
        if len(selected) < count:
            selected.setdefault(index, reason)

    for lead in ("o1", "oz", "o2"):
        for index, name in enumerate(names):
            if occipital_name(name) == lead:
                add(index, "枕区通道")
    bad = result["method_results"]["bad_channels"]["channels"]
    for flag, field, reason in (("high_frequency_noise", "hf_z", "高频噪声候选"),
                                ("amplitude_deviation", "amplitude_z", "幅度离群候选")):
        candidates = [r for r in bad if flag in r["flags"] and r.get(field) is not None]
        if candidates:
            row = max(candidates, key=lambda r: abs(r[field]) if field == "amplitude_z" else r[field])
            add(row["channel_index"], reason)
    line = [r for r in result["method_results"]["spectral"]["line_noise"]["observations"] if r["flagged"]]
    if line:
        row = max(line, key=lambda r: float("inf") if r["f_is_infinite"] else r["f_statistic"])
        add(row["channel_index"], "工频候选")
    for index in range(len(names)):
        add(index, "按原始顺序补充")
    return [{"channel_index": i, "channel_id": result["features"]["channel_ids"][i],
             "channel_name": names[i], "selection_reason": reason,
             "display_name": display_channel_names(names)[i]}
            for i, reason in selected.items()]


def spectral_display_rows(result):
    spectral = result["method_results"]["spectral"]
    if "alpha_power" not in spectral:
        return []
    rows = []
    for selected in select_spectral_channels(result):
        i = selected["channel_index"]
        fit = spectral["parameterization"]["full"][i]
        peaks = fit["alpha_peaks"]
        peak = max(peaks, key=lambda p: p["height_db"]) if peaks else None
        rows.append({**spectral["alpha_power"]["full"][i], **selected,
                     "fitted_alpha_peak_hz": peak["center_frequency_hz"] if peak else None})
    return rows


def write_alpha_files(result, output):
    """Write the same physical measurements used by figures and the review."""
    block = result["method_results"]["spectral"].get("alpha_power")
    if block is None:
        return
    output = Path(output)
    fields = ["scope", "channel_index", "channel_id", "channel_name", "window_index",
              "start_sample", "stop_sample", "start_seconds", "stop_seconds",
              "alpha_power_uv2", "reference_power_uv2", "relative_alpha_percent", "status",
              "alpha_power_status", "reference_power_status", "relative_alpha_status"]
    with (output/"alpha_power.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for scope in ("full", "windows"):
            writer.writerows({**row, "scope": scope} for row in block[scope])
    display = {"selection": "O1/Oz/O2, strongest flagged high-frequency/amplitude/line candidates, source order",
               "alpha_band_hz": block["alpha_band_hz"], "reference_band_hz": block["reference_band_hz"],
               "rows": spectral_display_rows(result)}
    (output/"spectral_display.json").write_text(json.dumps(display, ensure_ascii=False, indent=2,
                                                        allow_nan=False)+"\n", encoding="utf-8")
