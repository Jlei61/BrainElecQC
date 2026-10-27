"""Raw continuity evidence for reviewing short constant-value runs."""
from numbers import Integral

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..eeg.binding import validate_result_identity
from .eeg import _style


def make_continuity_review_figure(result, channel_indices, *, start_seconds,
                                 duration_seconds=2, min_constant_seconds=.1,
                                 title="短平线与掉线复核"):
    """Show raw samples and adjacent differences; return exact run coordinates.

    Shading denotes identical finite raw values lasting at least the specified
    display duration. This observation remains separate from saved detector
    candidates and review decisions. Ranges use the original half-open clock.
    """
    validate_result_identity(result)
    features, data = result["features"], np.asarray(result["raw_data_volts"])
    indices = list(channel_indices)
    if (not 1 <= len(indices) <= 6 or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or not 0 <= i < len(data) for i in indices)):
        raise ValueError("Choose one to six distinct EEG channel indices")
    if (not np.isfinite([start_seconds, duration_seconds, min_constant_seconds]).all()
            or duration_seconds <= 0 or min_constant_seconds <= 0):
        raise ValueError("Display interval and minimum constant duration must be finite and positive")
    fs, origin = features["sampling_rate_hz"], features["start_seconds"]
    left, right = round((start_seconds-origin)*fs), round((start_seconds+duration_seconds-origin)*fs)
    if not 0 <= left < right <= data.shape[1] or right-left < 2:
        raise ValueError("Choose a display interval inside the input with at least two samples")

    def span(a, b):
        return {"start_sample": int(features["start_sample"]+a),
                "stop_sample": int(features["start_sample"]+b),
                "start_seconds": origin+a/fs, "stop_seconds": origin+b/fs}

    runs = []
    for i in indices:
        y = data[i]
        equal = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (y[:-1] == y[1:])
        transitions = np.diff(np.r_[False, equal, False].astype(int))
        for a, last in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            b = last+1
            if b-a < max(2, int(np.ceil(min_constant_seconds*fs))) or b <= left or a >= right:
                continue
            runs.append({"channel_index": int(i), "channel_id": features["channel_ids"][i],
                         "channel_name": features["channel_names"][i], **span(a, b),
                         "sample_count": int(b-a), "duration_seconds": (b-a)/fs,
                         "value_uv": float(y[a]*1e6), "observation": "identical_raw_samples"})
    bad = result["method_results"]["bad_channels"]
    metadata = {"data_sha256": features["data_sha256"], "sampling_rate_hz": fs,
                "display": span(left, right), "channel_indices": [int(i) for i in indices],
                "min_constant_seconds": min_constant_seconds, "constant_runs": runs,
                "saved_continuity_candidates": [c for c in bad["candidates"]
                    if c["channel_index"] in indices and c["kind"] in
                    {"intermittent_dropout", "flat_signal", "nonfinite_input"}],
                "comparison_groups": bad["groups"], "review_status": "pending_review"}
    values = data[indices, left:right]*1e6
    differences = np.diff(values, axis=1)
    finite = values[np.isfinite(values)]
    finite_diff = differences[np.isfinite(differences)]
    limit = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1)*1.08
    diff_limit = max(float(np.max(np.abs(finite_diff))) if len(finite_diff) else 0, .01)*1.08
    times = origin+np.arange(left, right)/fs
    names = features["channel_names"]
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(len(indices), 2, squeeze=False, figsize=(11, 2.1*len(indices)+.6),
                                 layout="constrained")
        for row, i in enumerate(indices):
            name = f"{names[i]} [row {i}]" if names.count(names[i]) > 1 else names[i]
            for col, y in enumerate((values[row], differences[row])):
                ax = axes[row, col]
                ax.set_gid(f"continuity:{'raw' if col == 0 else 'difference'}:{i}")
                ax.plot(times if col == 0 else times[1:], y, color="#277898", lw=.85)
                for run in runs:
                    if run["channel_index"] == i:
                        ax.axvspan(max(run["start_seconds"], times[0]),
                                   min(run["stop_seconds"], origin+right/fs), color="#D39643", alpha=.2)
                ax.axhline(0, color="#888888", ls=":", lw=.7)
                bound = limit if col == 0 else diff_limit
                ax.set(xlim=(times[0], origin+right/fs), ylim=(-bound, bound), ylabel="µV")
                ax.set_title(name, loc="left", fontsize=9)
                if row == len(indices)-1:
                    ax.set_xlabel("原始时间 (s)")
                else:
                    ax.tick_params(labelbottom=False)
        first_name = axes[0, 0].get_title(loc="left")
        axes[0, 0].set_title(f"a  原始波形 · 同一幅度刻度\n{first_name}", loc="left", fontweight="bold")
        axes[0, 1].set_title(f"b  相邻样本电压差\n{first_name}", loc="left", fontweight="bold")
        from matplotlib.patches import Patch
        axes[-1, 0].legend(handles=[Patch(facecolor="#D39643", alpha=.2,
            label=f"原始值连续相同 ≥ {min_constant_seconds:g} s")], loc="lower left", fontsize=8)
        fig.suptitle(title, x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig, metadata
