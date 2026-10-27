"""Render external numeric/engineering acceptance evidence without source names."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_acceptance import configure, require_external, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-report", required=True)
    parser.add_argument("--checks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.reader_report).read_text())
    checks = json.loads(Path(args.checks).read_text())
    assert report["passed"] and report["source_stat_unchanged"]
    configure()
    output = require_external(args.output)
    fig = plt.figure(figsize=(13, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 2.5], hspace=.5, wspace=.35)
    summary = fig.add_subplot(grid[0, :]); summary.axis("off")
    summary.text(0, 1.05, "BrainElecQC · acceptance evidence", fontsize=20, weight="bold")
    summary.text(0, .75, "Python tests, independent numerical comparison and source preservation", fontsize=11, color="#666666")
    for x, number, label in [(0, checks["python_passed"], "Python tests passed"),
                             (.35, report["total_digital_samples_compared"], "Digital values: exact match"),
                             (.72, checks["source_files_unchanged"], "Source code files unchanged")]:
        summary.text(x, .25, f"{number:,}", fontsize=28, weight="bold", color="#166b72")
        summary.text(x, -.04, label, fontsize=10)
    error_ax, time_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    errors = [max(w["max_abs_physical_error_volts"] for w in f["windows"]) for f in report["files"]]
    error_ax.scatter(range(1, len(errors)+1), np.maximum(errors, 1e-20), s=50, color="#166b72")
    error_ax.axhline(1e-15, color="#b86b16", linestyle="--", label="Absolute tolerance: 1e-15 V")
    error_ax.set(yscale="log", ylim=(1e-21,1e-14), xlabel="Anonymized source file", ylabel="Maximum physical-value difference (V)", title="a  Independent byte-decoder comparison")
    error_ax.set_xticks(range(1,len(errors)+1)); error_ax.legend(fontsize=8,loc="upper left")
    for i, error in enumerate(errors):
        if error == 0:
            error_ax.annotate("exact 0", (i+1, 1e-20), xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    example = max(report["files"], key=lambda f: len(f["segments"][0]))
    spans = example["segments"][0]
    origin = spans[0]["recording_start_seconds"]
    # The source oracle carries sample bounds and source-record bounds, allowing
    # duration recovery from the independent report's verified header timebase.
    seconds_per_sample = checks["segmented_example_sample_period_seconds"]
    for i, span in enumerate(spans):
        start = span["recording_start_seconds"]-origin
        duration = (span["stop_sample"]-span["start_sample"])*seconds_per_sample
        time_ax.broken_barh([(start,duration)], (0,1), facecolors="#166b72")
        time_ax.text(start+duration/2,1.05,str(i+1),ha="center",fontsize=9)
    time_ax.set(ylim=(-.5,1.7), yticks=[], xlabel="Elapsed source-clock time (s)", title=f"b  Segmented SEEG: {len(spans)} spans; gaps preserved")
    time_ax.text(.01,.05,"Blank intervals are acquisition gaps.\nSegments retain their original sample times.",transform=time_ax.transAxes,fontsize=9)
    fig.text(.07,.025,"Source names, paths, waveforms and all acceptance outputs stay outside Git. Public repository contains source code, tests and generic documentation.",fontsize=10)
    fig.subplots_adjust(top=.9,bottom=.15,left=.08,right=.98)
    save(fig, output, "acceptance-summary")


if __name__ == "__main__":
    main()
