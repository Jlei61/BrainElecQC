"""Local CLI. Import never mutates the supplied source files."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile


def _format_hints(values, files):
    """Resolve hints against snapshot-relative names, never absolute host paths."""
    from .readers import FORMAT_CAPABILITIES
    from .storage import inventory, safe_relative
    if not values:
        return {}
    paths = [Path(value).expanduser() for value in files]
    members = set()
    for path in paths:
        if path.is_dir():
            members.update(p.relative_to(path).as_posix() for p in inventory(path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members.update(m.filename for m in archive.infolist() if not m.is_dir())
        else:
            members.add(path.name)
    hints = {}
    for value in values:
        if "=" in value:
            name, kind = value.split("=", 1)
        elif len(paths) == 1 and paths[0].is_file() and paths[0].suffix.lower() != ".zip":
            name, kind = paths[0].name, value
        else:
            raise ValueError("For directories, archives or multiple files use --format-hint RELATIVE_PATH=FORMAT")
        safe_relative(name)
        if name not in members:
            raise ValueError(f"Format hint does not identify an imported file: {name}")
        if kind not in FORMAT_CAPABILITIES:
            raise ValueError("Unknown format hint; choose " + ", ".join(FORMAT_CAPABILITIES))
        if name in hints and hints[name] != kind:
            raise ValueError(f"Conflicting format hints for {name}")
        hints[name] = kind
    return hints


def _outside_checkout(path):
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Choose a storage or output path outside Git checkouts")
    return path


def _new_output(path):
    path = _outside_checkout(path)
    if path.exists():
        raise ValueError("Choose a new output path to preserve the existing QC result")
    return path


def _write_seeg_report(output, run):
    template = {"base_revision": run["review"]["revision"], "reviewer": "", "decisions": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".seeg-qc-", dir=output.parent) as scratch:
        staging = Path(scratch) / "report"
        staging.mkdir()
        for name, payload in (("qc_result.json", run), ("review.json", template)):
            (staging / name).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                        encoding="utf-8")
        staging.rename(output)


def _seeg_command(args):
    from .config import settings
    from .db import init_db
    from .recordings import import_recording
    from .seeg import export_run, freeze_run, get_run, run_seeg_qc, save_review

    _outside_checkout(settings.storage_root)
    if args.command == "qc-seeg":
        if bool(args.files) == bool(args.artifact_id):
            raise ValueError("Select input files or --artifact-id")
        if args.artifact_id and args.format_hint:
            raise ValueError("Format hints are used only when importing files")
        config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(config, dict) or set(config) - {"config", "channel_map"}:
            raise ValueError("SEEG configuration accepts config and channel_map fields")
        output = _new_output(args.output)
        if args.files:
            hints = _format_hints(args.format_hint, args.files)
            init_db()
            record, _ = import_recording({
                "modality": "seeg", "label": Path(args.files[0]).name,
                "metadata": {"format_hints": hints} if hints else {},
            }, paths=args.files)
            artifact_id = record["input_artifact_id"]
        else:
            if not settings.db_path.is_file():
                raise ValueError("No database in the selected storage root; import data or select the existing store")
            init_db()
            artifact_id = args.artifact_id
        run = run_seeg_qc(artifact_id, **config)
        _write_seeg_report(output, run)
        return {"run_id": run["run_id"], "status": run["status"], "revision": run["review"]["revision"],
                "output": str(output), "initial_score": run["report"]["initial_score"]["total"]}

    if not settings.db_path.is_file():
        raise ValueError("No database in the selected storage root; select the store for this QC run")
    init_db()
    if args.command == "show-seeg":
        return get_run(args.run_id)
    if args.command == "export-seeg":
        output = _new_output(args.output)
        payload = export_run(args.run_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".seeg-export-", dir=output.parent) as scratch:
            staging = Path(scratch) / "qc.zip"
            staging.write_bytes(payload)
            os.link(staging, output)
        return {"run_id": args.run_id, "output": str(output)}
    if args.command == "review-seeg":
        run = save_review(args.run_id, json.loads(Path(args.review).expanduser().read_text(encoding="utf-8")))
    else:
        run = freeze_run(args.run_id, {"base_revision": args.base_revision, "reviewer": args.reviewer})
    result = {"run_id": run["run_id"], "revision": run["review"]["revision"], "frozen": run["frozen"]}
    if run["frozen"]:
        result["assessment"] = run["assessment"]
    return result


def main():
    parser = argparse.ArgumentParser(prog="python -m brainelecqc")
    parser.add_argument("--storage-root", help="Independent data directory outside the code checkout")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("files", nargs="+")
    ingest.add_argument("--modality", required=True, choices=["eeg", "seeg", "ecog"])
    ingest.add_argument("--label", help="Recording label; defaults to the first input filename")
    ingest.add_argument("--participant-id")
    ingest.add_argument("--format-hint", action="append", default=[], metavar="[RELATIVE_PATH=]FORMAT",
                        help="Explicit vendor format; repeat with relative paths for file bundles")
    commands.add_parser("list")
    qc = commands.add_parser("qc-eeg", help="Compute paper-defined EEG evidence, review candidates and Python figures")
    qc.add_argument("files", nargs="+")
    qc.add_argument("--output", required=True, help="New result directory outside the checkout")
    qc.add_argument("--config", help="JSON object: selection, method_config, feature_config and policy_config")
    qc.add_argument("--eeg-channels", help="Comma-separated source indices, e.g. 0,1,2,3")
    qc.add_argument("--start", type=float, help="Analysis start in the source clock (seconds)")
    qc.add_argument("--duration", type=float, help="Analysis duration (seconds)")
    qc.add_argument("--plot-start", type=float, help="Waveform display start in source clock")
    qc.add_argument("--plot-duration", type=float, default=15)
    qc.add_argument("--title", default="EEG QC")
    qc.add_argument("--format-hint", action="append", default=[], metavar="[RELATIVE_PATH=]FORMAT")
    qc.add_argument("--no-figures", action="store_true", help="Export numerical evidence only")
    seeg = commands.add_parser("qc-seeg", help="Compute and save technical SEEG quality evidence")
    seeg.add_argument("files", nargs="*")
    seeg.add_argument("--artifact-id", help="Scan an existing SEEG input or derived voltage artifact")
    seeg.add_argument("--output", required=True, help="New report directory outside Git checkouts")
    seeg.add_argument("--config", help="JSON object containing config and channel_map")
    seeg.add_argument("--format-hint", action="append", default=[], metavar="[RELATIVE_PATH=]FORMAT")
    show = commands.add_parser("show-seeg", help="Read a saved SEEG scan and its current review")
    show.add_argument("run_id")
    review = commands.add_parser("review-seeg", help="Save a complete decision list at its base revision")
    review.add_argument("run_id")
    review.add_argument("--review", required=True, help="JSON review payload")
    freeze = commands.add_parser("freeze-seeg", help="Freeze and verify the current review")
    freeze.add_argument("run_id")
    freeze.add_argument("--base-revision", required=True, type=int)
    freeze.add_argument("--reviewer", required=True)
    export = commands.add_parser("export-seeg", help="Export an immutable QC ZIP")
    export.add_argument("run_id")
    export.add_argument("--output", required=True, help="New ZIP path outside Git checkouts")
    args = parser.parse_args()
    from .config import settings, storage_context
    try:
        with storage_context(args.storage_root) if args.storage_root else nullcontext():
            from .db import init_db
            from . import recordings
            if args.command == "qc-eeg":
                from .eeg.pipeline import run_eeg_qc, validate_qc_config
                from .eeg.export import export_eeg_qc
                config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
                selection, feature_config, policy_config, method_config = validate_qc_config(config)
                if args.eeg_channels is not None:
                    selection["eeg_indices"] = [int(x) for x in args.eeg_channels.split(",")]
                if args.start is not None:
                    selection["start_seconds"] = args.start
                if args.duration is not None:
                    selection["duration_seconds"] = args.duration
                hints = _format_hints(args.format_hint, args.files)
                init_db()
                record, _ = recordings.import_recording({"modality": "eeg", "label": args.title,
                    "metadata": {"format_hints": hints} if hints else {}}, paths=args.files)
                computed = run_eeg_qc(record["input_artifact_id"], **selection,
                                     feature_config=feature_config, policy_config=policy_config, method_config=method_config)
                result = export_eeg_qc(computed, args.output, figures=not args.no_figures, title=args.title,
                                      display_start_seconds=args.plot_start, display_duration_seconds=args.plot_duration)
            elif args.command == "import":
                hints = _format_hints(args.format_hint, args.files)
                payload = {"modality": args.modality,
                           "label": args.label or Path(args.files[0]).expanduser().name,
                           "participant_id": args.participant_id,
                           "metadata": {"format_hints": hints} if hints else {}}
                init_db()
                record, reused = recordings.import_recording(payload, paths=args.files)
                result = {"recording": record, "reused": reused}
            elif args.command == "list":
                if not settings.db_path.is_file():
                    raise ValueError("No database in the selected storage root; import data or select the existing store")
                init_db()
                result = recordings.list_recordings()
            else:
                result = _seeg_command(args)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, LookupError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"brainelecqc: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
