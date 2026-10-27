"""Local CLI. Import never mutates the supplied source files."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
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
            else:
                if not settings.db_path.is_file():
                    raise ValueError("No database in the selected storage root; import data or select the existing store")
                init_db()
                result = recordings.list_recordings()
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, LookupError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"brainelecqc: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
