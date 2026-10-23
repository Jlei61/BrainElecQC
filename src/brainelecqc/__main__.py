"""Local CLI. Import never mutates the supplied source files."""
import argparse
import json
import os
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
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8840)
    ingest = commands.add_parser("import")
    ingest.add_argument("files", nargs="+")
    ingest.add_argument("--modality", required=True, choices=["eeg", "seeg", "ecog"])
    ingest.add_argument("--label", help="Recording label; defaults to the first input filename")
    ingest.add_argument("--participant-id")
    ingest.add_argument("--format-hint", action="append", default=[], metavar="[RELATIVE_PATH=]FORMAT",
                        help="Explicit vendor format; repeat with relative paths for file bundles")
    commands.add_parser("list")
    args = parser.parse_args()
    if args.storage_root:
        os.environ["BRAINELECQC_STORAGE_ROOT"] = args.storage_root
    try:
        if args.command == "serve":
            from . import create_app
            create_app().run(host="127.0.0.1", port=args.port, debug=False)
        else:
            from .db import init_db
            from .config import settings
            from . import recordings
            if args.command == "import":
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
