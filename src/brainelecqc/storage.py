"""Streaming snapshots for QC. No hard links, shared staging names or replacement.

Immutability is enforced by the import API, not a claim of OS-level WORM storage.
Explicit verification detects later external changes, including ZIP member changes.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import unicodedata
import zipfile

CHUNK_SIZE = 1024 * 1024
MAX_FILES = 100_000
MAX_BYTES = 4 * 1024 ** 4  # expanded input; bounded by local storage policy


class ImportConflict(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def safe_relative(name):
    if (not isinstance(name, str) or not name or name.startswith("/") or
            any(c in name for c in "\\\0:") or
            any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe input path: {name!r}")
    return Path(*name.split("/"))


def confined(path, root):
    """Reject links and escaping paths, including symlinked ancestor directories."""
    configured_root = Path(os.path.abspath(root))
    root = configured_root.resolve()
    path = Path(os.path.abspath(path))
    try:
        parts = path.relative_to(configured_root).parts
    except ValueError:
        try:
            parts = path.relative_to(root).parts
        except ValueError as exc:
            raise ValueError("Input must belong to the selected storage root") from exc
    cursor = root
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Symlink inputs/storage are not supported: {cursor.name}")
    return root.joinpath(*parts)


def hash_file(path):
    hasher = hashlib.sha256()
    size = 0
    # O_NOFOLLOW also protects against a leaf replaced between inventory and open.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("Only regular files are accepted")
        for block in iter(lambda: source.read(CHUNK_SIZE), b""):
            size += len(block)
            hasher.update(block)
    return {"size_bytes": size, "sha256": hasher.hexdigest()}


def inventory(root):
    """Inventory regular files, rejecting links/special files without following them."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Symlink input is not supported")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ValueError(f"Input does not exist: {root.name}")
    files = []
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(folder) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"Non-regular input: {path.relative_to(root)}")
        files.extend(Path(folder) / name for name in names)
        if len(files) > MAX_FILES:
            raise ValueError("Input exceeds the file count limit")
    return sorted(files)


def stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class SnapshotWriter:
    def __init__(self, stage):
        self.stage = Path(stage)
        self.root = self.stage / "files"
        self.root.mkdir()
        self.entries = []
        self.names = {}
        self.size = 0
        self.source_checks = []

    def _reserve(self, name):
        relative = safe_relative(name)
        # Reject ambiguous case/Unicode names even on case-sensitive hosts.
        for i in range(1, len(relative.parts) + 1):
            prefix = "/".join(relative.parts[:i])
            key = unicodedata.normalize("NFC", prefix).casefold()
            if key in self.names and (self.names[key] != prefix or i == len(relative.parts)):
                raise ValueError(f"Duplicate or ambiguous input name: {name}")
            self.names[key] = prefix
        if len(self.entries) >= MAX_FILES:
            raise ValueError("Input exceeds the file count limit")
        return relative

    def add_stream(self, name, source):
        relative = self._reserve(name)
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        size = 0
        with target.open("xb") as output:
            for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                size += len(block)
                self.size += len(block)
                if self.size > MAX_BYTES:
                    raise ValueError("Expanded input exceeds the storage limit")
                output.write(block)
                hasher.update(block)
        target.chmod(0o444)
        entry = {"relative_path": relative.as_posix(), "size_bytes": size,
                 "sha256": hasher.hexdigest()}
        self.entries.append(entry)
        return entry

    def add_zip(self, source):
        try:
            with zipfile.ZipFile(source) as archive:
                if len(archive.infolist()) > MAX_FILES:
                    raise ValueError("Archive exceeds the entry count limit")
                for member in archive.infolist():
                    safe_relative(member.filename.rstrip("/"))
                    kind = stat.S_IFMT(member.external_attr >> 16)
                    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError("ZIP links and special files are not supported")
                    if member.flag_bits & 1:
                        raise ValueError("Encrypted ZIP inputs are not supported")
                    if member.is_dir():
                        continue
                    if self.size + member.file_size > MAX_BYTES:
                        raise ValueError("Expanded archive exceeds the storage limit")
                    with archive.open(member) as stream:
                        self.add_stream(member.filename, stream)
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
            raise ValueError(f"Cannot read ZIP input: {exc}") from exc

    def add_upload(self, upload):
        name = upload.filename
        safe_relative(name)
        if name.lower().endswith(".zip"):
            # A seekable named stream lets ZipFile decompress one member at a time.
            self.add_zip(upload.stream)
        else:
            self.add_stream(name, upload.stream)

    def add_path(self, source, source_root):
        source = confined(source, source_root)
        paths = inventory(source)
        before = {path: stamp(path) for path in paths}
        copied = {}
        for path in paths:
            confined(path, source_root)
            if source.is_file() and path.suffix.lower() == ".zip":
                copied[path] = hash_file(path)
                self.add_zip(path)
            else:
                name = path.relative_to(source).as_posix() if source.is_dir() else path.name
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as stream:
                    entry = self.add_stream(name, stream)
                copied[path] = {k: entry[k] for k in ("size_bytes", "sha256")}
        self.source_checks.append((source, source_root, paths, before, copied))

    def verify_sources(self):
        # Verify the complete source set after ALL copies, not just each file locally.
        for source, source_root, paths, before, copied in self.source_checks:
            confined(source, source_root)
            if inventory(source) != paths:
                raise ImportConflict("Source file set changed during import; retry after upload completes")
            for path in paths:
                confined(path, source_root)
                if stamp(path) != before[path] or hash_file(path) != copied[path]:
                    raise ImportConflict("Source bytes changed during import; retry after upload completes")

    def finish(self):
        if not self.entries:
            raise ValueError("Input contains no regular files")
        return sorted(self.entries, key=lambda entry: entry["relative_path"])


def inspect_snapshot(root, expected_manifest_sha256, expected_content_sha256):
    """Return differences; never silently accept a present but stale artifact."""
    issues = []
    try:
        root = Path(root)
        if root.is_symlink() or (root / "manifest.json").is_symlink():
            raise ValueError("Snapshot or manifest is a symlink")
        raw = (root / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            issues.append({"code": "manifest_mismatch"})
        manifest = json.loads(raw)
        actual = [{"relative_path": p.relative_to(root / "files").as_posix(), **hash_file(p)}
                  for p in inventory(root / "files")]
        actual.sort(key=lambda entry: entry["relative_path"])
        if digest(actual) != expected_content_sha256:
            issues.append({"code": "content_mismatch"})
        expected = {entry["relative_path"]: entry for entry in manifest["files"]}
        observed = {entry["relative_path"]: entry for entry in actual}
        for name in sorted(expected.keys() | observed.keys()):
            if expected.get(name) != observed.get(name):
                kind = "added" if name not in expected else "missing" if name not in observed else "changed"
                issues.append({"code": kind, "relative_path": name})
        extras = sorted(p.name for p in root.iterdir() if p.name not in ("files", "manifest.json"))
        if extras:
            issues.append({"code": "unexpected_root_entries", "paths": extras})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({"code": "unreadable_snapshot", "reason": str(exc)})
    return issues
