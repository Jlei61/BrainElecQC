"""Reject data, private paths and automated attribution before publication."""
import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


def git(*args):
    return subprocess.check_output(["git", *args])


def inspect_file(name, content):
    path = PurePosixPath(name)
    permitted = (path.suffix in {".py", ".js", ".mjs", ".md", ".toml", ".sql"}
                 or name in {".gitignore", "package.json", ".githooks/pre-commit", ".githooks/pre-push"})
    problems = []
    if not permitted:
        problems.append("file type not in publication allowlist")
    if any(part.lower() in {"data", "dataset", "storage", "reports", "artifacts", "local", "node_modules"} for part in path.parts):
        problems.append("data/output directory")
    if len(content) > 500_000 or b"\x00" in content:
        problems.append("binary or unexpectedly large file")
    text = content.decode("utf-8", errors="replace")
    private_root = r"/(?:" + r"Users|home" + r")/[A-Za-z0-9_.-]+/"
    if re.search(private_root, text):
        problems.append("private absolute home path")
    if re.search(r"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}|^-----BEGIN [A-Z ]*PRIVATE KEY-----", text, re.M):
        problems.append("credential-like content")
    if re.search(r"^\s*(?:#|//|\*)?\s*(?:co-authored-by:|generated (?:by|with)\b)", text, re.I | re.M):
        problems.append("automated attribution in published content")
    if path.name.lower() in {"agents.md", "claude.md"}:
        problems.append("tool-specific session/rules file")
    return [f"{name}: {problem}" for problem in problems]


def inspect_commit(message):
    # Names here are policy matchers, never attribution for repository content.
    if re.search(r"co-authored-by:|generated (?:by|with)|\b(?:codex|claude|chatgpt|openai|agent)\b", message, re.I):
        return ["commit identity/message contains disallowed automated attribution"]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    errors, count, seen = [], 0, set()
    if args.history:
        commits = git("rev-list", "--all").decode().splitlines()
        for commit in commits:
            errors.extend(inspect_commit(git("show", "-s", "--format=%an <%ae>%n%cn <%ce>%n%B", commit).decode()))
            entries = git("ls-tree", "-rz", "--full-tree", commit).split(b"\0")
            for entry in filter(None, entries):
                meta, name = entry.split(b"\t", 1)
                mode, kind, oid = meta.split()
                key = (name, oid)
                if key in seen:
                    continue
                seen.add(key)
                if kind != b"blob" or mode == b"120000":
                    errors.append(f"{name.decode()}: symlink or external Git object")
                else:
                    errors.extend(inspect_file(name.decode(), git("cat-file", "blob", oid.decode())))
                count += 1
    else:
        # Audit the complete index, including previously staged paths.
        for entry in filter(None, git("ls-files", "--stage", "-z").split(b"\0")):
            meta, name = entry.split(b"\t", 1)
            mode, oid, stage = meta.split()
            if mode == b"120000" or stage != b"0":
                errors.append(f"{name.decode()}: symlink or unresolved index")
            else:
                errors.extend(inspect_file(name.decode(), git("cat-file", "blob", oid.decode())))
            count += 1
    if errors:
        print("Publication audit failed:\n" + "\n".join(sorted(set(errors))))
        return 1
    print(f"Publication audit passed: {count} file objects; no disallowed data/path/attribution detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
