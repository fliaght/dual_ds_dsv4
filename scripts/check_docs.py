#!/usr/bin/env python3
"""Check managed documentation and the pinned protocol without network access."""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_MARKER = b"<!-- BEGIN VERBATIM DEV-PROTOCOL -->\n"
PROTOCOL_SHA256 = "db7023cf370552b775cf6b279532310ebfd2a770b1784fce34475e7166ae3341"
SEMVER = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?"


def check_front_matter(path, errors):
    text = path.read_text()
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not match:
        errors.append(f"{path.relative_to(ROOT)}: missing YAML front matter")
        return
    # This repository uses a deliberately small YAML subset: scalars and inline tags.
    fields = {}
    for line in match[1].splitlines():
        key, separator, value = line.partition(":")
        if not separator or key in fields:
            errors.append(f"{path.relative_to(ROOT)}: malformed/duplicate front matter field")
        fields[key] = value.strip().strip('\"\'')
    if not fields.get("summary"):
        errors.append(f"{path.relative_to(ROOT)}: summary is required")
    if not re.fullmatch(r"\[\s*[A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)*\s*\]", fields.get("tags", "")):
        errors.append(f"{path.relative_to(ROOT)}: tags must be a nonempty inline list")
    try:
        date.fromisoformat(fields.get("updated", ""))
    except ValueError:
        errors.append(f"{path.relative_to(ROOT)}: updated must be YYYY-MM-DD")
    if not re.fullmatch(SEMVER, fields.get("version", "")):
        errors.append(f"{path.relative_to(ROOT)}: version must identify a project baseline")


def check_links(path, errors):
    # Exclude code examples; do not resolve HTTP or historical absolute host paths.
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    for target in re.findall(r"\[[^\]\n]*\]\(([^)\s]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or not parsed.path or parsed.path.startswith("/"):
            continue
        destination = (path.parent / unquote(parsed.path)).resolve()
        if not destination.is_relative_to(ROOT):
            # Archived guides refer to sibling projects on the historical host.
            if "ArchiveDocs" not in path.parts:
                errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
            continue
        if not destination.exists():
            errors.append(f"{path.relative_to(ROOT)}: broken relative link: {target}")


def main():
    if not (ROOT / "AGENTS.md").exists() and not (ROOT / "Docs").exists():
        # The published branch carries code only; the documentation is kept in the maintainer's checkout.
        print("Documentation is not in this checkout: documentation check skipped.")
        return
    errors = []
    version = (ROOT / "VERSION").read_text().strip()
    if not re.fullmatch(SEMVER, version):
        errors.append("VERSION must be a semantic project version")
    docs = ROOT / "Docs"
    expected = {"kedb", "runbooks", "playbooks", "postmortems", "current.md"}
    if {path.name for path in docs.iterdir()} != expected:
        errors.append("Docs/ must contain only current.md and the four protocol directories")
    for name in expected - {"current.md"}:
        if not (docs / name).is_dir():
            errors.append(f"Docs/{name} must be a directory")
    if (ROOT / "docs").exists():
        errors.append("obsolete docs/ directory must not shadow Docs/")

    instructions = (ROOT / "AGENTS.md").read_bytes()
    if instructions.count(PROTOCOL_MARKER) != 1:
        errors.append("AGENTS.md must contain exactly one imported-protocol marker")
    elif hashlib.sha256(instructions.split(PROTOCOL_MARKER, 1)[1]).hexdigest() != PROTOCOL_SHA256:
        errors.append("the imported dev-protocol text has changed")
    claude = ROOT / "CLAUDE.md"
    if not claude.is_symlink() or claude.readlink() != Path("AGENTS.md"):
        errors.append("CLAUDE.md must be a relative symlink to AGENTS.md")

    managed = sorted(p for directory in ("Docs", "Knowledge", "ArchiveDocs")
                     for p in (ROOT / directory).rglob("*.md"))
    for path in managed:
        check_front_matter(path, errors)
    for path in [*managed, ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "chat/README.md", ROOT / "AGENTS.md"]:
        check_links(path, errors)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Documentation OK: {len(managed)} managed documents; protocol bytes and symlink verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
