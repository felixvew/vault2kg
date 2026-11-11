#!/usr/bin/env python3
"""apply_plan - rewrite the frontmatter of an Obsidian vault by a plan (vault-prep skill).

    python apply_plan.py ~/Documents/MyVault plan.json                     # dry run: what would change
    python apply_plan.py ~/Documents/MyVault plan.json --write --backup ~/MyVault-backup

Only frontmatter changes, and only the properties the plan names: note bodies are never
touched. Running it again changes nothing. --write needs a backup (--backup DIR, a new
folder) or a vault inside a git repository with nothing uncommitted. Needs PyYAML.

plan.json (every part optional):
  {"split":  {"systems": [" AND ", ","]},          cut text values into a list first
   "rename": {"team": {"Imgaging": "Imaging"}},    fix variants, before linking
   "link":   ["team", "risk", "systems"],          values become [[links]]
   "alias":  ["item_id"],                          the value is added to the note's aliases
   "hubs":   {"folder": "Hubs",                    create a note for a link target that has none,
              "notes": {"R3": "Risk level 3: ..."}}}   but only with a definition you give
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

KEY_LINE = re.compile(r"([^\s#\-][^:]*?)\s*:(?:\s|$)")
BAD_LINK = re.compile(r"[\[\]|#^\\]")          # a link target can't contain these
BAD_FILE = re.compile(r'[<>:"/\\|?*\[\]#^]')     # nor a file name these


def notes_of(vault: Path):
    for p in sorted(vault.rglob("*.md")):
        if not any(part.startswith(".") for part in p.relative_to(vault).parts):
            yield p


def values_of(v) -> list:
    if v is None or v == "":
        return []
    return [x for x in (v if isinstance(v, list) else [v]) if x is not None and x != ""]


def yaml_scalar(v) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)   # a JSON string is a valid double-quoted YAML string
    return yaml.safe_dump(v, default_flow_style=True, allow_unicode=True).split("\n")[0]


def render(key: str, value) -> list[str]:
    if isinstance(value, list):
        return [f"{key}:"] + [f"  - {yaml_scalar(v)}" for v in value]
    return [f"{key}: {yaml_scalar(value)}"]


def parse(text: str):
    """(newline, lines, end index of the frontmatter, meta) or None without frontmatter."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    if not lines or lines[0].lstrip("﻿").strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
    if end is None:
        return None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        return "invalid"
    return (nl, lines, end, meta) if isinstance(meta, dict) else "invalid"


def key_spans(lines: list[str], end: int) -> dict[str, tuple[int, int]]:
    """Top-level key -> (first line, line after its value) inside the frontmatter."""
    spans, cur = {}, None
    for i in range(1, end):
        line = lines[i]
        m = KEY_LINE.match(line)
        if m and not line[:1].isspace():
            if cur:
                spans[cur[0]] = (cur[1], i)
            cur = (m.group(1).strip(), i)
        elif line.startswith("#") and cur:      # a comment ends the value before it
            spans[cur[0]] = (cur[1], i)
            cur = None
    if cur:
        spans[cur[0]] = (cur[1], end)
    return spans


class Planner:
    def __init__(self, plan: dict):
        self.split = {k: re.compile("|".join(re.escape(s) for s in seps)) for k, seps in plan.get("split", {}).items()}
        self.rename = plan.get("rename", {})
        self.link = set(plan.get("link", []))
        self.alias = list(plan.get("alias", []))
        hubs = plan.get("hubs", {})
        self.hub_folder, self.hubs = hubs.get("folder", "Hubs"), hubs.get("notes", {})
        self.stats: dict[str, Counter] = defaultdict(Counter)
        self.warnings: list[str] = []

    def value(self, key: str, value, where: str):
        """The new value of one property (unchanged if nothing applies)."""
        vals, was_list = values_of(value), isinstance(value, list)
        if key in self.split:
            parts = []
            for v in vals:
                cut = [p.strip() for p in self.split[key].split(v)] if isinstance(v, str) else [v]
                cut = [p for p in cut if p != ""]
                if len(cut) > 1:
                    self.stats[key]["split into lists"] += 1
                parts += cut
            vals = parts
        if key in self.rename:
            vals = [self.rename[key].get(v, v) if isinstance(v, str) else v for v in vals]
        if key in self.link:
            linked = []
            for v in vals:
                if isinstance(v, str) and "[[" not in v and not BAD_LINK.search(v):
                    self.stats[key][v] += 1
                    linked.append(f"[[{v}]]")
                    continue
                if not isinstance(v, str):
                    self.warnings.append(f"{where}: {key}: {v!r} is not text, left as it is")
                elif "[[" not in v:
                    self.warnings.append(f"{where}: {key}: {v!r} can't be a link target, left as it is")
                linked.append(v)                  # already a link, or left as it is
            vals = linked
        return vals if (was_list or len(vals) > 1) else (vals[0] if vals else value)

    def note(self, text: str, where: str) -> str:
        parsed = parse(text)
        if parsed is None:
            return text
        if parsed == "invalid":
            self.warnings.append(f"{where}: frontmatter is not valid YAML, left as it is")
            return text
        nl, lines, end, meta = parsed
        changes: dict[str, object] = {}
        for key, value in meta.items():
            if key in self.split or key in self.rename or key in self.link:
                new = self.value(key, value, where)
                if new != value:
                    changes[key] = new
        ids = [str(x).strip() for k in self.alias for x in values_of(meta.get(k))]
        if ids:
            akey = "alias" if "alias" in meta and "aliases" not in meta else "aliases"
            current = [str(a) for a in values_of(meta.get(akey))]
            add = [i for i in ids if i not in current]
            if add:
                changes[akey] = current + add
                self.stats["aliases"]["notes"] += 1
        if not changes:
            return text
        spans = key_spans(lines, end)
        out = list(lines)
        for key in sorted((k for k in changes if k in spans), key=lambda k: -spans[k][0]):   # bottom up
            a, b = spans[key]
            out[a:b] = render(key, changes[key])
        new_keys = [k for k in changes if k not in spans]
        close = next(i for i in range(1, len(out)) if out[i].strip() in ("---", "..."))
        for key in reversed(new_keys):
            out[close:close] = render(key, changes[key])
        return nl.join(out)

    def hub_notes(self, vault: Path) -> list[tuple[Path, str]]:
        stems = {p.stem.casefold() for p in notes_of(vault)}
        todo = []
        for name, spec in self.hubs.items():
            if name.casefold() in stems:
                continue
            if BAD_FILE.search(name):
                self.warnings.append(f"hub {name!r}: not a valid file name, not created")
                continue
            spec = {"text": spec} if isinstance(spec, str) else dict(spec)
            fm = f"---\ntype: {yaml_scalar(spec['type'])}\n---\n" if spec.get("type") else ""
            todo.append((vault / self.hub_folder / f"{name}.md", f"{fm}# {name}\n\n{spec.get('text', '').strip()}\n"))
        return todo


def safe_to_write(vault: Path, backup: Path | None) -> str | None:
    """None if writing is allowed, else why not."""
    if backup:
        return f"backup folder already exists: {backup}" if backup.exists() else None
    try:
        inside = subprocess.run(["git", "-C", str(vault), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True).stdout.strip() == "true"
        dirty = subprocess.run(["git", "-C", str(vault), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        inside, dirty = False, ""
    if not inside:
        return "the vault is not in a git repository: pass --backup DIR to copy it first"
    if dirty:
        return "the vault has uncommitted changes: commit them first, or pass --backup DIR"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apply_plan", description=__doc__.split("\n\n")[0])
    ap.add_argument("vault", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("--write", action="store_true", help="apply the changes (default: dry run)")
    ap.add_argument("--backup", type=Path, help="copy the vault here before writing (a new folder)")
    a = ap.parse_args(argv)
    vault = a.vault.expanduser()
    if not vault.is_dir():
        sys.exit(f"Not a folder: {vault}")
    planner = Planner(json.loads(a.plan.read_text(encoding="utf-8")))
    changed: list[tuple[Path, str, str]] = []
    for p in notes_of(vault):
        old = p.read_bytes().decode("utf-8")
        new = planner.note(old, p.relative_to(vault).as_posix())
        if new != old:
            changed.append((p, old, new))
    hubs = planner.hub_notes(vault)

    print(f"{vault.name}: {len(changed)} notes to change, {len(hubs)} hub notes to create")
    for key, c in planner.stats.items():
        detail = ", ".join(f"{k} ({n})" for k, n in c.most_common(6))
        print(f"  {key}: {detail}")
    for path, _ in hubs:
        print(f"  new note: {path.relative_to(vault).as_posix()}")
    for w in planner.warnings[:20]:
        print(f"  ! {w}")
    if changed:
        p, old, new = changed[0]
        print(f"\nExample, {p.relative_to(vault).as_posix()}:")
        old_fm, new_fm = old.split("\n---", 1)[0].splitlines(), new.split("\n---", 1)[0].splitlines()
        print("\n".join([f"  - {l}" for l in old_fm if l not in new_fm] + [f"  + {l}" for l in new_fm if l not in old_fm]))
    if not a.write:
        print("\nDry run: nothing written. Add --write (with --backup DIR, or in a clean git repository).")
        return 0
    why = safe_to_write(vault, a.backup)
    if why:
        sys.exit(f"Not writing: {why}")
    if a.backup:
        shutil.copytree(vault, a.backup)
        print(f"\nBackup: {a.backup}")
    for p, _, new in changed:
        p.write_bytes(new.encode("utf-8"))
    for path, text in hubs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Written: {len(changed)} notes changed, {len(hubs)} hub notes created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
