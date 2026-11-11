#!/usr/bin/env python3
"""profile_vault - what the frontmatter of an Obsidian vault looks like (vault-prep skill).

    python profile_vault.py ~/Documents/MyVault          # a table to read
    python profile_vault.py ~/Documents/MyVault --json   # the same, for an agent

For every property: in how many notes it appears, how many distinct values it has,
how many of those several notes share, a suggestion, and examples. The suggestion
is only a starting point for the plan; the user decides. Reads only. Needs PyYAML.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SEPARATORS = re.compile(r"\s+(?:AND|OR|and|or|&|\+)\s+|\s*;\s*|\s+/\s+")   # "A AND B", "A; B", "A / B"
NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?\s*%?")
ID_LIKE = re.compile(r"[A-Za-z]{0,4}[-_ ]?\d+(?:[.\-_]\d+)*")


def notes_of(vault: Path):
    for p in sorted(vault.rglob("*.md")):
        if any(part.startswith(".") for part in p.relative_to(vault).parts):
            continue
        yield p


def frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿").replace("\r\n", "\n")
    m = re.match(r"---\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|$)", text, re.S)
    if not m:
        return {}
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return meta if isinstance(meta, dict) else {}


def values_of(v) -> list:
    if v is None or v == "":
        return []
    return [x for x in (v if isinstance(v, list) else [v]) if x is not None and x != ""]


def suggest(key: str, values: list, n_notes: int, n_total: int) -> str:
    """A guess at what the plan should do with this property."""
    texts = [str(x) for x in values]
    counts = Counter(texts)
    if key in ("aliases", "alias", "tags", "tag", "cssclasses", "cssclass"):
        return "keep (Obsidian field)"
    if any("[[" in t for t in texts):
        return "keep (already links)"
    top = counts.most_common(1)[0][1] if counts else 0
    if n_total >= 5 and top >= 0.9 * n_total:
        return "keep (same in nearly every note)"
    if all(isinstance(x, (int, float, bool)) or NUMBER.fullmatch(str(x).strip()) for x in values):
        return "keep (numbers)"
    if all(isinstance(x, (dt.date, dt.datetime)) or re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", str(x)) for x in values):
        return "keep (dates)"
    if len(counts) == len(texts) and n_notes >= 3:
        if sum(1 for t in texts if ID_LIKE.fullmatch(t.strip())) >= 0.8 * len(texts):
            return "alias (an ID per note)"
        return "keep (different in every note)"
    if sum(1 for t in texts if SEPARATORS.search(t)) >= 0.2 * len(texts):
        return "split, then link?"
    if any(c >= 2 for c in counts.values()) and len(counts) <= max(3, 0.6 * len(texts)) \
            and max(len(t) for t in texts) <= 60:
        return "link?"
    return "keep (free text)"


def profile(vault: Path) -> dict:
    per_key: dict[str, list] = defaultdict(list)
    notes_with: Counter = Counter()
    total = 0
    for p in notes_of(vault):
        total += 1
        for k, v in frontmatter(p).items():
            if isinstance(v, dict):
                continue
            notes_with[k] += 1
            per_key[k].extend(values_of(v))
    rows = []
    for k in sorted(per_key, key=lambda k: -notes_with[k]):
        vals = per_key[k]
        counts = Counter(str(x) for x in vals)
        rows.append({
            "key": k, "notes": notes_with[k], "distinct": len(counts),
            "shared": sum(1 for c in counts.values() if c >= 2),
            "suggestion": suggest(k, vals, notes_with[k], total),
            "examples": [f"{v} ({c})" for v, c in counts.most_common(4)],
        })
    return {"vault": vault.name, "notes": total, "properties": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="profile_vault", description=__doc__.split("\n\n")[0])
    ap.add_argument("vault", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    if not a.vault.is_dir():
        sys.exit(f"Not a folder: {a.vault}")
    prof = profile(a.vault.expanduser())
    if a.json:
        print(json.dumps(prof, ensure_ascii=False, indent=1))
        return 0
    print(f"{prof['vault']}: {prof['notes']} notes\n")
    print("| property | notes | distinct | shared | suggestion | examples (notes) |")
    print("|---|---|---|---|---|---|")
    for r in prof["properties"]:
        print(f"| {r['key']} | {r['notes']} | {r['distinct']} | {r['shared']} | {r['suggestion']} | "
              f"{'; '.join(r['examples'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
