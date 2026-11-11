"""The vault-prep skill's scripts: profile a vault, rewrite its frontmatter by a plan."""
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

import vault2kg as v

HERE = Path(__file__).parent
SKILL = HERE / "skills" / "vault-prep"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SKILL / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


profile_vault, apply_plan = _load("profile_vault"), _load("apply_plan")


@pytest.fixture()
def demo(tmp_path):
    dst = tmp_path / "demo"
    shutil.copytree(HERE / "demo_vault", dst)
    return dst


def body(path: Path) -> str:
    return path.read_text(encoding="utf-8").split("\n---\n", 1)[1]


def test_profile_suggests_the_plan(demo):
    rows = {r["key"]: r for r in profile_vault.profile(demo)["properties"]}
    assert rows["item_id"]["suggestion"].startswith("alias")
    assert rows["team"]["suggestion"] == rows["risk"]["suggestion"] == "link?"
    assert rows["systems"]["suggestion"].startswith("split")
    assert rows["source_document"]["suggestion"].startswith("keep (same")


def test_dry_run_writes_nothing_and_write_needs_a_backup(demo, tmp_path, capsys):
    before = {p: p.read_bytes() for p in demo.rglob("*.md")}
    plan = SKILL / "example-plan.json"
    assert apply_plan.main([str(demo), str(plan)]) == 0
    assert "8 notes to change, 3 hub notes to create" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not in a git repository"):
        apply_plan.main([str(demo), str(plan), "--write"])
    assert {p: p.read_bytes() for p in demo.rglob("*.md")} == before


def test_write_links_splits_aliases_and_keeps_bodies(demo, tmp_path, capsys):
    plan, backup = SKILL / "example-plan.json", tmp_path / "backup"
    assert apply_plan.main([str(demo), str(plan), "--write", "--backup", str(backup)]) == 0
    note = demo / "Methods" / "1.1.2 - Keypoint Extraction.md"
    meta = v.split_frontmatter(note.read_text(encoding="utf-8"))[0]
    assert meta["team"] == "[[Imaging]]" and meta["risk"] == "[[R2]]"
    assert meta["systems"] == ["[[Lab server]]", "[[GPU node]]"] and meta["aliases"] == ["1.1.2"]
    assert meta["source_document"] == "Lab Methods Handbook.pdf"              # not in the plan: untouched
    for p in (demo / "Methods").glob("*.md"):                                  # note text never changes
        assert body(p) == body(backup / "Methods" / p.name)
    assert "Risk level 3" in (demo / "Hubs" / "R3.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert apply_plan.main([str(demo), str(plan)]) == 0                       # running it again: nothing to do
    assert "0 notes to change, 0 hub notes to create" in capsys.readouterr().out


def test_prepared_vault_gets_every_edge_without_an_llm(demo, tmp_path):
    apply_plan.main([str(demo), str(SKILL / "example-plan.json"), "--write", "--backup", str(tmp_path / "b")])
    out = tmp_path / "kg"
    v.VaultExtractor(v.Config(vault=demo, out=out), None, progress=lambda m: None).run()
    g = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    edges = [(e["source"], e["type"], e["target"]) for e in g["edges"]]
    methods = [n for n in g["nodes"] if n["kind"] == "note" and n["id"].startswith("note:Methods/")]
    for kind in ("TEAM", "RISK"):
        assert len({s for s, t, _ in edges if t == kind}) == len(methods) == 8
    r3 = next(n for n in g["nodes"] if n["name"] == "R3")
    assert r3["resolved"] is True                                              # the hub note exists


def test_edge_cases(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "A.md").write_bytes(
        b'---\r\naliases: [Alpha]\r\nteam: "Ops | Night"\r\nscore: 3\r\nowner: "[[Tom]]"\r\nid: "7"\r\n---\r\nText.\r\n')
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"link": ["team", "score", "owner"], "alias": ["id"]}), encoding="utf-8")
    apply_plan.main([str(vault), str(plan), "--write", "--backup", str(tmp_path / "b")])
    raw = (vault / "A.md").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")         # Windows line endings kept
    meta = v.split_frontmatter(raw.decode().replace("\r\n", "\n"))[0]
    assert meta["aliases"] == ["Alpha", "7"]                                   # existing aliases kept
    assert meta["team"] == "Ops | Night" and meta["score"] == 3                # can't be a link / not text
    assert meta["owner"] == "[[Tom]]"                                          # already a link
    assert raw.decode().endswith("---\r\nText.\r\n")
