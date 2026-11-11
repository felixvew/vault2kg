#!/usr/bin/env python3
"""kg_to_graphml - export graph.json as GraphML (Gephi, yEd, Cytoscape, networkx).

    python kg_to_graphml.py ~/Documents/MyVault-kg     # writes MyVault-kg/graph.graphml

No dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


def load_graph(path: str) -> tuple[dict, Path]:
    """Accept the -kg folder or graph.json itself."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "graph.json"
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun vault2kg.py on your vault first.")
    graph = json.loads(p.read_text(encoding="utf-8"))
    if graph.get("format") != "vault2kg-graph":
        sys.exit(f"{p} is not a vault2kg graph.json")
    return graph, p.parent


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_XML_BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def export_graphml(graph: dict, path: Path) -> None:
    x = lambda s: html.escape(_XML_BAD.sub("", str(s)), quote=True)  # noqa: E731
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">']
    node_keys = ["name", "kind", "type", "description", "notes"]
    edge_keys = ["type", "origins", "confidence", "description"]
    for k in node_keys:
        out.append(f'  <key id="n_{k}" for="node" attr.name="{k}" attr.type="string"/>')
    out.append('  <key id="n_degree" for="node" attr.name="degree" attr.type="int"/>')
    for k in edge_keys:
        out.append(f'  <key id="e_{k}" for="edge" attr.name="{k}" attr.type="{"double" if k == "confidence" else "string"}"/>')
    out.append('  <graph id="vault" edgedefault="directed">')
    for n in graph["nodes"]:
        out.append(f'    <node id="{x(n["id"])}">')
        for k in node_keys:
            v = "; ".join(n[k]) if isinstance(n[k], list) else n[k]
            if v not in ("", None):
                out.append(f'      <data key="n_{k}">{x(v)}</data>')
        out.append(f'      <data key="n_degree">{n["degree"]}</data>')
        out.append("    </node>")
    for e in graph["edges"]:
        out.append(f'    <edge id="{e["id"]}" source="{x(e["source"])}" target="{x(e["target"])}">')
        for k in edge_keys:
            v = ", ".join(e[k]) if isinstance(e[k], list) else e[k]
            if v not in ("", None):
                out.append(f'      <data key="e_{k}">{x(v)}</data>')
        out.append("    </edge>")
    out += ["  </graph>", "</graphml>"]
    write_text(path, "\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kg_to_graphml", description="graph.json -> graph.graphml")
    p.add_argument("graph", help="the -kg folder written by vault2kg.py (or its graph.json)")
    p.add_argument("-o", "--out", type=Path, help="output file (default: graph.graphml next to graph.json)")
    a = p.parse_args(argv)
    graph, folder = load_graph(a.graph)
    out = a.out or folder / "graph.graphml"
    export_graphml(graph, out)
    print(f"Wrote {out} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
