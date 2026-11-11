#!/usr/bin/env python3
"""kg_to_neo4j - load graph.json into Neo4j.

    pip install neo4j
    python src/kg_to_neo4j.py ~/Documents/MyVault-kg --password secret

    # also embed the passages and create a vector index (ready for vector search):
    python src/kg_to_neo4j.py ~/Documents/MyVault-kg --password secret --embed

    # or write a Cypher file instead (no driver needed), run it with cypher-shell:
    python src/kg_to_neo4j.py ~/Documents/MyVault-kg --cypher neo4j.cypher

Connection: --uri / --user / --password / --database, or the environment
variables NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE.

What you get:
  (:VaultNode:Note), (:VaultNode:Entity:Person), (:VaultNode:Tag), (:VaultNode:File)
      every node: id, name, kind, type, description, aliases, notes; notes also
      path and their frontmatter as fm_* properties
  (:Chunk {id, text, ...})        a note section (the unit the LLM read)
  (:Passage {id, text, embed_text, tokens, ...})   a small retrieval passage
  relationships as extracted (LINKS_TO, MENTIONS, HAS_TAG, WORKS_AT, ...), plus
  (:Note)-[:HAS_CHUNK]->(:Chunk)<-[:PART_OF]-(:Passage),
  (entity)-[:MENTIONED_IN]->(:Chunk) and (entity)-[:MENTIONED_IN]->(:Passage)

--embed: every Passage gets `embedding` (its embed_text, embedded) and
`embedding_model`, plus a cosine vector index `passage_embedding`. The embedding
model is found like vault2kg.py finds it: --embed-base-url / --embed-model,
else a vLLM container on localhost:8001, else Ollama's embedding model, else
OpenAI (OPENAI_API_KEY). Vectors are cached in <kg>/.cache, so re-runs only
embed changed passages. Embed your queries with the SAME model, then:

  CALL db.index.vector.queryNodes('passage_embedding', 5, $query_vector)
  YIELD node AS p, score
  MATCH (p)-[:PART_OF]->(c:Chunk)<-[:HAS_CHUNK]-(n:Note)
  OPTIONAL MATCH (e)-[:MENTIONED_IN]->(p)
  RETURN n.name AS note, p.text AS passage, c.text AS section, collect(e.name) AS entities, score

Re-running syncs: everything is MERGEd, and nodes/relationships of this vault
that are no longer in graph.json (deleted notes, removed links) are removed.
Nodes of other vaults in the same database are left alone.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

BATCH = 500


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


# --------------------------------------------------------------------------- #
# Graph -> parameterised Cypher statements
# --------------------------------------------------------------------------- #
def label(text: str) -> str:
    """'Person' -> 'Person', 'Machine learning!' -> 'Machinelearning', '42' -> ''."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    out = re.sub(r"[^0-9A-Za-z]", "", ascii_)
    return out if out[:1].isalpha() else ""


def node_props(n: dict, vault: str, run: str) -> dict:
    props = {k: n[k] for k in ("name", "kind", "type", "description", "aliases", "notes", "path", "resolved")
             if n.get(k) not in (None, "", [])}
    for k, v in (n.get("properties") or {}).items():  # frontmatter -> queryable fm_* properties
        if isinstance(v, (str, int, float, bool)):
            props[f"fm_{k}"] = v
        elif isinstance(v, list) and v and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
            props[f"fm_{k}"] = [str(x) for x in v]
    props.update(vault=vault, run=run)
    return props


def statements(graph: dict) -> list[tuple[str, list[dict] | None, dict]]:
    """(query, rows or None, extra params). Rows are sent in batches as $rows."""
    vault, run = graph.get("vault", "vault"), graph.get("generated_at", "")
    out: list[tuple[str, list[dict] | None, dict]] = [
        ("CREATE CONSTRAINT vault_node_id IF NOT EXISTS FOR (n:VaultNode) REQUIRE n.id IS UNIQUE", None, {}),
        ("CREATE CONSTRAINT vault_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE", None, {}),
        ("CREATE CONSTRAINT vault_passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE", None, {}),
    ]
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for n in graph["nodes"]:
        labels = [n["kind"].title()]
        t = label(n.get("type") or "")
        if n["kind"] in ("note", "entity") and t and t not in labels and n["type"] not in ("Note", "Other"):
            labels.append(t)
        groups[tuple(labels)].append({"id": n["id"], "props": node_props(n, vault, run)})
    for labels, rows in sorted(groups.items()):
        out.append(("UNWIND $rows AS row MERGE (n:VaultNode {id: row.id}) "
                    f"SET n += row.props, n:{':'.join(labels)}", rows, {}))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in graph["edges"]:
        props = {k: e[k] for k in ("description", "confidence", "origins", "notes") if e.get(k) not in (None, "", [])}
        props["run"] = run
        by_type[e["type"]].append({"s": e["source"], "t": e["target"], "props": props})
    for rtype, rows in sorted(by_type.items()):
        safe = rtype.replace("`", "")
        out.append(("UNWIND $rows AS row MATCH (a:VaultNode {id: row.s}) MATCH (b:VaultNode {id: row.t}) "
                    f"MERGE (a)-[r:`{safe}`]->(b) SET r += row.props", rows, {}))

    chunks = [{"id": c["id"], "note": c["note"], "props": {
        "text": c["text"], "path": c["path"], "heading": " > ".join(c["heading_path"]),
        "start_line": c["start_line"], "end_line": c["end_line"], "vault": vault, "run": run}}
        for c in graph["chunks"]]
    out.append(("UNWIND $rows AS row MERGE (c:Chunk {id: row.id}) SET c += row.props "
                "WITH c, row MATCH (n:VaultNode {id: row.note}) MERGE (n)-[h:HAS_CHUNK]->(c) SET h.run = $run",
                chunks, {"run": run}))
    mentions = [{"e": e, "c": c["id"]} for c in graph["chunks"] for e in c.get("entities", [])]
    out.append(("UNWIND $rows AS row MATCH (e:VaultNode {id: row.e}) MATCH (c:Chunk {id: row.c}) "
                "MERGE (e)-[m:MENTIONED_IN]->(c) SET m.run = $run", mentions, {"run": run}))
    passages = [{"id": p["id"], "chunk": p["chunk"], "props": {
        "text": p["text"], "embed_text": p["embed_text"], "tokens": p["tokens"], "path": p["path"],
        "heading": " > ".join(p["heading_path"]), "start_line": p["start_line"], "end_line": p["end_line"],
        "vault": vault, "run": run}} for p in graph.get("passages", [])]
    out.append(("UNWIND $rows AS row MERGE (p:Passage {id: row.id}) SET p += row.props "
                "WITH p, row MATCH (c:Chunk {id: row.chunk}) MERGE (p)-[r:PART_OF]->(c) SET r.run = $run",
                passages, {"run": run}))
    pmentions = [{"e": e, "p": p["id"]} for p in graph.get("passages", []) for e in p.get("entities", [])]
    out.append(("UNWIND $rows AS row MATCH (e:VaultNode {id: row.e}) MATCH (p:Passage {id: row.p}) "
                "MERGE (e)-[m:MENTIONED_IN]->(p) SET m.run = $run", pmentions, {"run": run}))
    return out


def cleanup_statements(graph: dict) -> list[tuple[str, dict]]:
    """Remove what this vault had before but graph.json no longer contains."""
    p = {"vault": graph.get("vault", "vault"), "run": graph.get("generated_at", "")}
    return [
        ("MATCH (a {vault: $vault})-[r]->() WHERE (a:VaultNode OR a:Chunk OR a:Passage) AND r.run <> $run "
         "WITH r LIMIT 10000 DELETE r RETURN count(*) AS n", p),
        ("MATCH (n:VaultNode {vault: $vault}) WHERE n.run <> $run "
         "WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS n", p),
        ("MATCH (c:Chunk {vault: $vault}) WHERE c.run <> $run "
         "WITH c LIMIT 10000 DETACH DELETE c RETURN count(*) AS n", p),
        ("MATCH (x:Passage {vault: $vault}) WHERE x.run <> $run "
         "WITH x LIMIT 10000 DETACH DELETE x RETURN count(*) AS n", p),
    ]


# --------------------------------------------------------------------------- #
# Passage embeddings (--embed)
# --------------------------------------------------------------------------- #
VECTOR_INDEX = "passage_embedding"


def _http_json(url: str, body: dict | None = None, api_key: str | None = None, timeout: float = 120) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        return json.load(resp)


def _models(base_url: str, api_key: str | None) -> list[str]:
    try:
        return [m["id"] for m in _http_json(f"{base_url.rstrip('/')}/models", api_key=api_key, timeout=3)
                .get("data", []) if m.get("id")]
    except (OSError, ValueError, AttributeError):
        return []


def find_embedder(base_url: str | None, model: str | None, api_key: str | None) -> tuple[str, str, str | None]:
    """(base_url, model, api_key) of an OpenAI-compatible /embeddings endpoint."""
    base_url = base_url or os.environ.get("VAULT2KG_EMBED_BASE_URL")
    model = model or os.environ.get("VAULT2KG_EMBED_MODEL")
    api_key = api_key or os.environ.get("VAULT2KG_API_KEY")
    if base_url:
        model = model or next(iter(_models(base_url, api_key)), None)
        if not model:
            sys.exit(f"No embedding model found at {base_url}. Pass --embed-model.")
        return base_url, model, api_key
    vllm = os.environ.get("VAULT2KG_VLLM_EMBED_URL", "http://localhost:8001/v1")
    served = _models(vllm, api_key)
    if served:
        return vllm, model or served[0], api_key
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else "http://" + host
    host = (host if re.search(r":\d+$", host) else host + ":11434").replace("0.0.0.0", "localhost")
    try:
        names = [m.get("name", "") for m in _http_json(f"{host}/api/tags", timeout=3).get("models", [])]
    except (OSError, ValueError):
        names = []
    emb = [n for n in names if "embed" in n.lower() or n.lower().startswith(("bge", "nomic", "mxbai", "all-minilm"))]
    if model or emb:
        if names:
            return f"{host}/v1", model or emb[0], None
    if os.environ.get("OPENAI_API_KEY"):
        return "https://api.openai.com/v1", model or "text-embedding-3-small", os.environ["OPENAI_API_KEY"]
    sys.exit("--embed needs an embedding model: start the vLLM embedding container (docker-compose.vllm.yml), "
             "`ollama pull nomic-embed-text`, set OPENAI_API_KEY, or pass --embed-base-url / --embed-model.")


def embed_passages(passages: list[dict], base_url: str, model: str, api_key: str | None,
                   cache_file: Path, batch: int = 64, log=print) -> dict[str, list[float]]:
    """{passage id: vector}. Cached by (model, text): only new or changed passages are sent."""
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    key = lambda text: hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:32]  # noqa: E731
    todo = sorted({p["embed_text"] for p in passages if key(p["embed_text"]) not in cache})
    if todo:
        log(f"Embedding {len(todo)} passages with {model} ({len(passages) - len(todo)} cached)")
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        try:
            rows = _http_json(f"{base_url.rstrip('/')}/embeddings", {"model": model, "input": part}, api_key)["data"]
        except urllib.error.HTTPError as exc:
            sys.exit(f"Embedding failed at {base_url}: HTTP {exc.code} {exc.read()[:200].decode('utf-8', 'replace')}")
        except OSError as exc:
            sys.exit(f"Embedding server not reachable at {base_url}: {exc}")
        for text, row in zip(part, sorted(rows, key=lambda r: r.get("index", 0))):
            cache[key(text)] = row["embedding"]
    if todo:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        live = {key(p["embed_text"]) for p in passages}
        cache_file.write_text(json.dumps({k: v for k, v in cache.items() if k in live}), encoding="utf-8")
    return {p["id"]: cache[key(p["embed_text"])] for p in passages}


def vector_statements(vectors: dict[str, list[float]], model: str) -> list[tuple[str, list[dict] | None, dict]]:
    if not vectors:
        return []
    dim = len(next(iter(vectors.values())))
    rows = [{"id": pid, "v": vec} for pid, vec in vectors.items()]
    return [
        ("UNWIND $rows AS row MATCH (p:Passage {id: row.id}) SET p.embedding = row.v, p.embedding_model = $model",
         rows, {"model": model}),
        (f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (p:Passage) ON p.embedding "
         f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
         None, {}),
    ]


# --------------------------------------------------------------------------- #
# Cypher file (for cypher-shell, no driver)
# --------------------------------------------------------------------------- #
def cypher_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if math.isfinite(v) else "null"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(cypher_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"`{str(k).replace('`', '``')}`: {cypher_value(x)}" for k, x in v.items()) + "}"
    return json.dumps(str(v), ensure_ascii=False)


def write_cypher(graph: dict, path: Path, extra: list | None = None) -> None:
    lines = ["// Generated by kg_to_neo4j.py. Run: cypher-shell -u neo4j -p <password> -f " + path.name]
    for query, rows, params in statements(graph) + (extra or []):
        for k, v in params.items():
            query = query.replace(f"${k}", cypher_value(v))
        if rows is None:
            lines.append(query + ";")
            continue
        for i in range(0, len(rows), BATCH):
            lines.append(query.replace("$rows", cypher_value(rows[i:i + BATCH]), 1) + ";")
    for query, params in cleanup_statements(graph):  # one pass is enough for most vaults
        for k, v in params.items():
            query = query.replace(f"${k}", cypher_value(v))
        lines.append(query.replace("WITH r LIMIT 10000 ", "WITH r ").replace("WITH n LIMIT 10000 ", "WITH n ")
                     .replace("WITH c LIMIT 10000 ", "WITH c ").replace("WITH x LIMIT 10000 ", "WITH x ") + ";")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Direct load with the official driver
# --------------------------------------------------------------------------- #
def load_into_neo4j(graph: dict, uri: str, user: str, password: str, database: str | None,
                    extra: list | None = None) -> dict:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("The Neo4j driver is not installed:  pip install neo4j\n"
                 "(or use --cypher FILE to write a Cypher script instead)")
    counts = defaultdict(int)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        if extra:  # an existing index with another dimension (different model) is rebuilt
            dim = len(extra[0][1][0]["v"])
            records, _, _ = driver.execute_query(
                "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options", name=VECTOR_INDEX,
                database_=database)
            if records:
                old = (records[0]["options"] or {}).get("indexConfig", {}).get("vector.dimensions")
                if old is not None and int(old) != dim:
                    driver.execute_query(f"DROP INDEX {VECTOR_INDEX}", database_=database)
                    counts["index_rebuilt"] = 1
        for query, rows, params in statements(graph) + (extra or []):
            if rows is None:
                driver.execute_query(query, database_=database)
                continue
            for i in range(0, len(rows), BATCH):
                driver.execute_query(query, rows=rows[i:i + BATCH], database_=database, **params)
            counts["rows"] += len(rows)
        for query, params in cleanup_statements(graph):
            while True:
                records, _, _ = driver.execute_query(query, database_=database, **params)
                n = records[0]["n"] if records else 0
                counts["removed"] += n
                if n == 0:
                    break
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kg_to_neo4j", description="graph.json -> Neo4j")
    p.add_argument("graph", help="the -kg folder written by vault2kg.py (or its graph.json)")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE"), help="default: the server's default")
    p.add_argument("--cypher", type=Path, metavar="FILE", help="write a Cypher script instead of connecting")
    p.add_argument("--embed", action="store_true", help="embed the passages and create a vector index")
    p.add_argument("--embed-base-url", help="embedding server (default: vLLM :8001, else Ollama, else OpenAI)")
    p.add_argument("--embed-model", help="embedding model (default: read from the server)")
    p.add_argument("--api-key", help="API key of the embedding server, if it needs one")
    a = p.parse_args(argv)
    graph, folder = load_graph(a.graph)
    extra = []
    if a.embed:
        base, model, key = find_embedder(a.embed_base_url, a.embed_model, a.api_key)
        vectors = embed_passages(graph.get("passages", []), base, model, key,
                                 folder / ".cache" / "passage_embeddings.json")
        extra = vector_statements(vectors, model)
        if vectors:
            print(f"Passage vectors: {model}, {len(next(iter(vectors.values())))} dimensions")
    if a.cypher:
        write_cypher(graph, a.cypher, extra)
        print(f"Wrote {a.cypher}. Run:  cypher-shell -u {a.user} -p <password> -f {a.cypher}")
        return 0
    password = a.password or getpass.getpass(f"Neo4j password for {a.user}@{a.uri}: ")
    counts = load_into_neo4j(graph, a.uri, a.user, password, a.database, extra)
    print(f"Loaded {len(graph['nodes'])} nodes, {len(graph['edges'])} relationships and "
          f"{len(graph['chunks'])} sections and {len(graph.get('passages', []))} passages into {a.uri}"
          + (f"; removed {counts['removed']} outdated items" if counts.get("removed") else ""))
    if extra:
        print(f"Vector index '{VECTOR_INDEX}' on Passage.embedding"
              + (" (rebuilt: the embedding size changed)" if counts.get("index_rebuilt") else "")
              + ". Embed queries with the same model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
