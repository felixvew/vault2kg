"""Offline tests for the follow-up scripts (viewer, GraphML, Neo4j, LightRAG)."""
import asyncio
import json
import re
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import kg_to_graphml
import kg_to_lightrag
import kg_to_neo4j
import kg_view
import vault2kg as v
from test_vault2kg import SAMPLE, FakeClient


@pytest.fixture()
def kg(tmp_path):
    vault = tmp_path / "vault"
    shutil.copytree(SAMPLE, vault)
    out = tmp_path / "vault-kg"
    v.VaultExtractor(v.Config(vault=vault, out=out, min_chars=50), FakeClient(),
                     progress=lambda m: None, llm_label="fake-model").run()
    return out


def graph_of(kg):
    return json.loads((kg / "graph.json").read_text())


def test_graph_json_contract(kg):
    g = graph_of(kg)
    assert g["format"] == "vault2kg-graph" and g["schema_version"] == v.SCHEMA_VERSION
    ids = {n["id"] for n in g["nodes"]}
    for n in g["nodes"]:
        assert {"id", "name", "kind", "type", "description", "aliases", "notes", "mentions", "degree"} <= set(n)
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids
    chunk_ids = {c["id"] for c in g["chunks"]}
    for c in g["chunks"]:
        assert {"id", "note", "path", "title", "heading_path", "text", "entities"} <= set(c)
        assert set(c["entities"]) <= ids
    assert all(set(n["mentions"]) <= chunk_ids for n in g["nodes"])


def test_missing_graph_gives_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="Run vault2kg.py"):
        kg_view.load_graph(str(tmp_path))


# ---------- viewer ----------
def test_view(kg):
    assert kg_view.main([str(kg)]) == 0
    page = (kg / "graph.html").read_text()
    blob = re.search(r'<script id="graph-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    data = json.loads(blob.replace("<\\/", "</"))
    assert len(data["nodes"]) == len(graph_of(kg)["nodes"]) and data["llm"] == "fake-model"


# ---------- GraphML ----------
def test_graphml(kg):
    assert kg_to_graphml.main([str(kg / "graph.json")]) == 0
    root = ET.parse(kg / "graph.graphml").getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    g = graph_of(kg)
    assert len(root.findall(".//g:node", ns)) == len(g["nodes"])
    assert len(root.findall(".//g:edge", ns)) == len(g["edges"])


# ---------- Neo4j ----------
def _cypher_rows(text):
    rows = []
    for stmt in text.split(";\n"):
        m = re.match(r"UNWIND (\[.*\]) AS row ", stmt, re.S)
        if m:
            lit = re.sub(r"`((?:[^`]|``)*)`: ", lambda k: json.dumps(k.group(1).replace("``", "`")) + ": ", m.group(1))
            rows.extend(json.loads(lit))
    return rows


def test_neo4j_cypher_file(kg, tmp_path):
    out = tmp_path / "load.cypher"
    assert kg_to_neo4j.main([str(kg), "--cypher", str(out)]) == 0
    cy = out.read_text()
    assert "$" not in cy.split("\n", 1)[1]                        # all parameters inlined
    assert "n:Entity:Tool" in cy and "n:Note:Person" in cy and "HAS_CHUNK" in cy
    rows = _cypher_rows(cy)                                          # every literal is valid
    g = graph_of(kg)
    assert {r["id"] for r in rows if "props" in r and "note" not in r and "id" in r} >= {n["id"] for n in g["nodes"]}
    assert any(r.get("props", {}).get("fm_status") == "active" for r in rows)
    assert cy.rstrip().endswith("DETACH DELETE x RETURN count(*) AS n;")
    assert "MERGE (p:Passage" in cy and "PART_OF" in cy


def test_neo4j_driver_load(kg, monkeypatch):
    calls = []

    class Result(tuple):
        pass

    class FakeDriver:
        def __init__(self, uri, auth):
            calls.append(("connect", uri, auth))
            self.deletes = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def verify_connectivity(self):
            pass

        def execute_query(self, query, database_=None, **params):
            calls.append((query, params, database_))
            if "DELETE" in query:
                self.deletes += 1
                return ([{"n": 3 if self.deletes == 1 else 0}], None, None)
            return ([], None, None)

    fake = types.ModuleType("neo4j")
    fake.GraphDatabase = types.SimpleNamespace(driver=lambda uri, auth: FakeDriver(uri, auth))
    monkeypatch.setitem(sys.modules, "neo4j", fake)
    assert kg_to_neo4j.main([str(kg), "--password", "pw", "--database", "kg"]) == 0
    assert calls[0] == ("connect", "bolt://localhost:7687", ("neo4j", "pw"))
    queries = [c for c in calls[1:]]
    assert all(db == "kg" for _, _, db in queries)
    assert all("$rows" not in q or "rows" in p for q, p, _ in queries)   # parameters, not string-built
    node_rows = [r for q, p, _ in queries if "MERGE (n:VaultNode" in q for r in p["rows"]]
    assert len(node_rows) == len(graph_of(kg)["nodes"])
    assert all(r["props"]["vault"] == "vault" for r in node_rows)


# ---------- LightRAG ----------
def _lightrag_checks(ckg):
    """The rules LightRAG's ainsert_custom_kg enforces (read from its source)."""
    aliases = {c["source_id"] for c in ckg["chunks"]}
    names = [e["entity_name"] for e in ckg["entities"]]
    assert len(names) == len(set(names))
    assert all(e["source_id"] in aliases or e["source_id"] == "UNKNOWN" for e in ckg["entities"])
    seen = set()
    for r in ckg["relationships"]:
        assert r["src_id"] != r["tgt_id"]                               # self-loops raise
        assert r["source_id"] in aliases                                # one alias, looked up as a whole
        assert r["weight"] >= 1.0                                       # < evidence count raises
        pair = tuple(sorted((r["src_id"], r["tgt_id"])))
        assert pair not in seen                                         # one relationship per pair
        seen.add(pair)


def test_lightrag_json(kg, tmp_path):
    out = tmp_path / "lr.json"
    assert kg_to_lightrag.main([str(kg), "--json", str(out)]) == 0
    ckg = json.loads(out.read_text())
    _lightrag_checks(ckg)
    names = {e["entity_name"] for e in ckg["entities"]}
    assert "#people" in names and "pipeline.png" not in names
    multi = [r for r in ckg["relationships"] if "," in r["keywords"]]
    assert multi                                                        # e.g. links to + organization + works at
    assert {"content", "source_id", "file_path", "chunk_order_index"} <= set(ckg["chunks"][0])


def test_lightrag_build_drops_bad_names():
    g = {"chunks": [{"id": "n#c0", "note": "note:n", "path": "n.md", "title": "n", "heading_path": [], "text": "x"}],
         "passages": [{"id": "n#c0.p0", "chunk": "n#c0", "note": "note:n", "path": "n.md", "embed_text": "n\n\nx",
                       "entities": []}],
         "nodes": [{"id": "note:n", "name": "n", "kind": "note", "type": "Note", "path": "n.md", "notes": ["n.md"],
                    "mentions": ["n#c0"]},
                   {"id": "ent:42", "name": "42", "kind": "entity", "type": "Other", "notes": ["n.md"],
                    "mentions": ["n#c0"]},
                   {"id": "ent:N", "name": "N ", "kind": "entity", "type": "Other", "notes": ["n.md"], "mentions": []}],
         "edges": [{"source": "note:n", "target": "ent:42", "type": "MENTIONS", "notes": ["n.md"], "chunks": []},
                   {"source": "note:n", "target": "ent:N", "type": "MENTIONS", "notes": ["n.md"], "chunks": []}]}
    ckg = kg_to_lightrag.build_custom_kg(g)
    names = [e["entity_name"] for e in ckg["entities"]]
    assert "42" not in names and len(names) == 2 and names[1] == "N (entity)"
    _lightrag_checks(ckg)


def test_lightrag_note_description_has_properties(kg, tmp_path):
    ckg = kg_to_lightrag.build_custom_kg(graph_of(kg))
    desc = next(e["description"] for e in ckg["entities"] if e["entity_name"] == "Pose Pipeline")
    assert "status: active" in desc and "start: 2026-03-01" in desc
    assert "type:" not in desc                                       # already the entity type
    assert not any("status: active" in c["content"] for c in ckg["chunks"])   # passages stay frontmatter-free
    out = tmp_path / "load.cypher"                                   # exact filtering stays in Neo4j
    assert kg_to_neo4j.main([str(kg), "--cypher", str(out)]) == 0
    assert any(r.get("props", {}).get("fm_status") == "active" for r in _cypher_rows(out.read_text(encoding="utf-8")))


def test_property_text_skips_technical_and_shared():
    notes = [{"kind": "note", "properties": {"source": "Report.pdf", "risk": f"R{i % 3}"}} for i in range(6)]
    shared = kg_to_lightrag.shared_properties(notes)
    assert shared == {"source": "Report.pdf"}
    props = {"source": "Report.pdf", "risk": "R3", "lead": "[[Anna Weber|Anna]]", "created": "2026-01-01T10:00",
             "uuid": "123e4567-e89b-12d3-a456-426614174000", "cssclasses": ["wide"], "systems": ["SAP", "Excel"],
             "nested": {"x": 1}, "data_source": "Excel", "start": "2026-03-01"}
    assert kg_to_lightrag.property_text(props, shared) == \
        "risk: R3; lead: Anna; systems: SAP, Excel; data source: Excel; start: 2026-03-01"
    assert len(kg_to_lightrag.property_text({f"k{i}": "v" * 100 for i in range(20)})) <= 600


def test_lightrag_missing_package_message(kg, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightrag", None)
    with pytest.raises(SystemExit, match="pip install lightrag-hku"):
        kg_to_lightrag.main([str(kg)])


def test_lightrag_uses_passages(kg, tmp_path):
    out = tmp_path / "lr.json"
    kg_to_lightrag.main([str(kg), "--json", str(out)])
    ckg = json.loads(out.read_text())
    g = graph_of(kg)
    assert [c["source_id"] for c in ckg["chunks"]] == [p["id"] for p in g["passages"]]
    by_id = {c["source_id"]: c["content"] for c in ckg["chunks"]}
    ent = {e["entity_name"]: e for e in ckg["entities"]}
    assert "Northwind" in by_id[ent["Northwind Labs"]["source_id"]]      # source passage names the entity
    assert "ViTPose" in by_id[ent["ViTPose"]["source_id"]]


def test_neo4j_passages_loaded(kg):
    sts = kg_to_neo4j.statements(graph_of(kg))
    q = {query: rows for query, rows, _ in sts}
    prow = next(rows for query, rows in q.items() if "MERGE (p:Passage" in query)
    assert len(prow) == len(graph_of(kg)["passages"])
    assert any("MENTIONED_IN]->(p)" in query for query in q)



# ---------- Neo4j --embed ----------
def _fake_neo4j(monkeypatch, existing_dim=None):
    calls = []

    class FakeDriver:
        def __init__(self, uri, auth):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def verify_connectivity(self):
            pass

        def execute_query(self, query, database_=None, **params):
            calls.append((query, params))
            if query.startswith("SHOW INDEXES"):
                if existing_dim is None:
                    return ([], None, None)
                return ([{"options": {"indexConfig": {"vector.dimensions": existing_dim}}}], None, None)
            if "DELETE" in query:
                return ([{"n": 0}], None, None)
            return ([], None, None)

    fake = types.ModuleType("neo4j")
    fake.GraphDatabase = types.SimpleNamespace(driver=lambda uri, auth: FakeDriver(uri, auth))
    monkeypatch.setitem(sys.modules, "neo4j", fake)
    return calls


def test_neo4j_embed_passages_and_vector_index(kg, monkeypatch):
    from test_vault2kg import _openai_server
    srv, url, emb_calls = _openai_server(["BAAI/bge-m3"], embed_dim=8)
    try:
        monkeypatch.setenv("VAULT2KG_VLLM_EMBED_URL", url)          # found like the vLLM embedding container
        monkeypatch.delenv("VAULT2KG_EMBED_BASE_URL", raising=False)
        calls = _fake_neo4j(monkeypatch)
        assert kg_to_neo4j.main([str(kg), "--password", "pw", "--embed"]) == 0
        n_passages = len(graph_of(kg)["passages"])
        set_rows = [r for q, p in calls if "SET p.embedding = row.v" in q for r in p["rows"]]
        assert len(set_rows) == n_passages and all(len(r["v"]) == 8 for r in set_rows)
        assert any(p.get("model") == "BAAI/bge-m3" for q, p in calls if "embedding_model" in q)
        idx = [q for q, _ in calls if q.startswith("CREATE VECTOR INDEX passage_embedding")]
        assert idx and "`vector.dimensions`: 8" in idx[0] and "'cosine'" in idx[0]
        q_order = [q for q, _ in calls]
        assert q_order.index(idx[0]) > max(i for i, q in enumerate(q_order) if "MERGE (p:Passage" in q)
        posts = [c for c in emb_calls if c[0] == "POST"]
        assert sum(len(c[2]["input"]) for c in posts) == n_passages

        # second run: vectors come from the cache, nothing is sent again
        _fake_neo4j(monkeypatch)
        kg_to_neo4j.main([str(kg), "--password", "pw", "--embed"])
        assert len([c for c in emb_calls if c[0] == "POST"]) == len(posts)

        # an index built for another embedding size is dropped and rebuilt
        calls3 = _fake_neo4j(monkeypatch, existing_dim=1024)
        kg_to_neo4j.main([str(kg), "--password", "pw", "--embed"])
        assert any(q == "DROP INDEX passage_embedding" for q, _ in calls3)
    finally:
        srv.shutdown()


def test_neo4j_embed_into_cypher_file(kg, tmp_path, monkeypatch):
    from test_vault2kg import _openai_server
    srv, url, _ = _openai_server(["nomic-embed-text"], owned_by="ollama", embed_dim=4)
    try:
        out = tmp_path / "e.cypher"
        assert kg_to_neo4j.main([str(kg), "--cypher", str(out), "--embed", "--embed-base-url", url]) == 0
        cy = out.read_text()
        assert "CREATE VECTOR INDEX passage_embedding" in cy and "`vector.dimensions`: 4" in cy
        assert '"nomic-embed-text"' in cy and "$" not in cy.split("\n", 1)[1]
        rows = [r for r in _cypher_rows(cy) if "v" in r]
        assert len(rows) == len(graph_of(kg)["passages"])
    finally:
        srv.shutdown()


def test_neo4j_embed_without_model_explains(kg, monkeypatch):
    for k in ("VAULT2KG_EMBED_BASE_URL", "VAULT2KG_EMBED_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAULT2KG_VLLM_EMBED_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    with pytest.raises(SystemExit, match="needs an embedding model"):
        kg_to_neo4j.main([str(kg), "--cypher", "x.cypher", "--embed"])


# ---------- LightRAG insert (fake LightRAG enforcing its rules) ----------
def _fake_lightrag(monkeypatch, embedding_sig="new"):
    import hashlib
    got = {"entity_edits": {}, "relation_edits": {}}

    class KV:
        def __init__(self):
            self.data = {}

        async def get_by_ids(self, ids):
            return [self.data.get(i) for i in ids]

    def sanitize(text, replacement_char="", *, strip=True):
        return text.strip() if strip else text

    def mdhash(content, prefix=""):
        return prefix + hashlib.md5(content.encode()).hexdigest()

    class FakeRAG:
        def __init__(self, **kw):
            got["init"] = kw
            self.text_chunks = KV()
            self.max_source_ids_per_entity = 200
            self.max_source_ids_per_relation = 200
            self.entities, self.relations = {}, {}

        async def initialize_storages(self):
            got["initialized"] = True

        async def ainsert_custom_kg(self, ckg):
            _lightrag_checks(ckg)
            got["kg"] = ckg
            alias = {}
            for c in ckg["chunks"]:
                cid = mdhash(sanitize(c["content"]), prefix="chunk-")
                self.text_chunks.data[cid] = c
                alias[c["source_id"]] = cid
            for e in ckg["entities"]:
                self.entities[e["entity_name"]] = alias.get(e["source_id"], "UNKNOWN")
            for r in ckg["relationships"]:
                self.relations[(r["src_id"], r["tgt_id"])] = r["weight"]
            fn = got["init"].get("llm_model_func")
            if callable(fn) and getattr(fn, "__name__", "") == "llm_model_func":   # the vLLM wrapper
                got["llm_answer"] = await fn("hi", system_prompt="s")

        async def aedit_entity(self, name, updated_data, allow_rename=True, allow_merge=False):
            assert set(updated_data) == {"source_id"} and not allow_rename
            assert name in self.entities
            ids = updated_data["source_id"].split("<SEP>")
            assert all(i in self.text_chunks.data for i in ids) and len(ids) <= 200
            got["entity_edits"][name] = ids

        async def aedit_relation(self, src, tgt, updated_data):
            if (src, tgt) not in self.relations:
                raise ValueError("relation not found")
            ids = updated_data["source_id"].split("<SEP>")
            assert updated_data["weight"] >= len(ids)          # LightRAG's evidence-count floor
            assert all(i in self.text_chunks.data for i in ids)
            got["relation_edits"][(src, tgt)] = (ids, updated_data["weight"])

        async def aquery(self, query, param=None, system_prompt=None):
            got["query"] = (query, param)
            return "the answer"

        async def finalize_storages(self):
            got["finalized"] = True

    class EmbeddingFunc:
        if embedding_sig == "new":
            def __init__(self, embedding_dim, max_token_size, func, model_name=None, send_dimensions=True):
                self.embedding_dim, self.func, self.model_name = embedding_dim, func, model_name
                self.max_token_size = max_token_size
        else:
            def __init__(self, embedding_dim, max_token_size, func):
                self.embedding_dim, self.func, self.model_name = embedding_dim, func, None
                self.max_token_size = max_token_size

    async def complete(model, prompt, system_prompt=None, history_messages=None, base_url=None, api_key=None, **kw):
        got["complete"] = (model, base_url, api_key)
        got["complete_kw"] = kw
        return "ok"

    names = ["lightrag", "lightrag.utils", "lightrag.constants", "lightrag.llm", "lightrag.llm.ollama",
             "lightrag.llm.openai", "lightrag.rerank"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["lightrag"].LightRAG = FakeRAG
    mods["lightrag"].QueryParam = lambda **kw: kw
    mods["lightrag.utils"].EmbeddingFunc = EmbeddingFunc
    mods["lightrag.utils"].normalize_entity_name = lambda x: x.strip()
    mods["lightrag.utils"].compute_mdhash_id = mdhash
    mods["lightrag.utils"].sanitize_text_for_encoding = sanitize
    mods["lightrag.constants"].GRAPH_FIELD_SEP = "<SEP>"
    mods["lightrag.llm.ollama"].ollama_model_complete = object()
    mods["lightrag.llm.ollama"].ollama_embed = lambda texts, embed_model, host: None
    mods["lightrag.llm.openai"].openai_complete_if_cache = complete
    mods["lightrag.llm.openai"].openai_embed = lambda texts, model, base_url, api_key: None

    async def cohere_rerank(query, documents, top_n=None, api_key=None, model=None, base_url=None,
                            extra_body=None, enable_chunking=False, max_tokens_per_doc=4096):
        return [{"index": i, "relevance_score": -i} for i in range(len(documents))]
    mods["lightrag.rerank"].cohere_rerank = cohere_rerank
    for k, m in mods.items():
        monkeypatch.setitem(sys.modules, k, m)
    return got


def test_lightrag_insert_with_ollama(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch, embedding_sig="old")
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: [])   # no vLLM -> Ollama
    monkeypatch.setattr(kg_to_lightrag, "_ollama_models", lambda host: ("qwen2.5:7b", "nomic-embed-text"))
    monkeypatch.setattr(kg_to_lightrag, "_post_json", lambda url, body, timeout=60: {"embeddings": [[0.0] * 768]})
    wd = tmp_path / "rag"
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(wd)]) == 0
    assert got["initialized"] and got["finalized"] and got["kg"]["entities"]
    assert got["init"]["llm_model_name"] == "qwen2.5:7b"
    assert got["init"]["embedding_func"].embedding_dim == 768
    assert wd.is_dir()


def _small_vllm(monkeypatch, rerank=False):
    """vLLM as in docker-compose.vllm-8gb.yml: an 8k chat model, a 512-token embedder and,
    with its rerank profile, a 512-token reranker."""
    served = {"http://localhost:8000/v1": ("Qwen/Qwen3-4B-AWQ", 8192), "http://localhost:8001/v1": ("BAAI/bge-base-en-v1.5", 512)}
    if rerank:
        served["http://localhost:8002/v1"] = ("cross-encoder/ms-marco-MiniLM-L-6-v2", 512)
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: [served[url][0]] if url in served else [])
    monkeypatch.setattr(kg_to_lightrag, "_max_model_len", lambda url, model, key: served[url][1])
    monkeypatch.setattr(kg_to_lightrag, "_embedding_dim", lambda url, model, key: 768)


def test_lightrag_fits_a_small_vllm_setup(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch)
    _small_vllm(monkeypatch)
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(tmp_path / "rag")]) == 0
    assert got["init"]["embedding_func"].max_token_size == 435           # 512 with tiktoken headroom
    assert got["init"]["summary_context_size"] == 5792 and got["init"]["summary_max_tokens"] == 1200
    assert got["complete_kw"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}  # Qwen3


def test_lightrag_query(monkeypatch, tmp_path, capsys):
    got = _fake_lightrag(monkeypatch)
    _small_vllm(monkeypatch)
    assert kg_to_lightrag.main(["--working-dir", str(tmp_path / "rag"), "--query", "Why?", "--mode", "local"]) == 0
    question, param = got["query"]
    assert question == "Why?" and param["mode"] == "local" and param["enable_rerank"] is False
    assert param["max_total_tokens"] == 6144                             # 8192 context minus room for the answer
    assert param["max_entity_tokens"] + param["max_relation_tokens"] < param["max_total_tokens"]
    assert "the answer" in capsys.readouterr().out and got["finalized"]
    with pytest.raises(SystemExit):                                      # neither a graph nor a question
        kg_to_lightrag.main(["--working-dir", str(tmp_path / "rag")])
    assert "rerank_model_func" not in got["init"]                        # no reranker unless asked for


def test_lightrag_query_with_reranker(monkeypatch, tmp_path, capsys):
    got = _fake_lightrag(monkeypatch)
    _small_vllm(monkeypatch, rerank=True)
    assert kg_to_lightrag.main(["--working-dir", str(tmp_path / "rag"), "--query", "Why?", "--rerank"]) == 0
    assert got["query"][1]["enable_rerank"] is True
    func = got["init"]["rerank_model_func"]
    assert func.keywords["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert func.keywords["base_url"] == "http://localhost:8002/v1/rerank"
    assert func.keywords["enable_chunking"] and func.keywords["max_tokens_per_doc"] == 371   # 512: question + passage
    assert func.keywords["extra_body"] == {"truncate_prompt_tokens": 512}
    assert got["init"]["min_rerank_score"] == 0.0                         # raw scores: reorder, never drop
    assert "Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2" in capsys.readouterr().out


def test_lightrag_rerank_without_reranker_explains(monkeypatch, tmp_path):
    _fake_lightrag(monkeypatch)
    _small_vllm(monkeypatch)                                              # nothing on :8002
    with pytest.raises(SystemExit, match="--profile rerank"):
        kg_to_lightrag.main(["--working-dir", str(tmp_path / "rag"), "--query", "Why?", "--rerank"])


def test_lightrag_vllm_provider(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch)
    served = {"http://gpu:8000/v1": ["Qwen/Qwen2.5-7B-Instruct"], "http://gpu:8001/v1": ["BAAI/bge-m3"]}
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: served.get(url, []))
    monkeypatch.setattr(kg_to_lightrag, "_embedding_dim", lambda url, model, key: 1024)
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(tmp_path / "rag"),
                                "--base-url", "http://gpu:8000/v1", "--embed-base-url", "http://gpu:8001/v1"]) == 0
    assert got["init"]["llm_model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert got["init"]["embedding_func"].embedding_dim == 1024
    assert got["init"]["embedding_func"].model_name == "BAAI/bge-m3"
    assert got["init"]["embedding_func"].func.keywords == {"model": "BAAI/bge-m3", "base_url": "http://gpu:8001/v1",
                                                           "api_key": "not_needed"}
    assert got["complete"] == ("Qwen/Qwen2.5-7B-Instruct", "http://gpu:8000/v1", "not_needed")


def test_lightrag_links_every_passage(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch)
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: ["m"] if "8000" in url else ["e"])
    monkeypatch.setattr(kg_to_lightrag, "_embedding_dim", lambda url, model, key: 8)
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(tmp_path / "rag")]) == 0
    g = graph_of(kg)
    passage_text = {p["id"]: p["embed_text"] for p in g["passages"]}
    chunk_by_content = {}
    for c in got["kg"]["chunks"]:
        chunk_by_content[c["source_id"]] = c["content"]
    # MediaPipe is named in passages of several notes: all of them are linked now
    mp = got["entity_edits"]["MediaPipe"]
    contents = [got_c["content"] for got_c in got["kg"]["chunks"]
                if any(cid.endswith(__import__("hashlib").md5(got_c["content"].encode()).hexdigest()) for cid in mp)]
    assert len(mp) >= 3 and all("MediaPipe" in c for c in contents)
    assert len({c.split("\n", 1)[0].split(" > ")[0] for c in contents}) >= 2       # from several notes
    # a note entity is linked to its own passages too
    assert len(got["entity_edits"]["Pose Pipeline"]) >= 2
    # relationship sources are passages of both ends, weight raised to the evidence count
    assert got["relation_edits"]
    for (src, tgt), (ids, weight) in got["relation_edits"].items():
        assert weight >= len(ids) >= 2


def test_lightrag_single_source_skips_linking(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch)
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: ["m"] if "8000" in url else ["e"])
    monkeypatch.setattr(kg_to_lightrag, "_embedding_dim", lambda url, model, key: 8)
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(tmp_path / "rag"), "--single-source"]) == 0
    assert got["kg"] and not got["entity_edits"] and not got["relation_edits"]


def test_lightrag_chunk_id_mismatch_falls_back(kg, monkeypatch, tmp_path):
    got = _fake_lightrag(monkeypatch)
    sys.modules["lightrag.utils"].compute_mdhash_id = lambda content, prefix="": prefix + "other"
    monkeypatch.setattr(kg_to_lightrag, "_served_models", lambda url, key: ["m"] if "8000" in url else ["e"])
    monkeypatch.setattr(kg_to_lightrag, "_embedding_dim", lambda url, model, key: 8)
    assert kg_to_lightrag.main([str(kg), "--working-dir", str(tmp_path / "rag")]) == 0
    assert got["kg"] and not got["entity_edits"]                                   # insert kept, no bad edits
