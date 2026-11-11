#!/usr/bin/env python3
"""kg_to_lightrag - load graph.json into LightRAG (as a custom knowledge graph).

    pip install lightrag-hku
    python src/kg_to_lightrag.py ~/Documents/MyVault-kg --working-dir ./rag

    # or only write LightRAG's custom-KG JSON (no lightrag needed):
    python src/kg_to_lightrag.py ~/Documents/MyVault-kg --json lightrag.json

LightRAG embeds the chunks, entities and relationships itself, so it needs an
embedding model, and an LLM for answering queries later. --provider:
  auto    (default) vLLM if a container answers on localhost:8000, else Ollama
  vllm    any OpenAI-compatible server: --base-url (chat, default :8000/v1) and
          --embed-base-url (embeddings, default :8001/v1); models are read from
          the servers, the embedding size is detected
  ollama  the local Ollama, models picked from what is installed
  openai  OpenAI with OPENAI_API_KEY

How the graph is mapped (LightRAG's insert_custom_kg rules):
  - LightRAG's chunks are the retrieval passages (small, "Title > Heading"
    header included), so its vector search works on focused text
  - entities are keyed by name: names are made unique; a tag becomes "#tag"
  - insert_custom_kg accepts ONE source passage per entity / relationship.
    Native LightRAG links an entity to every chunk it occurs in (up to 200),
    and its queries rely on that: candidate chunks shared by several matched
    entities rank first, then the rest by vector similarity. So after the
    insert, every entity and relationship is patched with ALL its passages via
    LightRAG's own aedit_entity / aedit_relation (--single-source skips this).
    An entity's passages: those naming it (name or alias) and, for a note, its
    own passages. A relationship's: passages that belong to both ends.
  - LightRAG keeps one relationship per pair of entities: several edges between
    the same two nodes are combined (types become the keywords)
  - weight = number of notes supporting it (LightRAG rejects weights below the
    evidence count), self-loops and empty names are dropped
  - attachments (images, PDFs) are left out
  - a note's description includes its plain frontmatter ("status: active; risk:
    R3"), so the note entity can be found by it; technical fields and values
    nearly every note shares are left out. Passages stay frontmatter-free.
    For exact filtering ("all notes with risk R3") use Neo4j's fm_* properties.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable


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


TECHNICAL_KEYS = {"type", "cssclasses", "cssclass", "publish", "position", "permalink", "uuid", "id", "created",
                  "updated", "modified", "date_created", "date_modified", "last_modified", "created_at", "updated_at"}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def _value_text(value: Any) -> str:
    """A property value as text: lists joined, [[Link|Shown]] -> Shown."""
    text = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text).strip()


def shared_properties(notes: list[dict], share: float = 0.9, min_notes: int = 5) -> dict[str, str]:
    """key -> value that nearly every note has (source file, version): it doesn't tell
    one note from another, so it stays out of the descriptions."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for n in notes:
        for k, v in (n.get("properties") or {}).items():
            counts[k][_value_text(v)] += 1
    need = max(min_notes, share * len(notes))
    out = {}
    for k, values in counts.items():
        value, c = max(values.items(), key=lambda kv: kv[1])
        if c >= need:
            out[k] = value
    return out


def property_text(props: dict, shared: dict[str, str] | None = None, limit: int = 600) -> str:
    """A note's plain frontmatter as one line for its LightRAG description, so the note
    entity can be found by what its properties say. Left out: technical fields, IDs,
    timestamps, nested values, and values nearly every note shares. Exact filtering on
    properties is Neo4j's job (fm_* properties), not LightRAG's."""
    parts = []
    for k, v in props.items():
        if re.sub(r"[^0-9a-z]+", "_", k.casefold()).strip("_") in TECHNICAL_KEYS or isinstance(v, dict):
            continue
        text = _value_text(v)
        if not text or UUID_RE.fullmatch(text) or TIMESTAMP_RE.match(text) or (shared or {}).get(k) == text:
            continue
        parts.append(f"{k.replace('_', ' ')}: {text[:150]}")
    line = "; ".join(parts)
    return line if len(line) <= limit else line[:limit].rsplit(";", 1)[0]


def _default_normalize(name: str) -> str:
    """Close to LightRAG's normalize_entity_name for the cases that matter here:
    trimmed, single spaces, and 1-2 character numbers become empty."""
    name = re.sub(r"\s+", " ", name.replace(" ", " ")).strip()
    return "" if re.fullmatch(r"[\d.]{1,2}", name) else name


def build_custom_kg(graph: dict, normalize: Callable[[str], str] | None = None,
                    links: dict | None = None) -> dict:
    """graph.json -> the dict LightRAG's insert_custom_kg() expects.

    If `links` is a dict, it is filled with every passage of each entity and
    relationship: {"entities": {name: [passage ids]},
                   "relationships": [(src, tgt, [passage ids], weight)]}.
    """
    normalize = normalize or _default_normalize
    chunks, first_of_note, first_of_section = [], {}, {}
    order: dict[str, int] = defaultdict(int)
    by_entity: dict[str, list[str]] = defaultdict(list)
    for psg in graph.get("passages", []):
        chunks.append({"content": psg["embed_text"], "source_id": psg["id"], "file_path": psg["path"],
                       "chunk_order_index": order[psg["note"]]})
        order[psg["note"]] += 1
        first_of_note.setdefault(psg["note"], psg["id"])
        first_of_section.setdefault(psg["chunk"], psg["id"])
        for e in psg.get("entities", []):
            by_entity[e].append(psg["id"])
    note_id = {n.get("path"): n["id"] for n in graph["nodes"] if n["kind"] == "note" and n.get("path")}

    def source_for(node_ids: list[str], sections: list[str], notes: list[str]) -> str:
        """First passage naming all the given nodes, else one naming any of them,
        else the first passage of a section they were found in, else of their note."""
        sets = [set(by_entity.get(n, [])) for n in node_ids]
        both = set.intersection(*sets) if sets and all(sets) else set()
        for pid in sorted(both) or sorted(set().union(*sets) if sets else ()):
            return pid
        for sec in sections:
            if sec in first_of_section:
                return first_of_section[sec]
        for nid in node_ids + [note_id.get(p) for p in notes]:
            if nid in first_of_note:
                return first_of_note[nid]
        return "UNKNOWN"

    names: dict[str, str] = {}
    used: set[str] = set()
    for n in graph["nodes"]:
        if n["kind"] == "file":
            continue
        base = normalize(("#" + n["name"]) if n["kind"] == "tag" else n["name"])
        if not base:
            continue
        name, k = base, 2
        while name.casefold() in used:
            name = normalize(f"{base} ({n['kind']})" if k == 2 else f"{base} ({n['kind']} {k})")
            k += 1
        used.add(name.casefold())
        names[n["id"]] = name

    name_to_id = {nm: i for i, nm in names.items()}
    shared = shared_properties([n for n in graph["nodes"] if n["kind"] == "note"])
    entities = []
    for n in graph["nodes"]:
        if n["id"] not in names:
            continue
        desc = n.get("description") or {
            "note": f"Note '{n['name']}' in the Obsidian vault.",
            "tag": f"Tag #{n['name']} used in the Obsidian vault.",
        }.get(n["kind"], n["name"])
        if n["kind"] == "note" and (props := property_text(n.get("properties") or {}, shared)):
            desc = f"{desc.rstrip()} Properties: {props}."
        entities.append({"entity_name": names[n["id"]], "entity_type": n.get("type") or "Other",
                         "description": desc,
                         "source_id": source_for([n["id"]], n.get("mentions", []), n.get("notes", [])),
                         "file_path": n.get("path") or (n["notes"][0] if n.get("notes") else "vault")})

    own: dict[str, list[str]] = defaultdict(list)
    for psg in graph.get("passages", []):
        own[psg["note"]].append(psg["id"])

    def passages_of(node_id: str) -> list[str]:
        """Passages naming the node, plus a note's own passages."""
        return list(dict.fromkeys(by_entity.get(node_id, []) + own.get(node_id, [])))

    if links is not None:
        links["entities"] = {}
        for e in entities:
            nid = name_to_id[e["entity_name"]]
            ids = passages_of(nid)
            if e["source_id"] != "UNKNOWN" and e["source_id"] not in ids:
                ids.insert(0, e["source_id"])
            links["entities"][e["entity_name"]] = ids
        links["relationships"] = []

    pairs: dict[tuple[str, str], dict] = {}
    for e in graph["edges"]:
        a, b = names.get(e["source"]), names.get(e["target"])
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        p = pairs.setdefault(key, {"src_id": a, "tgt_id": b, "types": [], "descriptions": [], "notes": set(),
                                   "chunks": [], "nodes": [e["source"], e["target"]]})
        if e["type"] not in p["types"]:
            p["types"].append(e["type"])
        text = e.get("description") or f"{a} {e['type'].lower().replace('_', ' ')} {b}"
        if text not in p["descriptions"]:
            p["descriptions"].append(text)
        p["notes"].update(e.get("notes", []))
        p["chunks"] += e.get("chunks", [])
    relationships = []
    for p in pairs.values():
        source = source_for(p["nodes"], p["chunks"], sorted(p["notes"]))
        if links is not None:
            a_ids, b_ids = passages_of(p["nodes"][0]), set(passages_of(p["nodes"][1]))
            both = [x for x in a_ids if x in b_ids]
            if source != "UNKNOWN" and source not in both:
                both.insert(0, source)
            links["relationships"].append((p["src_id"], p["tgt_id"], both, float(max(1, len(p["notes"])))))
        relationships.append({
            "src_id": p["src_id"], "tgt_id": p["tgt_id"],
            "description": " ".join(p["descriptions"][:5]),
            "keywords": ", ".join(t.lower().replace("_", " ") for t in p["types"]),
            # LightRAG: weight must be >= number of distinct evidence sources (1 here)
            "weight": float(max(1, len(p["notes"]))),
            "source_id": source,
            "file_path": sorted(p["notes"])[0] if p["notes"] else "vault",
        })
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


# --------------------------------------------------------------------------- #
# Insert with LightRAG
# --------------------------------------------------------------------------- #
def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else "http://" + host
    host = host if re.search(r":\d+$", host) else host + ":11434"
    return host.replace("0.0.0.0", "localhost")


def _post_json(url: str, body: dict, timeout: float = 60) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _ollama_models(host: str) -> tuple[str | None, str | None]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except OSError:
        sys.exit(f"Ollama is not reachable at {host}. Start it, or use --provider openai.")
    is_embed = lambda n: "embed" in n.lower() or n.lower().startswith(("bge", "nomic", "mxbai", "all-minilm"))  # noqa: E731
    chat = [n for n in names if not is_embed(n)]
    embed = [n for n in names if is_embed(n)]
    for pref in ("qwen3", "qwen2.5", "llama3.1", "gemma3", "mistral", "llama3.2"):
        hits = [n for n in chat if n.lower().startswith(pref)]
        if hits:
            return hits[0], (embed[0] if embed else None)
    return (chat[0] if chat else None), (embed[0] if embed else None)


def _served_models(base_url: str, api_key: str | None) -> list[str]:
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return [m["id"] for m in json.load(resp).get("data", []) if m.get("id")]
    except (OSError, ValueError, AttributeError):
        return []


def _max_model_len(base_url: str, model: str, api_key: str | None) -> int | None:
    """The context length a vLLM server reports for a model (None for other servers)."""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            for m in json.load(resp).get("data", []):
                if m.get("id") == model and isinstance(m.get("max_model_len"), int):
                    return m["max_model_len"]
    except (OSError, ValueError, AttributeError):
        pass
    return None


def query_param(QueryParam, mode: str, context_len: int | None, rerank: bool = False) -> Any:
    """LightRAG's query settings. Its default budget (30k tokens) assumes a large model:
    with a known context, entities, relations and passages are sized to fit it, with
    room left for the answer. Reranking only with a reranker (--rerank)."""
    kw: dict[str, Any] = {"mode": mode, "enable_rerank": bool(rerank)}
    if context_len:
        total = max(2000, context_len - 2048)
        kw.update(max_total_tokens=total, max_entity_tokens=total // 5, max_relation_tokens=total * 3 // 10)
    return QueryParam(**kw)


def _embedding_dim(base_url: str, model: str, api_key: str | None) -> int:
    req = urllib.request.Request(f"{base_url.rstrip('/')}/embeddings",
                                 data=json.dumps({"model": model, "input": ["dimension probe"]}).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return len(json.load(resp)["data"][0]["embedding"])


def _embedding_func(EmbeddingFunc, dim: int, func, model: str, max_tokens: int = 8192):
    """max_tokens: LightRAG cuts longer inputs to this, so it must fit the embedding model."""
    try:  # newer LightRAG takes the model name; older versions don't
        return EmbeddingFunc(embedding_dim=dim, max_token_size=max_tokens, func=func, model_name=model,
                             send_dimensions=False)
    except TypeError:
        return EmbeddingFunc(embedding_dim=dim, max_token_size=max_tokens, func=func)


def _binding(module: str, package: str):
    """LightRAG's model bindings need a client package that lightrag-hku doesn't install."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        sys.exit(f"LightRAG needs the '{package}' package for this provider:  pip install {package}\n({exc})")


RERANK_URL = "http://localhost:8002/v1"  # the optional reranker in the docker-compose files


def make_rerank(args) -> dict:
    """LightRAG arguments for a cross-encoder reranker (--rerank), or {} without one. It
    reorders the passages LightRAG found by how well each answers the question, before
    they fill the answer's context. Any server with a Jina/Cohere-style /rerank endpoint
    works; vLLM serves one."""
    args.rerank_on = False
    if not (getattr(args, "rerank", False) or getattr(args, "rerank_url", None)):
        return {}
    url = (args.rerank_url or os.environ.get("VAULT2KG_RERANK_URL", RERANK_URL)).rstrip("/")
    key = args.api_key or os.environ.get("VAULT2KG_API_KEY") or "not_needed"
    model = args.rerank_model or next(iter(_served_models(url, key)), None)
    if not model:
        sys.exit(f"No reranker answers at {url}. Start the one in the compose file:\n"
                 "  docker compose -f docker-compose.vllm-8gb.yml --profile rerank up -d\n"
                 "or pass --rerank-url / --rerank-model.")
    max_len = _max_model_len(url, model, key)
    rerank = _binding("lightrag.rerank", "lightrag-hku")
    # The question and a passage must fit the model together. LightRAG splits longer passages,
    # counting with tiktoken (fewer tokens than most rerankers' tokenizers): keep 15% spare.
    per_doc = max(64, int(max_len * 0.85) - 64) if max_len else 480
    func = partial(rerank.cohere_rerank, model=model, base_url=f"{url}/rerank", api_key=key,
                   enable_chunking=True, max_tokens_per_doc=per_doc,
                   extra_body={"truncate_prompt_tokens": max_len} if max_len else None)
    print(f"Reranker: {model} ({url}" + (f", {max_len} tokens" if max_len else "") + ")")
    args.rerank_on = True
    # Cross-encoders return raw scores, not 0-1 probabilities: reorder, never drop by score.
    return {"rerank_model_func": func, "min_rerank_score": 0.0}


def make_vllm_rag(args, LightRAG, EmbeddingFunc, extra_kw: dict | None = None) -> Any:
    binding = _binding("lightrag.llm.openai", "openai")
    openai_complete_if_cache, openai_embed = binding.openai_complete_if_cache, binding.openai_embed
    base = args.base_url or os.environ.get("VAULT2KG_BASE_URL", "http://localhost:8000/v1")
    embed_base = args.embed_base_url or os.environ.get("VAULT2KG_EMBED_BASE_URL", "http://localhost:8001/v1")
    key = args.api_key or os.environ.get("VAULT2KG_API_KEY") or "not_needed"
    chat = args.model or next(iter(_served_models(base, key)), None)
    emb = args.embed_model or next(iter(_served_models(embed_base, key)), None)
    if not chat:
        sys.exit(f"No chat model answers at {base}. Start the vLLM container or pass --base-url/--model.")
    if not emb:
        sys.exit(f"No embedding model answers at {embed_base}. Start a vLLM container with "
                 "--runner pooling (see docker-compose.vllm.yml) or pass --embed-base-url/--embed-model.")
    dim = args.embed_dim or _embedding_dim(embed_base, emb, key)
    chat_len, emb_len = _max_model_len(base, chat, key), _max_model_len(embed_base, emb, key)
    args.context_len = chat_len
    print(f"LightRAG with vLLM: {chat} ({base}" + (f", {chat_len} tokens" if chat_len else "") +
          f"), embeddings {emb} ({dim} dimensions" + (f", {emb_len} tokens" if emb_len else "") + f", {embed_base})")
    # Qwen3 thinks before answering by default; for keywords and answers that only costs time
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in chat.lower() else None

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if extra:
            kwargs["extra_body"] = {**extra, **(kwargs.get("extra_body") or {})}
        return await openai_complete_if_cache(model=chat, prompt=prompt, system_prompt=system_prompt,
                                              history_messages=history_messages or [], base_url=base,
                                              api_key=key, **kwargs)

    # LightRAG cuts embedding inputs to max_token_size, counted with tiktoken. Most embedders'
    # tokenizers produce more tokens for the same text, so keep 15% to spare.
    emb_tokens = int(emb_len * 0.85) if emb_len else 8192
    # LightRAG's summary defaults assume a 32k+ model; keep those LLM calls inside this one
    sizing = {"summary_context_size": max(2000, min(12000, chat_len - 2400)),
              "summary_max_tokens": min(1200, chat_len // 6)} if chat_len else {}
    embed_fn = getattr(openai_embed, "func", openai_embed)  # newer versions wrap it in an EmbeddingFunc
    return LightRAG(working_dir=args.working_dir, llm_model_func=llm_model_func, llm_model_name=chat,
                    embedding_func=_embedding_func(EmbeddingFunc, dim,
                                                   partial(embed_fn, model=emb, base_url=embed_base, api_key=key),
                                                   emb, emb_tokens), **sizing, **(extra_kw or {}))


def make_rag(args) -> Any:
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        sys.exit("LightRAG is not installed:  pip install lightrag-hku\n"
                 "(or use --json FILE to only write the custom-KG JSON)")
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)
    provider = args.provider
    if provider == "auto":
        url = args.base_url or os.environ.get("VAULT2KG_BASE_URL", "http://localhost:8000/v1")
        provider = "vllm" if _served_models(url, args.api_key) else "ollama"
    rerank = make_rerank(args)
    if provider == "vllm":
        return make_vllm_rag(args, LightRAG, EmbeddingFunc, rerank)
    if provider == "openai":
        binding = _binding("lightrag.llm.openai", "openai")
        return LightRAG(working_dir=args.working_dir, llm_model_func=binding.gpt_4o_mini_complete,
                        embedding_func=binding.openai_embed, **rerank)
    binding = _binding("lightrag.llm.ollama", "ollama")
    ollama_embed, ollama_model_complete = binding.ollama_embed, binding.ollama_model_complete
    host = _ollama_host()
    chat, emb = _ollama_models(host)
    chat, emb = args.model or chat, args.embed_model or emb
    if not chat or not emb:
        sys.exit("Ollama needs a chat model and an embedding model, e.g.:\n"
                 "  ollama pull qwen2.5:7b && ollama pull nomic-embed-text")
    dim = args.embed_dim or len(_post_json(f"{host}/api/embed", {"model": emb, "input": ["dimension probe"]})
                                ["embeddings"][0])
    print(f"LightRAG with Ollama: {chat}, embeddings {emb} ({dim} dimensions)")
    args.context_len = 8192  # num_ctx below
    embed_fn = getattr(ollama_embed, "func", ollama_embed)  # newer versions wrap it in an EmbeddingFunc
    return LightRAG(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=chat,
        llm_model_kwargs={"host": host, "options": {"num_ctx": 8192}},
        embedding_func=_embedding_func(EmbeddingFunc, dim, partial(embed_fn, embed_model=emb, host=host), emb),
        **rerank,
    )


async def link_all_sources(rag: Any, custom_kg: dict, links: dict, log=print) -> dict:
    """Give every entity / relationship all of its passages (LightRAG's native
    shape), using LightRAG's own edit functions so its chunk tracking stays in sync."""
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding
    try:
        from lightrag.constants import GRAPH_FIELD_SEP as sep
    except ImportError:
        sep = "<SEP>"
    # the chunk ids LightRAG assigned during the insert (same formula it uses)
    chunk_id = {c["source_id"]: compute_mdhash_id(sanitize_text_for_encoding(c["content"]), prefix="chunk-")
                for c in custom_kg["chunks"]}
    sample = list(dict.fromkeys(chunk_id.values()))[:50]
    if sample:
        found = await rag.text_chunks.get_by_ids(sample)
        if sum(1 for f in found if f) < 0.9 * len(sample):
            log("Could not match LightRAG's chunk ids (different LightRAG version?); "
                "kept one source passage per entity.")
            return {"skipped": 1}
    cap_e = int(getattr(rag, "max_source_ids_per_entity", 200) or 200)
    cap_r = int(getattr(rag, "max_source_ids_per_relation", 200) or 200)
    ids_of = lambda aliases, cap: list(dict.fromkeys(chunk_id[a] for a in aliases if a in chunk_id))[:cap]  # noqa: E731
    stats = defaultdict(int)
    todo_e = [(n, ids_of(a, cap_e)) for n, a in links["entities"].items()]
    todo_e = [(n, ids) for n, ids in todo_e if len(ids) > 1]
    todo_r = [(s, t, ids_of(a, cap_r), w) for s, t, a, w in links["relationships"]]
    todo_r = [(s, t, ids, w) for s, t, ids, w in todo_r if len(ids) > 1]
    if todo_e or todo_r:
        log(f"Linking all passages: {len(todo_e)} entities and {len(todo_r)} relationships appear in more than one")
    for k, (name, ids) in enumerate(todo_e, 1):
        try:
            await rag.aedit_entity(name, {"source_id": sep.join(ids)}, allow_rename=False)
            stats["entities"] += 1
        except Exception as exc:  # noqa: BLE001 - one entity must not stop the rest
            stats["entity_errors"] += 1
            if stats["entity_errors"] <= 3:
                log(f"  could not update '{name}': {str(exc)[:150]}")
        if k % 200 == 0:
            log(f"  {k}/{len(todo_e)} entities")
    for src, tgt, ids, weight in todo_r:
        update = {"source_id": sep.join(ids), "weight": float(max(weight, len(ids)))}  # weight >= sources
        for a, b in ((src, tgt), (tgt, src)):
            try:
                await rag.aedit_relation(a, b, update)
                stats["relationships"] += 1
                break
            except Exception as exc:  # noqa: BLE001
                err = exc
        else:
            stats["relationship_errors"] += 1
            if stats["relationship_errors"] <= 3:
                log(f"  could not update '{src}' - '{tgt}': {str(err)[:150]}")
    return dict(stats)


async def ask(question: str, args) -> str:
    """Query the LightRAG in --working-dir with the same models it was loaded with."""
    rag = make_rag(args)
    from lightrag import QueryParam
    await rag.initialize_storages()
    try:
        return await rag.aquery(question, param=query_param(QueryParam, args.mode, args.context_len,
                                                            getattr(args, "rerank_on", False)))
    finally:
        await rag.finalize_storages()


async def insert(custom_kg: dict, args, links: dict | None = None) -> dict:
    rag = make_rag(args)
    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(custom_kg)
        return await link_all_sources(rag, custom_kg, links) if links else {}
    finally:
        await rag.finalize_storages()


def lightrag_normalizer() -> Callable[[str], str] | None:
    try:
        from lightrag.utils import normalize_entity_name
        return normalize_entity_name
    except ImportError:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kg_to_lightrag", description="graph.json -> LightRAG")
    p.add_argument("graph", nargs="?", help="the -kg folder written by vault2kg.py (or its graph.json)")
    p.add_argument("--query", metavar="QUESTION",
                   help="ask the LightRAG in --working-dir a question (instead of loading a graph)")
    p.add_argument("--mode", choices=["mix", "hybrid", "local", "global", "naive"], default="mix",
                   help="query mode for --query (default mix: graph and passage search together)")
    p.add_argument("--rerank", action="store_true",
                   help="for --query: reorder the passages found by how well they answer the question, with a "
                        "cross-encoder (the vLLM reranker on :8002 in the compose files)")
    p.add_argument("--rerank-url", help=f"server with a /rerank endpoint (default {RERANK_URL}); implies --rerank")
    p.add_argument("--rerank-model", help="reranker model (default: read from the server)")
    p.add_argument("--working-dir", default="./rag_storage", help="LightRAG working directory (default ./rag_storage)")
    p.add_argument("--provider", choices=["auto", "vllm", "ollama", "openai"], default="auto")
    p.add_argument("--base-url", help="vLLM / OpenAI-compatible chat server (default http://localhost:8000/v1)")
    p.add_argument("--embed-base-url", help="embedding server (default http://localhost:8001/v1)")
    p.add_argument("--api-key", help="API key of the vLLM server, if it was started with --api-key")
    p.add_argument("--model", help="chat model for LightRAG (default: read from the server / Ollama)")
    p.add_argument("--embed-model", help="embedding model (default: read from the server / Ollama)")
    p.add_argument("--embed-dim", type=int, help="embedding size (default: detected)")
    p.add_argument("--json", type=Path, metavar="FILE", help="only write the custom-KG JSON, don't insert")
    p.add_argument("--single-source", action="store_true",
                   help="keep one source passage per entity/relationship (skip linking all passages)")
    a = p.parse_args(argv)
    a.context_len = None  # set by make_rag when the chat model's context is known
    if a.query:
        print(asyncio.run(ask(a.query, a)))
        return 0
    if not a.graph:
        p.error("give the -kg folder to load, or --query QUESTION to ask")
    graph, _ = load_graph(a.graph)
    links = None if (a.json or a.single_source) else {}
    custom_kg = build_custom_kg(graph, None if a.json else lightrag_normalizer(), links)
    n = f"{len(custom_kg['entities'])} entities, {len(custom_kg['relationships'])} relationships, " \
        f"{len(custom_kg['chunks'])} chunks"
    if a.json:
        a.json.write_text(json.dumps(custom_kg, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {a.json} ({n}). Load it with rag.insert_custom_kg(json.load(open(...))).\n"
              "Note: this file has one source passage per entity (LightRAG's custom-KG format); "
              "inserting with this script instead links all passages.")
        return 0
    stats = asyncio.run(insert(custom_kg, a, links))
    print(f"Inserted {n} into LightRAG at {a.working_dir}")
    if stats.get("entities") or stats.get("relationships"):
        print(f"Linked all passages for {stats.get('entities', 0)} entities and "
              f"{stats.get('relationships', 0)} relationships"
              + (f" ({stats.get('entity_errors', 0) + stats.get('relationship_errors', 0)} failed)"
                 if stats.get("entity_errors") or stats.get("relationship_errors") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
