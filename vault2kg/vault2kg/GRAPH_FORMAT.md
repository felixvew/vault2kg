# graph.json format

`vault2kg.py` writes one file. Everything else (the viewer, Neo4j, LightRAG, GraphML) reads it, so this format is the contract between them.

```jsonc
{
  "format": "vault2kg-graph",
  "schema_version": "2.0",
  "vault": "MyVault",                 // folder name of the vault
  "llm": "Ollama · qwen2.5:14b",      // null when built without an LLM
  "generated_at": "2026-11-10T18:00:00+00:00",
  "stats": { "notes": 120, "nodes": 640, "entities": 410, "tags": 35, "files": 12,
             "edges": 2100, "chunks": 380, "passages": 1150, "merged": 27 },
  "nodes":    [ ... ],
  "edges":    [ ... ],
  "chunks":   [ ... ],   // sections: what the LLM read
  "passages": [ ... ]    // retrieval passages: what you embed
}
```

## nodes

| field | |
|---|---|
| `id` | Stable ID. `note:<path without .md>`, `ent:<lowercased name>`, `tag:<tag>`, `file:<path>` |
| `name` | Display name. A note uses its file name, or its path when two notes share a name |
| `kind` | `note`, `entity` (found by the LLM), `tag` or `file` (attachment) |
| `type` | Person, Organization, Project, Tool, Concept, Method, Topic, Event, Place, Document, Other. For notes: the frontmatter `type:` if set, else what the LLM found, else `Note` |
| `description` | Up to three descriptions joined; `descriptions` holds all of them |
| `aliases` | Other names: note aliases, link texts, short forms and merged duplicates |
| `notes` | Paths of the notes this node appears in |
| `mentions` | IDs of the chunks it was found in |
| `degree` | Number of edges |
| `path`, `resolved` | Notes only. `resolved: false` means it's linked but doesn't exist yet |
| `properties` | Notes only: their frontmatter |
| `merged_from` | IDs merged into this node, each with how: `name`, `alias`, `embedding 0.95` or `llm check 0.86` |

## edges

| field | |
|---|---|
| `source`, `target`, `type` | Directed. Structural types: `LINKS_TO`, `EMBEDS`, `HAS_TAG`, `SUBTAG_OF`, `MENTIONS` (note → entity found in it), and frontmatter keys (`owner` → `OWNER`). Everything else came from the LLM (`WORKS_AT`, `USES`…) |
| `origins` | Where it came from: `wikilink`, `markdown_link`, `embed`, `frontmatter`, `tag`, `llm` |
| `description`, `confidence` | Set for LLM edges; `confidence` is 0–1 |
| `notes`, `chunks` | Evidence: which notes and chunks support it |

## chunks

| field | |
|---|---|
| `id` | `<note path>#c<index>` |
| `note`, `path`, `title` | The note it belongs to |
| `heading_path` | Headings above it, e.g. `["Project X", "Methods"]` |
| `start_line`, `end_line` | Line numbers in the note file (1-based) |
| `text` | Raw Markdown of the section |
| `entities` | IDs of the nodes found in it |
| `passages` | IDs of its passages |

## passages

Small chunks for vector search, cut inside each section. Search on the passages, then hand the LLM the parent section (small-to-big retrieval).

| field | |
|---|---|
| `id` | `<chunk id>.p<index>` |
| `chunk` | Parent section (chunk) ID |
| `note`, `path` | The note it belongs to |
| `heading_path` | Headings above it |
| `start_line`, `end_line` | Line range in the note file. With `--docling`, this is the parent section's range |
| `text` | Raw Markdown |
| `embed_text` | What to embed (and LightRAG's chunk text): a `Title > Heading` header, then the text with links as readable text that keeps their information: `[[Tom Becker\|Tom]]` → `Tom (Tom Becker)`, `[docs](https://…)` → `docs (https://…)`, `![[plan.png]]` → `plan.png`. Only an image's display size (`\|300`) is dropped |
| `tokens` | Token count of `embed_text` |
| `entities` | Nodes whose name or alias appears in the passage. String match only, among the entities found in its section |

**Sizing.** A passage holds at most `--passage-tokens` tokens (default 384, which fits embedding models capped at 512). Token counts are exact when `tiktoken` is installed; otherwise they're estimated at 4 characters per token.

**How they're cut.**
- Paragraphs, lists, tables and code blocks stay whole whenever they fit.
- Anything too long is split by lines, then sentences, then spaces. A word is only cut when it alone is longer than a passage.
- A passage never spans two headings, unless the leftover is tiny (under about 24 tokens).

**Nothing is lost.** The sections of a note together hold its whole body, and the passages of a section hold the whole section: every character except whitespace, in order. Each run checks this; a gap would show up as a `coverage` error in the note's JSON and a warning at the end of the run. The body is the note without its frontmatter, which becomes `properties`. Frontmatter that isn't valid YAML stays in the body as text. What vault2kg hides from the LLM (template lines, see the README) is only hidden from the LLM: sections and passages keep it.

**With `--docling`.** Docling's HybridChunker cuts each section instead, using the tokenizer of your embedding model (`--docling-tokenizer`). Docling re-renders the Markdown, so those passages are not checked character by character; the sections still are.

## How notes with several chunks are handled

- **Splitting:** a note is split by headings. Short sections are merged up to about 400 characters; long ones are split on paragraphs at about 2,500 characters, and a paragraph longer than that at a line break, a sentence end or a space, never inside a word. A heading always stays with the text below it.
- **Extraction:** each chunk is extracted on its own, in parallel and cached. It gets the note-level context: title, aliases, tags, properties, the note's opening lines, its heading path, and every note linked anywhere in the note. That way "Tom" in section 3 is extracted as "Tom Becker", who was linked in section 1.
- **Clean-up within a note:** short forms are folded into full names of the same type, but only when exactly one candidate fits ("Tom" → "Tom Becker", "ML" → "Machine Learning"). The relationships move along with them.
- **Cross-section relationships:** a relationship whose two ends only appear in different sections is not guessed. Every entity is still connected to its note through `MENTIONS`.
