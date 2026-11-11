"""Offline tests for vault2kg (no LLM server needed)."""
import json
import re
import shutil
from pathlib import Path

import pytest

import vault2kg as v

SAMPLE = Path(__file__).parent / "sample_vault"


class FakeClient:
    """Returns canned extractions keyed on text found in the prompt."""

    model = "fake-model"

    def __init__(self):
        self.calls = 0
        self.schemas = []

    def complete(self, messages, schema=None):
        self.calls += 1
        self.schemas.append(schema)
        user = messages[-1]["content"].split("TEXT:", 1)[-1]  # match on the chunk, not the context
        if "motion-capture" in user:
            return json.dumps({
                "entities": [
                    {"name": "Anna", "type": "Person", "description": "Leads motion capture."},
                    {"name": "Northwind Labs", "type": "Organization", "description": "Research lab."},
                    {"name": "Tom Becker", "type": "Person", "description": "Co-author."},
                ],
                "relationships": [
                    {"source": "Anna", "target": "Northwind Labs", "type": "works at", "confidence": 0.9,
                     "description": "Anna leads motion capture there."},
                    {"source": "Anna Weber", "target": "Release Timing Study", "type": "CO_AUTHORED", "confidence": 0.8},
                    {"source": "Anna", "target": "Somebody Unmentioned", "type": "KNOWS"},
                ],
            })
        if "ViTPose" in user:
            # reasoning-model style output wrapped in think tags and a code fence
            return "<think>hmm</think>```json\n" + json.dumps({
                "entities": [{"name": "ViTPose", "type": "Tool", "description": "Pose model."},
                             {"name": "[[MediaPipe]]", "type": "Tool", "description": "Prototyping."}],
                "relationships": [{"source": "Anna Weber", "target": "ViTPose", "type": "USES", "confidence": 0.7}],
            }) + "\n```"
        return '{"entities": [], "relationships": []}'


@pytest.fixture()
def vault(tmp_path):
    dst = tmp_path / "vault"
    shutil.copytree(SAMPLE, dst)
    return dst


def run(vault, out, client=None, **cfg):
    cfg.setdefault("resolve", False)
    ex = v.VaultExtractor(v.Config(vault=vault, out=out, **cfg), client, progress=lambda m: None)
    ex.run()
    return ex, ex.summary


def load(out, rel):
    base = out if rel == "graph.json" else out / "notes"
    return json.loads((base / rel).read_text(encoding="utf-8"))


def ents(doc):
    return {e["id"]: e for e in doc["entities"]}


def rels(doc):
    return {(r["source"], r["type"], r["target"]): r for r in doc["relationships"]}


# ---------- parsing ----------
def test_unquoted_yaml_wikilinks_and_property_split(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Projects/Pose Pipeline.json")
    r = rels(doc)
    assert ("note:Projects/Pose Pipeline", "RELATED", "note:Tools/MediaPipe") in r  # unquoted [[MediaPipe]]
    assert ("note:Projects/Pose Pipeline", "OWNER", "note:People/Anna Weber") in r
    props = ents(doc)["note:Projects/Pose Pipeline"]["properties"]
    assert props == {"type": "Project", "status": "active", "start": "2026-03-01"}


def test_code_comments_and_same_note_links_ignored(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Projects/Pose Pipeline.json")
    ids = set(ents(doc))
    assert "note:NotALink" not in ids
    assert not any(i.startswith("tag:") for i in ids)  # #notatag in code, `#inline-code` in inline code
    assert ents(doc)["note:Missing Note"]["resolved"] is False
    assert ents(doc)["file:attachments/pipeline.png"]["kind"] == "file"
    assert ents(doc)["note:Specs/Pipeline Spec"]["resolved"] is True  # ../ relative markdown link
    assert doc["external_links"][0]["url"].startswith("https://")


def test_tags_aliases_and_nested_tags(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "People/Anna Weber.json")
    e, r = ents(doc), rels(doc)
    assert {"tag:people", "tag:research/biomech", "tag:todo/followup", "tag:research", "tag:todo"} <= set(e)
    assert ("tag:research/biomech", "SUBTAG_OF", "tag:research") in r
    assert set(e["note:People/Anna Weber"]["aliases"]) == {"Anna", "A. Weber"}
    assert e["note:Tom Becker"]["aliases"] == ["Tom"]  # display text of [[Tom Becker|Tom]]


def test_ambiguous_stems_prefer_same_folder(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Notes/Index.json")
    targets = {r["target"] for r in doc["relationships"]}
    assert "note:Archive/Index" in targets           # explicit path link
    assert "note:Notes/Index" not in targets          # [[Index]] from Notes/ resolves to itself -> no self loop
    assert ents(doc)["note:Notes/Index"]["name"] == "Notes/Index"  # path-qualified display name


def test_has_file_ext():
    assert v.has_file_ext("report.pdf") and v.has_file_ext("attachments/clip.MP4") and v.has_file_ext("a.7z")
    assert not v.has_file_ext("Note.md") and not v.has_file_ext("Plain Note")
    assert not v.has_file_ext("2.1.2 - Offer, Confirmation Creator")  # numbered note titles
    assert not v.has_file_ext("Release 2.1")


def test_dotted_note_names_resolve_to_notes(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Plans").mkdir(parents=True)
    (vault / "Plans" / "1.2.3 - Budget, Q3 Review.md").write_text(
        "# Budget\n\nSee [[1.4.2 - Hiring Plan]], [[Node.js]] and [[report.pdf]].\n"
        "Missing: [[missing.pdf]]\n\n![[1.4.2 - Hiring Plan]]\n", encoding="utf-8")
    (vault / "Plans" / "1.4.2 - Hiring Plan.md").write_text(
        "# Hiring\n\nLinked with [[1.2.3 - Budget, Q3 Review]].\n", encoding="utf-8")
    (vault / "Node.js.md").write_text("# Node.js\n", encoding="utf-8")
    (vault / "report.pdf").write_bytes(b"%PDF-1.4")
    out = tmp_path / "out"
    run(vault, out)
    g = load(out, "graph.json")
    links = {(e["source"], e["target"]) for e in g["edges"] if e["type"] in ("LINKS_TO", "EMBEDS")}
    budget, hiring = "note:Plans/1.2.3 - Budget, Q3 Review", "note:Plans/1.4.2 - Hiring Plan"
    assert {(budget, hiring), (hiring, budget), (budget, "note:Node.js"), (budget, "file:report.pdf")} <= links
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes[hiring]["resolved"] is True and nodes["note:Node.js"]["resolved"] is True
    assert nodes["file:report.pdf"]["resolved"] is True
    assert nodes["file:missing.pdf"]["resolved"] is False                 # unknown file stays a file
    assert {n["kind"] for n in g["nodes"] if n["name"].startswith("1.")} == {"note"}  # no phantom file nodes


# ---------- hidden properties and note IDs ----------
class PromptRecorder(FakeClient):
    """Records every prompt; answers with reply(chunk text)."""

    def __init__(self, reply=None):
        super().__init__()
        self.prompts = []
        self.reply = reply or (lambda text: {"entities": [], "relationships": []})

    def complete(self, messages, schema=None):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        return json.dumps(self.reply(messages[-1]["content"].split("TEXT:", 1)[1]))


def _numbered_vault(tmp_path):
    """Two notes in the layout of a PDF-generated vault: ID in the title and frontmatter,
    metadata repeated in the body, a provenance footer."""
    vault = tmp_path / "nv"
    (vault / "Methods").mkdir(parents=True)
    fm = ('---\nitem_id: "{id}"\nteam: "Imaging"\nclassification: "P3, R2, 75%, 0"\n'
          'source_document: "Handbook.pdf"\nsource_flag: "Odd Flag"\n---\n')
    body = ("# {id}. {title}\n\n**Team:** Imaging  \n**Classification:** P3, R2, 75%, 0\n\n"
            "**Source flag:** Odd Flag\n\n## Method\n\n{text}\n\n---\nSource: Handbook.pdf, section {id}, page 2.\n")
    for id_, title, text in [
            ("4.1.1", "Data Loader", "Loads raw recordings into the analysis store for the imaging team."),
            ("4.1.3", "Plot Export", "Reuses the loader from chapter 4.1.1 with other file formats.")]:
        (vault / "Methods" / f"{id_} - {title}.md").write_text(
            fm.format(id=id_) + body.format(id=id_, title=title, text=text), encoding="utf-8")
    return vault


def test_strip_for_llm():
    text = ("**Source flag:** X\n**Risk**: R3\n- risk:: R3\nSource: a.pdf, page 2\n"
            "**Team:** Imaging\nSolution: centralise it.\nSee https://x.y")
    assert v.strip_for_llm(text, ["source*", "risk"]).split("\n") == [
        "**Team:** Imaging", "Solution: centralise it.", "See https://x.y"]
    assert v.strip_for_llm(text) == text
    templ = {v.norm_line("Source: a.pdf, section 2.1, page 2.")}
    assert v.strip_for_llm("Source: a.pdf, section 4.7, pages 20, 21.\nkeep", templates=templ) == "keep"
    assert v.restates_property("**Team:** Imaging", {"team": "imaging"})
    assert not v.restates_property("Team: Imaging and Optics", {"team": "imaging"})


def test_title_id():
    assert v.title_id("4.1.1 - Data Loader") == "4.1.1" and v.title_id("4.12.3. Report Builder") == "4.12.3"
    assert v.title_id("11.01 Topic") == "11.01"
    assert v.title_id("Tom Becker") is None and v.title_id("1. Intro") is None and v.title_id("2024.01.05") is None


def _templated_vault(tmp_path):
    """Six notes from one template: a constant property, a data line, a template sentence,
    a footer (one variant with a page range)."""
    vault = tmp_path / "tv"
    (vault / "Items").mkdir(parents=True)
    for i, tool in enumerate(["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"], 1):
        pages = f"pages {i}, {i + 1}" if i == 6 else f"page {i + 3}"
        (vault / "Items" / f"Item {tool}.md").write_text(
            f'---\nsource_document: "Report.pdf"\nowner: "Team {tool}"\nscore: {i}\n---\n'
            f"# Item {tool}\n\n**Score:** {i}\n\n"
            f"The {tool} pipeline is maintained by Northwind and runs on {tool}DB.\n\n"
            "No further notes are listed for this item.\n\n"
            f"---\nSource: Report.pdf, section {i}, {pages}.\n", encoding="utf-8")
    return vault


def test_template_lines_and_constant_properties_are_not_shown(tmp_path):
    client, out = PromptRecorder(), tmp_path / "out"
    run(_templated_vault(tmp_path), out, client)
    assert len(client.prompts) == 6
    for p in client.prompts:
        assert "Report.pdf" not in p                 # constant property and footer (both variants)
        assert "No further notes" not in p           # template sentence
        assert "maintained by Northwind" in p and '"owner": "Team' in p   # content and varying property stay
        assert "**Score:**" in p                     # same shape in every note, but it is data
    g = load(out, "graph.json")
    assert all(n["properties"]["source_document"] == "Report.pdf" for n in g["nodes"] if n["kind"] == "note")
    assert any("Source: Report.pdf" in p["text"] for p in g["passages"])  # passages keep the full text


def test_templates_off_and_small_vaults_untouched(tmp_path):
    client = PromptRecorder()
    run(_templated_vault(tmp_path), tmp_path / "out", client, template_share=0)
    assert all("No further notes" in p for p in client.prompts)
    assert v.template_lines([], 0.5) == {} and v.constant_properties([]) == {}


def test_hidden_keys_never_reach_the_llm(tmp_path):
    vault, client, out = _numbered_vault(tmp_path), PromptRecorder(), tmp_path / "out"
    run(vault, out, client, llm_hidden_keys=["source*", "classification"])
    assert len(client.prompts) == 2
    for p in client.prompts:
        assert "Handbook.pdf" not in p and "Odd Flag" not in p and "P3, R2" not in p
        assert '"team": "Imaging"' in p and "**Team:** Imaging" in p   # not hidden: property and body line stay
    note = next(n for n in load(out, "graph.json")["nodes"] if n["id"] == "note:Methods/4.1.1 - Data Loader")
    assert note["properties"]["source_document"] == "Handbook.pdf"   # still a property of the note


@pytest.mark.parametrize("id_keys", [[], ["item_id"]])  # from the title alone, or also a property
def test_note_ids_resolve_to_notes(tmp_path, id_keys):
    def reply(text):
        if "chapter 4.1.1" not in text:
            return {"entities": [], "relationships": []}
        return {"entities": [{"name": "4.1.1", "type": "Project", "description": "Referenced method."},
                             {"name": "4.1.1 - Loader tool", "type": "Project", "description": "Same."}],
                "relationships": [{"source": "4.1.3 - Plot Export", "target": "4.1.1", "type": "BASED_ON",
                                   "description": "", "confidence": 0.9}]}
    out = tmp_path / "out"
    run(_numbered_vault(tmp_path), out, PromptRecorder(reply), id_keys=id_keys)
    g = load(out, "graph.json")
    a, b = "note:Methods/4.1.1 - Data Loader", "note:Methods/4.1.3 - Plot Export"
    assert not [n["name"] for n in g["nodes"] if n["kind"] == "entity"]   # both names became the note
    assert (b, "BASED_ON", a) in {(e["source"], e["type"], e["target"]) for e in g["edges"]}
    assert "4.1.1" in next(n for n in g["nodes"] if n["id"] == a)["aliases"]


def test_short_form_not_folded_into_numbered_note(tmp_path):
    vault = tmp_path / "wv"
    vault.mkdir()
    (vault / "5.2.1 - MediaPipe.md").write_text("# MediaPipe\n\nPose keypoints from video.\n", encoding="utf-8")
    (vault / "5.2.4 - Wrist Tracking.md").write_text(
        "# Wrist Tracking\n\nMediaPipe keypoints are checked for every frame where the wrist is lost.\n\n"
        "Linked with [[5.2.1 - MediaPipe]]\n", encoding="utf-8")

    def reply(text):
        found = [{"name": "MediaPipe", "type": "Tool", "description": "Pose model."}] if "are checked" in text else []
        return {"entities": found, "relationships": []}
    out = tmp_path / "out"
    run(vault, out, PromptRecorder(reply))
    nodes = {n["id"]: n for n in load(out, "graph.json")["nodes"]}
    assert nodes["ent:mediapipe"]["type"] == "Tool"                      # the tool stays a tool
    assert "MediaPipe" not in nodes["note:5.2.1 - MediaPipe"]["aliases"]  # and is not the numbered note


def test_note_by_id_prefix():
    r = v.Resolver({"S/2.7.3 - Feedback": ["2.7.3"], "S/2.7.4 - X": ["2.7.4"], "S/Dup": ["9.9"], "S/Dup2": ["9.9"]},
                   [], note_ids={"S/2.7.3 - Feedback": ["2.7.3"], "S/2.7.4 - X": ["2.7.4"],
                                 "S/Dup": ["9.9"], "S/Dup2": ["9.9"]})
    assert r.note_by_id("2.7.3") == r.note_by_id("2.7.3 - Survey feedback") == r.note_by_id("2.7.3. Survey") == "S/2.7.3 - Feedback"
    assert r.note_by_id("2.7.30") is None and r.note_by_id("Report 2.7.3") is None
    assert r.note_by_id("9.9 - shared ID") is None                       # two notes claim it: ignored


def test_invalid_yaml_reported_body_still_processed(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Broken.json")
    assert doc["extraction"]["errors"][0]["stage"] == "frontmatter"
    assert "note:People/Anna Weber" in ents(doc)


def test_line_numbers_point_into_file(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Projects/Pose Pipeline.json")
    lines = (vault / "Projects/Pose Pipeline.md").read_text().split("\n")
    r = rels(doc)[("note:Projects/Pose Pipeline", "LINKS_TO", "note:Missing Note")]
    assert "[[Missing Note]]" in lines[r["evidence"][0]["line"] - 1]
    fm = rels(doc)[("note:Projects/Pose Pipeline", "OWNER", "note:People/Anna Weber")]
    assert lines[fm["evidence"][0]["line"] - 1].startswith("owner:")


# ---------- chunking ----------
def test_chunks_never_end_on_bare_heading(vault, tmp_path):
    ex, _ = run(vault, tmp_path / "out", max_chars=200, min_chars=50)
    for note in ex.notes.values():
        for c in note.chunks:
            last = c.text.strip().split("\n")[-1]
            assert not v.HEADING_RE.match(last), (note.path, c.text)
            assert len(c.text) <= 200


def test_long_section_is_split():
    text = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    note = v.ParsedNote(path="x.md", key="x", text=text, meta={}, body=text, body_offset=0, fm_error=None,
                        key_lines={}, masked=v.mask_body(text)[0], code_lines=v.mask_body(text)[1])
    note.line_starts = [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    chunks = v.chunk_note(note, max_chars=1000, min_chars=200)
    assert len(chunks) > 3 and all(len(c.text) <= 1000 for c in chunks)
    assert all(c.heading_path == ["Big"] for c in chunks)


# ---------- LLM layer ----------
def test_llm_entities_merge_with_notes(vault, tmp_path):
    out = tmp_path / "out"
    client = FakeClient()
    run(vault, out, client, min_chars=50)  # one chunk per section
    doc = load(out, "People/Anna Weber.json")
    e, r = ents(doc), rels(doc)
    anna = e["note:People/Anna Weber"]
    assert "llm" in anna["origins"] and anna["type"] == "Person"  # alias "Anna" resolved to the note
    assert ("note:People/Anna Weber", "WORKS_AT", "note:Northwind Labs") in r
    assert ("note:People/Anna Weber", "CO_AUTHORED", "note:Release Timing Study") in r
    assert ("note:People/Anna Weber", "USES", "ent:vitpose") in r
    assert e["note:Tools/MediaPipe"]["descriptions"] == ["Prototyping."]  # [[MediaPipe]] name cleaned
    assert doc["extraction"]["stats"]["dropped_relationships"] == 1      # unknown endpoint
    wa = r[("note:People/Anna Weber", "WORKS_AT", "note:Northwind Labs")]
    assert wa["confidence"] == 0.9 and wa["evidence"] == [{"chunk_id": "People/Anna Weber#c0"}]


def test_prompt_renders_links_and_known_entities(vault, tmp_path):
    ex = v.VaultExtractor(v.Config(vault=vault, out=tmp_path / "o"), FakeClient(), progress=lambda m: None)
    ex.load()
    note = ex.notes["People/Anna Weber"]
    rendered, known = ex.render_chunk(note, note.chunks[0])
    assert "[[" not in rendered and "Tom (Tom Becker)" in rendered
    assert "Anna Weber (this note)" in known and "Anna (alias of this note)" in known


def test_cache_and_incremental(vault, tmp_path):
    out = tmp_path / "out"
    c1 = FakeClient()
    _, s1 = run(vault, out, c1, min_chars=100)
    assert c1.calls == s1["extracted"] > 0
    before = (out / "notes/Tools/MediaPipe.json").stat().st_mtime_ns

    c2 = FakeClient()
    run(vault, out, c2, min_chars=100)
    assert c2.calls == 0  # everything cached
    assert (out / "notes/Tools/MediaPipe.json").stat().st_mtime_ns == before  # unchanged file not rewritten

    p = vault / "Tools/MediaPipe.md"
    p.write_text(p.read_text() + "\n## Notes\n\n" + "Runs on mobile. " * 40 + "\n")
    c3 = FakeClient()
    _, s3 = run(vault, out, c3, min_chars=100)
    assert 0 < c3.calls < c1.calls  # only changed/new chunks go to the LLM

    (vault / "Archive/Index.md").unlink()
    run(vault, out)
    assert not (out / "notes/Archive/Index.json").exists() and not (out / "notes/Archive").exists()


def test_failed_chunk_is_reported_not_cached(vault, tmp_path):
    class Boom(FakeClient):
        def complete(self, messages, schema=None):
            raise v.LLMError("HTTP 500: boom")

    out = tmp_path / "out"
    run(vault, out, Boom())
    doc = load(out, "Tools/MediaPipe.json")
    assert doc["extraction"]["errors"][0]["stage"] == "llm"
    c = FakeClient()
    run(vault, out, c)
    assert c.calls > 0  # failures were not cached


def test_rel_type_normalisation():
    assert v.rel_type("works at") == "WORKS_AT"
    assert v.rel_type("worksAt") == "WORKS_AT"
    assert v.rel_type("gehört zu") == "GEHORT_ZU"
    assert v.rel_type("???") == "RELATED_TO"


def test_openai_compat_client_against_local_server():
    import http.server
    import threading

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"], seen["auth"], seen["body"] = self.path, self.headers.get("api-key"), body
            payload = {"choices": [{"message": {"content": '{"entities": [{"name": "X", "type": "Tool"}], "relationships": []}'}}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = v.OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "m", api_key="k", api_key_header="api-key")
        res = v.run_extraction(client, [{"role": "user", "content": "hi"}], ["Tool", "Person"])
    finally:
        srv.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "k"
    rf = seen["body"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["properties"]["entities"]["items"]["properties"]["type"]["enum"] == ["Tool", "Person", "Other"]
    assert res["entities"] == [{"name": "X", "type": "Tool", "description": ""}]


def test_structured_output_falls_back_to_json_mode():
    import http.server
    import threading

    formats = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            formats.append(body["response_format"]["type"])
            if body["response_format"]["type"] == "json_schema":
                data, code = b'{"error": "response_format json_schema not supported"}', 400
            else:
                data, code = json.dumps({"choices": [{"message": {"content": '{"entities": [], "relationships": []}'}}]}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = v.OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "m")
        v.run_extraction(client, [{"role": "user", "content": "a"}], [])
        v.run_extraction(client, [{"role": "user", "content": "b"}], [])
    finally:
        srv.shutdown()
    assert formats == ["json_schema", "json_object", "json_object"]  # falls back once, then stays


def test_context_overflow_keeps_structured_outputs_and_cap_is_optional():
    import http.server
    import threading

    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            if "LONG" in body["messages"][-1]["content"]:  # too long with or without a schema
                data, code = b'{"error": {"message": "maximum context length is 8192 tokens"}}', 400
            else:
                data, code = json.dumps({"choices": [{"message": {"content": '{"entities": [], "relationships": []}'}}]}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1"
        client = v.OpenAICompatClient(url, "m")
        with pytest.raises(v.LLMError, match="context length"):
            v.run_extraction(client, [{"role": "user", "content": "LONG"}], [])
        assert client.json_mode == "schema"                     # the schema was not the problem
        v.run_extraction(client, [{"role": "user", "content": "short"}], [])
        assert bodies[-1]["response_format"]["type"] == "json_schema"
        assert "max_tokens" not in bodies[-1]                   # no cap unless asked for
        v.run_extraction(v.OpenAICompatClient(url, "m", max_tokens=2048), [{"role": "user", "content": "short"}], [])
        assert bodies[-1]["max_tokens"] == 2048
    finally:
        srv.shutdown()


# ---------- selection ----------
def test_globs_obsidian_excludes_and_opt_out(vault, tmp_path):
    out = tmp_path / "out"
    ex, _ = run(vault, out, ignores=v.IGNORE + ["Specs"])
    assert "Templates/Daily" not in ex.notes           # Obsidian "Excluded files"
    assert "Specs/Pipeline Spec" not in ex.notes        # plain folder name excludes the folder
    assert "Private" in ex.notes and not (out / "notes/Private.json").exists()  # vault2kg: false
    assert not any(n["id"] == "note:Private" for n in load(out, "graph.json")["nodes"])
    g = v.compile_glob("Projects/*")
    assert g.match("Projects/Pose Pipeline.md") and not g.match("Projects/sub/x.md")
    assert v.compile_glob("**/drafts/**").match("a/b/drafts/c.md")


# ---------- improvements borrowed from Simple Graph Builder ----------
def test_mentions_edges_and_small_chunks_skipped(vault, tmp_path):
    out = tmp_path / "out"
    client = FakeClient()
    run(vault, out, client, min_chars=50)
    doc = load(out, "People/Anna Weber.json")
    assert ("note:People/Anna Weber", "MENTIONS", "ent:vitpose") in rels(doc)
    small = load(out, "Archive/Index.json")  # "Old index." is below min_llm_chars
    assert small["extraction"]["stats"]["skipped_small_chunks"] == 1
    assert all(s is not None for s in client.schemas)


def test_frontmatter_edit_does_not_reextract(vault, tmp_path):
    out = tmp_path / "out"
    run(vault, out, FakeClient())
    p = vault / "Projects/Pose Pipeline.md"
    p.write_text(p.read_text().replace("status: active", "status: paused\ntags: [new-tag]"))
    c = FakeClient()
    run(vault, out, c)
    assert c.calls == 0  # JSON updated, no LLM call
    assert load(out, "Projects/Pose Pipeline.json")["note"]["frontmatter"]["status"] == "paused"


# ---------- vault-level resolution ----------
class FakeEmbedClient(FakeClient):
    embed_model = "fake-embed"

    def __init__(self, vectors, same=True):
        super().__init__()
        self.vectors, self.same, self.verified = vectors, same, []

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]

    def complete(self, messages, schema=None):
        user = messages[-1]["content"]
        if user.startswith("Do these two entities"):
            self.verified.append(user)
            return json.dumps({"same": self.same})
        return super().complete(messages, schema)


def _resolution_vault(tmp_path):
    vault = tmp_path / "rv"
    vault.mkdir()
    (vault / "A.md").write_text("Notes on pose-estimation and the ML pipeline at Acme Corp. " * 3)
    (vault / "B.md").write_text("More about Pose Estimation and Machine Learning, built with Acme Corporation. " * 3)
    (vault / "Index.md").write_text("root index")
    (vault / "sub").mkdir()
    (vault / "sub/Index.md").write_text("sub index")
    return vault


class ResolutionLLM(FakeEmbedClient):
    def complete(self, messages, schema=None):
        user = messages[-1]["content"]
        if user.startswith("Do these two entities"):
            return super().complete(messages, schema)
        self.calls += 1
        if "Acme Corp." in user:
            ents = [("pose-estimation", "Method"), ("ML", "Concept"), ("Acme Corp", "Organization")]
        else:
            ents = [("Pose Estimation", "Method"), ("Machine Learning", "Concept"), ("Acme Corporation", "Organization")]
        return json.dumps({"entities": [{"name": n, "type": t, "description": f"{n} desc"} for n, t in ents],
                           "relationships": []})


def test_resolution_name_embedding_and_llm_verification(tmp_path):
    vault, out = _resolution_vault(tmp_path), tmp_path / "out"
    vectors = {"ML": [1.0, 0.0, 0.0], "Machine Learning": [0.99, 0.14, 0.0],      # cos ~0.99 -> auto
               "Acme Corp": [0.0, 1.0, 0.0], "Acme Corporation": [0.5, 0.866, 0.0]}  # cos ~0.87 -> verify
    client = ResolutionLLM(vectors, same=True)
    run(vault, out, client, resolve=True)
    graph = load(out, "graph.json")
    m = {mf["id"]: n["id"] for n in graph["nodes"] for mf in n.get("merged_from", [])}
    assert m.get("ent:pose estimation") == "ent:pose-estimation" or m.get("ent:pose-estimation") == "ent:pose estimation"
    assert {"ent:ml", "ent:machine learning"} & set(m)                          # embedding auto-merge
    assert {"ent:acme corp", "ent:acme corporation"} & set(m)                   # merged after LLM said yes
    assert len(client.verified) == 1
    assert "note:Index" not in m and "note:sub/Index" not in m                   # two notes never merge
    st = graph["stats"]
    assert st["merged_by_name"] == 1 and st["merged_by_embedding"] == 1 and st["merged_by_llm_check"] == 1
    # relationships of merged entities point at the canonical node
    ids = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])

    client2 = ResolutionLLM(vectors, same=False)
    shutil.rmtree(out / ".cache" / "verify")
    run(vault, out, client2, resolve=True)
    g2 = load(out, "graph.json")
    m2 = {mf["id"] for n in g2["nodes"] for mf in n.get("merged_from", [])}
    assert not ({"ent:acme corp", "ent:acme corporation"} & m2)


def test_without_resolve_entities_stay_separate(tmp_path):
    vault, out = _resolution_vault(tmp_path), tmp_path / "out"
    run(vault, out, ResolutionLLM({}))
    graph = load(out, "graph.json")
    assert graph["stats"]["merged"] == 0
    acme = next(e for e in graph["nodes"] if e["id"] == "ent:acme corp")
    assert acme["notes"] == ["A.md"] and acme["type"] == "Organization"


def test_default_output_next_to_vault(tmp_path):
    assert v.default_out(tmp_path / "My Vault") == tmp_path / "My Vault-kg"


def test_cli_end_to_end(vault, capsys):
    assert v.main([str(vault), "--no-llm"]) == 0
    out = vault.parent / f"{vault.name}-kg"
    assert (out / "graph.json").exists() and (out / "notes").is_dir()
    assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".")) == ["graph.json", "notes"]
    err = capsys.readouterr().err
    assert "graph.json" in err and "kg_to_neo4j.py" in err


def test_pick_ollama_models():
    names = ["nomic-embed-text:latest", "llama3.2:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:72b"]
    assert v.pick_ollama_models(names) == ("qwen2.5:14b", "nomic-embed-text:latest")
    assert v.pick_ollama_models(["some-model:latest"]) == ("some-model:latest", None)
    assert v.pick_ollama_models([]) == (None, None)


def test_detect_llm(monkeypatch):
    import http.server
    import threading

    class Tags(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "mxbai-embed-large"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    for k in ("VAULT2KG_MODEL", "VAULT2KG_BASE_URL", "VAULT2KG_API_KEY", "VAULT2KG_EMBED_MODEL", "OPENAI_API_KEY",
              "VAULT2KG_EMBED_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAULT2KG_VLLM_URL", "http://127.0.0.1:9/v1")  # no vLLM in this test
    srv = http.server.HTTPServer(("127.0.0.1", 0), Tags)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_port}")
        client, desc = v.detect_llm()
        assert client.model == "qwen2.5:7b" and client.embed_model == "mxbai-embed-large"
        assert client.base_url.endswith("/v1") and "Ollama" in desc
    finally:
        srv.shutdown()
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens there
    assert v.detect_llm()[0] is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client, _ = v.detect_llm()
    assert client.base_url == "https://api.openai.com/v1" and client.api_key == "sk-test"


# ---------- multi-chunk notes ----------
def _multichunk_vault(tmp_path):
    vault = tmp_path / "mv"
    vault.mkdir()
    (vault / "Project X.md").write_text(
        "Project X is run by [[Tom Becker]] at Acme with a focus on Machine Learning.\n\n"
        "## Team\n\n" + "Tom handles the data pipeline and the evaluation. " * 12 + "\n\n"
        "## Methods\n\n" + "We rely on ML for the pose model and keep everything reproducible. " * 10 + "\n")
    return vault


class ShortFormLLM(FakeClient):
    def complete(self, messages, schema=None):
        self.calls += 1
        self.schemas.append(messages[-1]["content"])
        user = messages[-1]["content"]
        text = user.split("TEXT:", 1)[1]
        if "Tom handles" in text:
            ents, rels = [("Tom", "Person"), ("data pipeline", "Tool")], [("Tom", "data pipeline", "MAINTAINS")]
        elif "We rely on ML" in text:
            ents, rels = [("ML", "Concept"), ("pose model", "Tool")], [("pose model", "ML", "USES")]
        else:
            ents, rels = [("Tom Becker", "Person"), ("Machine Learning", "Concept"), ("Acme", "Organization")], []
        return json.dumps({"entities": [{"name": n, "type": t, "description": f"{n}."} for n, t in ents],
                           "relationships": [{"source": a, "target": b, "type": t, "description": "",
                                              "confidence": 0.8} for a, b, t in rels]})


def test_every_chunk_gets_note_context(tmp_path):
    vault, out = _multichunk_vault(tmp_path), tmp_path / "out"
    client = ShortFormLLM()
    run(vault, out, client, min_chars=50)
    prompts = client.schemas
    assert len(prompts) == 3
    later = [p for p in prompts if "SECTION:" in p]
    assert later and all("NOTE OPENING" in p and "Project X is run by Tom Becker" in p for p in later)
    assert all("Tom Becker" in p.split("KNOWN ENTITIES:", 1)[1].split("\n", 1)[0] for p in prompts)
    first = [p for p in prompts if "SECTION:" not in p][0]
    assert "NOTE OPENING" not in first


def test_short_forms_folded_within_note(tmp_path):
    vault, out = _multichunk_vault(tmp_path), tmp_path / "out"
    run(vault, out, ShortFormLLM(), min_chars=50)
    doc = load(out, "Project X.json")
    e, r = ents(doc), rels(doc)
    assert "ent:tom" not in e and "ent:ml" not in e
    assert "Tom" in e["note:Tom Becker"]["aliases"] and "ML" in e["ent:machine learning"]["aliases"]
    assert ("note:Tom Becker", "MAINTAINS", "ent:data pipeline") in r          # relationship moved
    assert ("ent:pose model", "USES", "ent:machine learning") in r
    assert doc["extraction"]["stats"]["short_forms_merged"] == 2
    tom = e["note:Tom Becker"]
    assert len(tom["mentions"]) == 2                                            # both chunks
    graph = load(out, "graph.json")
    chunk_ents = {c["id"]: c["entities"] for c in graph["chunks"]}
    assert any("note:Tom Becker" in v_ for k, v_ in chunk_ents.items() if k.endswith("#c1"))


def test_ambiguous_short_form_is_not_folded():
    g = v.NoteGraph()
    for name in ("Tom", "Tom Becker", "Tom Smith"):
        g.entity(f"ent:{name.lower()}", name, "entity", "Person", origin="llm")
    assert g.consolidate_short_forms() == 0 and "ent:tom" in g.entities


# ---------- passages (retrieval chunks) ----------
def _long_vault(tmp_path):
    vault = tmp_path / "lv"
    vault.mkdir()
    para = lambda i: f"Paragraph {i} about [[Tom Becker|Tom]] and the release phase of the shot. " * 6  # noqa: E731
    (vault / "Long.md").write_text(
        "# Long\n\nIntro sentence about the project and its scope, long enough to stand alone as text. " * 3
        + "\n\n## Part A\n\n" + "\n\n".join(para(i) for i in range(12))
        + "\n\n## Part B\n\n" + "Part B text about anchor and draw length, also long enough. " * 8 + "\n")
    return vault


def test_passages_are_token_capped_and_inside_sections(tmp_path):
    vault, out = _long_vault(tmp_path), tmp_path / "out"
    run(vault, out, passage_tokens=200)
    g = load(out, "graph.json")
    ps = g["passages"]
    assert len(ps) > len(g["chunks"])
    lines = (vault / "Long.md").read_text().split("\n")
    for p in ps:
        assert count_body(p["text"]) <= 200
        assert p["chunk"] in {c["id"] for c in g["chunks"]}
        assert p["embed_text"].startswith("Long")
        assert "[[" not in p["embed_text"] and "## " not in p["embed_text"].split("\n\n", 1)[1][:4]
        assert lines[p["start_line"] - 1].strip() and p["text"].split("\n")[0].strip() in lines[p["start_line"] - 1]
    part_a = [p for p in ps if p["heading_path"][-1:] == ["Part A"]]
    part_b = [p for p in ps if p["heading_path"][-1:] == ["Part B"]]
    assert len(part_a) >= 3 and part_b
    assert not any("Part B text" in p["text"] for p in part_a)            # no passage spans two sections
    chunk_of = {c["id"]: c for c in g["chunks"]}
    assert all(p["id"] in chunk_of[p["chunk"]]["passages"] for p in ps)


def count_body(text):
    return v.count_tokens(text)


def test_passage_entities_by_name_and_alias(tmp_path):
    vault, out = _long_vault(tmp_path), tmp_path / "out"
    run(vault, out, passage_tokens=200)
    g = load(out, "graph.json")
    with_tom = [p for p in g["passages"] if "note:Tom Becker" in p["entities"]]
    assert with_tom and all("Tom" in p["text"] for p in with_tom)
    assert not any("note:Tom Becker" in p["entities"] for p in g["passages"] if "Tom" not in p["text"])


# ---------- every character of a note ends up in its sections and passages ----------
def _edge_vault(tmp_path):
    import random
    rnd = random.Random(7)
    words = ["alpha", "beta", "gamma", "Tom", "Acme", "pipeline", "über", "naïve", "2026", "R3", "x"]
    say = lambda n: " ".join(rnd.choice(words) for _ in range(n))  # noqa: E731
    notes = {
        "Long line.md": "# Long\n\n" + say(900) + ".\n",
        "Sentences.md": "Intro before any heading.\n\n" + " ".join(f"Sentence {i} is about {say(3)}." for i in range(300)),
        "No spaces.md": "## Blob\n\n" + "x" * 5000 + "\n\nafter the blob\n",
        "Code.md": "# Code\n\n```python\n# not a heading\n\nx = 1\n\n\n# still code\n```\n\ntext after\n" * 20,
        "Comments.md": "%% # hidden heading %%\nvisible %% inline comment %% text\n<!-- html\ncomment -->\n# Real\n\nbody\n",
        "Table.md": "# T\n\n| a | b |\n|---|---|\n" + "".join(f"| {i} | {say(2)} |\n" for i in range(200)),
        "Headings only.md": "# A\n## B\n### C\n",
        "Crlf.md": "﻿# Win\r\n\r\nline one\r\nline two\r\n",
        "Bad yaml.md": "---\ntitle: [unclosed\n---\nBody [[Long line]].\n",
        "List yaml.md": "---\n- a\n- b\n---\nBody of a note with list frontmatter.\n",
        "Unterminated.md": "---\nkey: value\nno closing line\n",
        "Empty.md": "",
        "Only frontmatter.md": "---\nkey: value\n---\n",
        "Links.md": "See [[Long line|the long one]], ![[pic.png|300]], [docs](https://x.y/z) and [[#Local]].\n",
    }
    for i in range(30):  # random notes: headings, blank lines, lists, long and short lines
        lines = []
        for _ in range(rnd.randint(5, 80)):
            r = rnd.random()
            lines.append("#" * rnd.randint(1, 4) + " " + say(2) if r < 0.1 else "" if r < 0.3 else
                         "- " + say(rnd.randint(1, 30)) if r < 0.45 else say(rnd.randint(1, 400)) + rnd.choice(".!; "))
        notes[f"Random {i}.md"] = "\n".join(lines)
    vault = tmp_path / "ev"
    vault.mkdir()
    for name, text in notes.items():
        (vault / name).write_bytes(text.encode("utf-8"))
    return vault


def _body_of(raw: bytes) -> str:
    """Independent of vault2kg: the file minus frontmatter that reads as properties."""
    import yaml
    text = raw.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    m = re.match(r"---\n(.*?)\n(?:---|\.\.\.)(?:\n|$)", text, re.S)
    if m:
        try:
            if isinstance(yaml.safe_load(m.group(1)) or {}, dict):
                return text[m.end():]
        except yaml.YAMLError:
            pass
    return text


def test_sections_and_passages_hold_every_character(tmp_path):
    vault, out = _edge_vault(tmp_path), tmp_path / "out"
    _, summary = run(vault, out, max_chars=300, min_chars=100, passage_tokens=40)
    assert summary["coverage_gaps"] == 0
    g = load(out, "graph.json")
    ns = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    idx = lambda c: int(c["id"].rsplit("#c", 1)[1])  # noqa: E731
    order = {c["id"]: (c["path"], idx(c)) for c in g["chunks"]}
    for f in sorted(vault.glob("*.md")):
        body = _body_of(f.read_bytes())
        cs = sorted((c for c in g["chunks"] if c["path"] == f.name), key=idx)
        ps = sorted((p for p in g["passages"] if p["path"] == f.name), key=lambda p: (order[p["chunk"]], p["index"]))
        assert ns("".join(c["text"] for c in cs)) == ns(body), f"sections of {f.name}"
        assert ns("".join(p["text"] for p in ps)) == ns(body), f"passages of {f.name}"
        if f.name != "No spaces.md":  # LightRAG's text: every word, links included (target, URL, file)
            missing = set(re.findall(r"\w+", body)) - set(re.findall(r"\w+", " ".join(p["embed_text"] for p in ps)))
            assert missing <= {"300"}, (f.name, sorted(missing)[:5])   # "|300" in ![[pic.png|300]] is a size


def test_long_lines_are_never_cut_inside_a_word(tmp_path):
    vault, out = _edge_vault(tmp_path), tmp_path / "out"
    run(vault, out, max_chars=300, min_chars=100, passage_tokens=40)
    g = load(out, "graph.json")
    for kind in ("chunks", "passages"):
        items = [x for x in g[kind] if x["path"] in ("Long line.md", "Sentences.md", "Table.md")]
        assert len(items) > 6
        for a, b in zip(items, items[1:]):
            if a["path"] == b["path"] and a["end_line"] == b["start_line"]:  # cut inside one line
                assert not (a["text"][-1:].isalnum() and b["text"][:1].isalnum()), (kind, a["text"][-15:], b["text"][:15])


def test_lost_text_is_reported_not_silent(tmp_path, monkeypatch):
    real = v.builtin_passages
    monkeypatch.setattr(v, "builtin_passages", lambda *a, **k: real(*a, **k)[1:])  # a bug that loses text
    out = tmp_path / "out"
    _, summary = run(_edge_vault(tmp_path), out)
    assert summary["coverage_gaps"] > 0
    assert any(e["stage"] == "coverage" for e in load(out, "Long line.json")["extraction"]["errors"])


def test_invalid_frontmatter_stays_text(tmp_path):
    vault = _edge_vault(tmp_path)
    out = tmp_path / "out"
    run(vault, out)
    doc = load(out, "Bad yaml.json")
    assert doc["extraction"]["errors"][0]["stage"] == "frontmatter"
    assert "title: [unclosed" in doc["chunks"][0]["text"] and doc["note"]["frontmatter"] == {}


def test_plain_links_keep_targets_urls_and_files():
    t = ("See [[Tom Becker|Tom]], [[Tom Becker]], [[Note#Part]], ![[pic.png|300]], ![[plan.png|Site plan]], "
         "[docs](https://x.y/z), [[#Local]].")
    assert v.plain_links(t) == ("See Tom (Tom Becker), Tom Becker, Note > Part, pic.png, Site plan (plan.png), "
                                "docs (https://x.y/z), Local.")
    assert len(v.plain_links(t)) <= len(t)                    # never longer: passages keep their token budget


def test_docling_mode_uses_hybrid_chunker(tmp_path, monkeypatch):
    import sys
    import types
    seen = {}

    class Tok:
        @classmethod
        def from_pretrained(cls, model_name, max_tokens):
            seen["tokenizer"] = (model_name, max_tokens)
            return cls()

        def count_tokens(self, text):
            return len(text.split())

    class Chunk:
        def __init__(self, text, headings):
            self.text, self.meta = text, types.SimpleNamespace(headings=headings)

    class Hybrid:
        def __init__(self, tokenizer, merge_peers):
            self.tokenizer = tokenizer

        def chunk(self, dl_doc):
            return [Chunk(part, ["Sub"]) for part in dl_doc.split("\n\n") if part.strip()]

    class Converter:
        def __init__(self, allowed_formats):
            pass

        def convert_string(self, text, format, name):
            seen.setdefault("calls", 0)
            seen["calls"] += 1
            return types.SimpleNamespace(document=text)

    mods = {n: types.ModuleType(n) for n in ["docling", "docling.datamodel", "docling.datamodel.base_models",
                                             "docling.document_converter", "docling_core", "docling_core.transforms",
                                             "docling_core.transforms.chunker",
                                             "docling_core.transforms.chunker.hybrid_chunker",
                                             "docling_core.transforms.chunker.tokenizer",
                                             "docling_core.transforms.chunker.tokenizer.huggingface"]}
    mods["docling.datamodel.base_models"].InputFormat = types.SimpleNamespace(MD="md")
    mods["docling.document_converter"].DocumentConverter = Converter
    mods["docling_core.transforms.chunker.hybrid_chunker"].HybridChunker = Hybrid
    mods["docling_core.transforms.chunker.tokenizer.huggingface"].HuggingFaceTokenizer = Tok
    for k, m in mods.items():
        monkeypatch.setitem(sys.modules, k, m)
    vault, out = _long_vault(tmp_path), tmp_path / "out"
    run(vault, out, docling=True, passage_tokens=256, docling_tokenizer="my/embedder")
    g = load(out, "graph.json")
    assert seen["tokenizer"] == ("my/embedder", 256)
    assert g["passages"] and all(p["heading_path"][-1] == "Sub" for p in g["passages"])
    first_calls = seen["calls"]
    run(vault, out, docling=True, passage_tokens=256, docling_tokenizer="my/embedder")
    assert seen["calls"] == first_calls                                    # cached per section


def test_docling_missing_gives_clear_error(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(SystemExit, match="pip install docling"):
        run(_long_vault(tmp_path), tmp_path / "out", docling=True)



# ---------- vLLM ----------
def _openai_server(models, owned_by="vllm", embed_dim=None):
    import http.server
    import threading
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            calls.append(("GET", self.path))
            self._send({"object": "list", "data": [{"id": m, "object": "model", "owned_by": owned_by} for m in models]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body))
            if self.path.endswith("/embeddings"):
                self._send({"data": [{"index": i, "embedding": [0.1] * (embed_dim or 4)} for i in range(len(body["input"]))]})
            else:
                self._send({"choices": [{"message": {"content": '{"entities": [], "relationships": []}'}}]})

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", calls


def test_detect_vllm_with_separate_embedding_container(monkeypatch):
    for k in ("VAULT2KG_MODEL", "VAULT2KG_BASE_URL", "VAULT2KG_EMBED_MODEL", "VAULT2KG_EMBED_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    chat_srv, chat_url, chat_calls = _openai_server(["Qwen/Qwen3-8B"])
    emb_srv, emb_url, emb_calls = _openai_server(["BAAI/bge-m3"])
    try:
        monkeypatch.setenv("VAULT2KG_VLLM_URL", chat_url)
        monkeypatch.setenv("VAULT2KG_VLLM_EMBED_URL", emb_url)
        client, desc = v.detect_llm()
        assert client.model == "Qwen/Qwen3-8B" and client.embed_model == "BAAI/bge-m3"
        assert client.base_url == chat_url and client.embed_base_url == emb_url
        assert client.suggested_workers == v.VLLM_WORKERS and desc.startswith("vLLM")
        client.complete([{"role": "user", "content": "x"}], v.extraction_schema([]))
        body = [c for c in chat_calls if c[0] == "POST"][-1][2]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}   # Qwen3: no thinking
        assert body["response_format"]["type"] == "json_schema"
        assert len(client.embed(["a", "b"])) == 2
        assert any(c[0] == "POST" and c[1].endswith("/embeddings") for c in emb_calls)
        assert not any(c[0] == "POST" and c[1].endswith("/embeddings") for c in chat_calls)
    finally:
        chat_srv.shutdown()
        emb_srv.shutdown()


def test_explicit_base_url_reads_model_from_server(monkeypatch):
    for k in ("VAULT2KG_MODEL", "VAULT2KG_EMBED_BASE_URL", "VAULT2KG_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAULT2KG_VLLM_EMBED_URL", "http://127.0.0.1:9/v1")
    srv, url, _ = _openai_server(["my-served-model"], owned_by="vllm")
    try:
        client, _ = v.detect_llm(base_url=url)
        assert client.model == "my-served-model" and client.embed_model is None
        client2, _ = v.detect_llm(base_url=url, model="other", embed_base_url=url)
        assert client2.model == "other" and client2.embed_model == "my-served-model"
    finally:
        srv.shutdown()
