# Vault guide: notes that make a good graph

vault2kg works with any vault. This guide is for when you write notes, or generate them from a PDF, a CMS or a database export, and want the best graph out of them. It follows the structure of a real vault we tested on: 88 notes generated from a PDF that describes project ideas. That vault gave good results without any preparation, and [`demo_vault/`](demo_vault) is built the same way.

## The note

```markdown
---
item_id: "1.1.2"
team: "Imaging"
risk: "R2"
systems: "Lab server AND GPU node"
source_document: "Lab Methods Handbook.pdf"
---

# 1.1.2. Keypoint Extraction

**Team:** Imaging
**Risk:** R2

**Systems:** Lab server AND GPU node

## Method

MediaPipe extracts body keypoints from every video that 1.1.1 accepted; final analyses use ViTPose,
because MediaPipe loses the wrist during fast releases. Results go to the data warehouse.

## Linked with

- [[1.1.1 - Video Intake]]
- [[1.2.1 - Release Timing]]

---
Source: Lab Methods Handbook.pdf, section 1.1.2, page 4.
```

## The rules, and why

1. **One thing per note, and the title is its name.** Every note becomes a node. Numbered titles (`1.1.2 - Keypoint Extraction`) are fine: the number becomes an alias, so "see 1.1.2" anywhere resolves to this note.

2. **Every fact about the note goes into the frontmatter.** Properties stay exact. Neo4j can filter them (`WHERE n.fm_risk = 'R2'`), and LightRAG gets them in the note's description, so the note can be found by them.

3. **Repeat the facts that should become relationships in the text, as `**Label:** value`.** The LLM reads the text, not the frontmatter. These lines are how it connects each note to its team, service or system:

   | | team | systems |
   |---|---|---|
   | Demo vault, fact only in frontmatter | — | 5 of 12 links |
   | Demo vault, fact also as a `**Label:**` line | 8 of 8 | 12 of 12 |
   | Real vault, `**Service:**` line | 82 of 88 | — |

   vault2kg treats these lines as data, never as template text, even though every note has one.

4. **Codes need words.** Values like `R3` or `P2` rarely become entities: in the real vault, 1 of 87 notes was linked to its risk level. Keep codes as properties for exact filtering in Neo4j. If they should be nodes in the graph, see [Links in frontmatter](#links-in-frontmatter).

5. **Relationships between notes are `[[links]]`,** for example in a "Linked with" or "Related" section. Every link becomes an edge, with no LLM involved. A link only says *that* two notes are connected; the text around it says *how*.

6. **Headings for topics.** Sections are cut at headings, and each passage carries its `Title > Heading` header, so a heading helps both the LLM and vector search.

7. **Name things in full in the text.** Tools, people and organisations written out ("MediaPipe", "Tom Becker") become entities. Short forms ("Tom") are folded into the full name within a note.

8. **Boilerplate is fine.** A footer, a fixed sentence or source metadata repeated in every note is detected and hidden from the LLM automatically. It stays in the note, the passages and the properties.

## What to avoid

- **Facts only in the frontmatter** that you want as relationships (rule 3).
- **Several values in one field without a clear separator.** Use one separator throughout (`A AND B`) or a YAML list.
- **One note about many things.** Split it, or at least give each topic a heading.
- **Only a number or a code as the description** ("75%", "R3") with nothing in words.
- **Vague references** ("the tool above"). Name it or link it.

## For converters

If your notes come out of a script (from a PDF, a CMS, a ticket system), have it write:

- [ ] one note per item, with the item's number in the title if the source numbers them
- [ ] all fields in the frontmatter
- [ ] a `**Label:** value` line in the text for each field that should become a relationship
- [ ] the free text under a heading
- [ ] references to other items as `[[links]]`
- [ ] provenance (source file, page) in a footer: it's hidden from the LLM automatically

## Links in frontmatter

`team: "[[Imaging]]"` makes a relationship certain: a `TEAM` edge for every note, built without an LLM, also for codes like `R3`. Obsidian shows these links too. The trade-off, measured on the demo vault: when a note's facts are all links, a small model (4B) mostly repeats those names and misses new ones in the text. MediaPipe and ViTPose were missed that way. A strong model handles it better.

So the structure above is the default, and links in frontmatter are an option for strong models or facts you need complete. For an existing vault, the [vault-prep skill](skills/vault-prep/SKILL.md) can do that rewrite, with your approval.
