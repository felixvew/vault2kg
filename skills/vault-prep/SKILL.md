---
name: vault-prep
description: Optional preparation of an Obsidian vault for vault2kg. Rewrites its frontmatter so chosen facts become certain relationships without an LLM; turns property values that name things (a team, a system, a risk level) into [[links]], splits combined values, adds note IDs as aliases and creates hub notes with definitions the user gives. Use when the user asks to rewrite a vault's frontmatter into links for vault2kg, or needs a fact complete in the graph that the vault guide's structure doesn't deliver. For general advice on how to structure notes, use VAULT_GUIDE.md instead.
---

# Vault prep for vault2kg (optional)

**Start with the vault guide** (`VAULT_GUIDE.md` in the vault2kg repository). For most vaults the better fix is in the notes: a `**Label:** value` line in the text for each fact that should become a relationship. That's how a real 88-note vault got its services linked in 82 of 88 notes, with no rewrite. If the notes come from a converter, change the converter, not the vault.

This skill is the other way. vault2kg turns a property into a relationship only when its value is a link (`team: "[[Imaging]]"` becomes a `TEAM` edge for every note that has it). This skill rewrites the frontmatter so chosen facts are links: certain, without an LLM, codes like `R3` included. The vault stays a normal Obsidian vault, and the links also appear in Obsidian's graph view and backlinks.

**The trade-off.** Linked frontmatter values reach the extraction prompt as known names. A small model (4B) then mostly repeats those names and misses new ones in the text: on the demo vault, MediaPipe and ViTPose were missed after the rewrite and found without it. So recommend this skill when the user runs a strong model, or needs a particular fact (such as a risk code) complete. Tell the user about the trade-off before planning.

## Rules

- **Only frontmatter changes.** Never edit, move or delete note text. Body text is the user's; vault2kg guarantees it reaches the graph unchanged.
- **Nothing is written without a backup:** a clean git repository, or `--backup DIR`. `apply_plan.py` refuses otherwise. Never work around that.
- **Plan first, then ask.** Show the user the plan and wait for a clear yes before `--write`.
- **Never invent definitions.** Hub note text comes from the user or from the vault's own source documents. Without a definition, no hub note: an unresolved link works fine in Obsidian and in vault2kg.
- **Use the scripts** for every change, not hand edits: they are repeatable and safe to run twice.

## Steps

1. **Profile.** Run `python scripts/profile_vault.py <vault>`. It lists every property with its number of notes, distinct values, values shared by several notes, a suggestion and examples. Read a few notes too, to see what the values mean.

2. **Draft the plan** (a JSON file, all parts optional; see `example-plan.json`):

   | Part | For | Typical properties |
   |---|---|---|
   | `link` | values that name a thing several notes share | team, service, department, system, tool, client, risk level, status |
   | `split` | one text holding several values: `"A AND B"`, `"A; B"` | data sources, systems, owners |
   | `rename` | spelling variants of one value, fixed before linking | `"Imgaging": "Imaging"` |
   | `alias` | an ID per note that the text uses to refer to it ("see 2.1.1") | item_id, ticket, code |
   | `hubs` | a note for a link target, with a definition the user gives | `"R3": "Risk level 3: …"` |

   Leave as properties: numbers and scores, dates, free text, values that differ in every note, and values (nearly) every note shares, such as a source file. Those last ones would link everything to everything. In Neo4j, properties stay filterable as `fm_*`.

   **Linked properties stop being plain properties.** In Neo4j, filter them through the edge instead:
   `MATCH (n:Note)-[:RISK]->(r {name: 'R3'}) RETURN n.name` rather than `WHERE n.fm_risk = 'R3'`.

3. **Show the plan** as a table: property, action, number of notes, example before and after. Ask about anything unclear, such as which values are spelling variants and what a code like `R3` means if a hub note is wanted.

4. **Dry run.** `python scripts/apply_plan.py <vault> plan.json` prints what would change and one example note, and writes nothing. Show the user the result.

5. **Write**, after a clear yes. In a git repository with nothing uncommitted: `--write`. Otherwise: `--write --backup <new folder>`.

6. **Check.** Run apply_plan again: it should report 0 notes to change. Then build the graph (`python vault2kg.py <vault>`, or `--no-llm` for a quick structural check) and confirm the new edges exist for every note that has the property. With an LLM, also compare the entities from the note text with a run on the backup: if names from the text went missing, the model is too small for the rewritten vault (see the trade-off above).

7. **Report** what changed, where the backup is, and how to undo it: `git checkout -- .` in git, or copy the backup back.

## Example

Before: `team: "Imaging"`, `systems: "Lab server AND GPU node"`, `item_id: "1.1.2"`. After:

```yaml
team: "[[Imaging]]"
systems:
  - "[[Lab server]]"
  - "[[GPU node]]"
aliases:
  - "1.1.2"
```

vault2kg then builds `TEAM`, `SYSTEMS` edges for every note, and "see 1.1.2" in any note resolves to this one.

## Model

This step decides what the graph will contain, and it is run once. Use the strongest model available for it.
