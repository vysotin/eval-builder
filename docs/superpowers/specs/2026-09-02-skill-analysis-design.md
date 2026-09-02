# Skill analysis in the agent map — design (2026-09-02)

Deepens how Agent Skills are analysed during discovery and the map stage, and how the
`agent-map.json` artifact records them. Follows the standing rule: determinism where it
is cheap (discovery), the generator LLM where judgment is needed (map stage), everything
LLM-written validated and dropped-with-a-problem when invalid.

## 1. Deterministic skill analysis (discovery, `skills.py` + `discover.py`)

Per-skill additions to the `skills[]` entry (`describe_skill`):

| field | meaning |
|---|---|
| `tools` | the tools the skill can use, **resolved against the agent's tools**: frontmatter `allowed-tools` that exist, then body-mentioned tools not already listed (order preserved) |
| `unknown_tools` | `allowed-tools` names that match no agent tool (a skill/agent drift signal) |
| `rules` | imperative lines extracted from the body (`never/must/always/only/don't`) — deterministic seeds for constraint and failure analysis |
| `chars` | body length |
| `instruction` | the **full body** when `chars <= SKILL_INSTRUCTION_CHARS` (1500), else a **deterministic summary** (description + section headings + their first line, capped) |
| `summarized` | whether `instruction` is a summary |

`prompt` keeps the full body — the artifact of record; `instruction` is what generator
briefs (`skill_brief`) and compact UI views consume.

Node ↔ skill mapping ("which node *might* use which skill"):

- existing links stay (rendered inline body / listing / prompt naming);
- **new**: an LLM node whose tools include a skill-loader but whose prompt links no
  skill is linked to **every** skill (`skills_source: "loader"`) — holding the loader
  means it can read any of them;
- every linked LLM node gets `capabilities`: one deterministic line per skill —
  `skill <name> (<disclosure>) → tools <resolved∩node tools> — <description>` — the
  "what it can call, which tools that reaches, what result it achieves" summary.

## 2. LLM analysis (map stage, `pipeline/generator.py` + `stages.py`)

`MAP_SCHEMA` gains two sections, both validated leniently (missing → empty, so scripted
generators without them keep working):

- `skill_failures: [{skill, failure_cases: [{description, failure_mode, expected_behavior, evidence}]}]`
  — failure cases **beyond tool failure** (skipping mandated steps, misordering,
  misapplying policy, acting without loading the skill, exceeding allowed-tools scope).
  Validation: unknown skill → entry dropped + problem; `failure_mode`, when given, must
  be in the applicable taxonomy; a case that is only "the tool fails" belongs in
  `tool_failures` (prompt rule, not code-enforced).
- `tool_failures: [{tool, scenarios: [{description, failure_mode, expected_behavior, evidence}]}]`
  — per-tool failure scenarios (error payloads, timeouts, empty results, misuse).
  Validation: unknown tool → dropped + problem; `failure_mode` from the applicable list
  (defaults to `tool_error_handling`).

The map stage merges the validated sections into the artifact:
`skills[].failure_cases` and `tools[].failure_scenarios` (alongside the deterministic
`edge_cases`). Stage details report the counts.

## 3. UI (`app_pages/agent.py`, `setup.py` preview)

- Skills table: `tools` column (resolved) + `summarized` marker; skill expander shows
  `instruction` (with a "summary of N chars" caption when summarized), `rules`, and
  the map-stage `failure_cases`.
- Nodes: `capabilities` lines rendered in the node expander.
- Tools: expander gains **Failure scenarios** (map stage) next to the schema edge cases.
- Setup preview: skills table gains the resolved `tools` column.

## 4. e2e

`examples/support_bot` is the vehicle (skills + external `*.example.invalid` tools that
must be mocked; `on_miss: llm` path already covered). Changes:

- `refund-policy/SKILL.md` body extended past the threshold → exercises the summary
  path end to end (its loader-returned text and `# Refund policy` heading stay, so the
  scripted agent model is unaffected);
- `offline.py` map response gains `skill_failures` + `tool_failures` (incident_desk
  likewise for the inline-disclosure path);
- pipeline e2e asserts: `skills[].tools/instruction/summarized/failure_cases`,
  `tools[].failure_scenarios`, node `capabilities` in the written `agent-map.json`;
- AppTest covers the Agent page rendering; the Playwright skills flow asserts the new
  sections in the browser.

## Non-goals

- No LLM calls during discovery (the CLI `discover` and the setup preview stay offline).
- No change to prompt rendering or the loader tool contract.
- `prompt` on the skill entry is not truncated; compaction happens in `instruction`
  and the briefs that use it.
