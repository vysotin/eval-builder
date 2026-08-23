---
name: agent-eval-discover
description: Use when starting eval work on a LangGraph agent, before generating any test dataset, or after the agent's code changes — maps the agent's test surface (graph, tools, prompts, intents, scenarios, failure modes, data-domain topics) into eval/agent-map.json.
---

# Discover a LangGraph agent's test surface

Everything downstream of this map knows what the agent is *for*. Generation without
it is behaviorally blind: plausible text, accidental coverage, failure cases invented
rather than derived from how this agent can actually break.

## Workflow

1. Check readiness. If `ready` is true, say so in one line and move on — do not
   narrate a healthy setup. Fix `blocking` items first (each carries a `fix`);
   `degraded` entries (judge, langsmith) limit later steps but never block discovery.

   ```bash
   evalbuilder check --target-module MODULE
   ```

2. Extract the structural skeleton (AST parse + live introspection):

   ```bash
   evalbuilder discover MODULE --source path/to/agent.py
   ```

3. Read the agent source and its prompts yourself. Node titles and tool names alone
   are weak evidence — the prompts and conditional edges say what the agent decides.

4. Draft intents and scenarios. **Every entry must cite evidence** the code supports:
   `source:<file>:<line>`, `prompt:<node>`, `tool:<name>`, or `edge:<a>-><b>`.
   Everything you derive is a hypothesis for the user to confirm, never a fact.

5. Propose failure scenarios **only** from `references/failure-taxonomy.md`, and only
   the types whose structural precondition holds in this graph. An unjustifiable
   failure scenario is worse than none: it produces cases that can never fail for the
   reason claimed.

6. Derive data-domain topics only from documents the user provides or descriptions
   they give. If neither is available, ask the user to describe or locate the corpus.
   **Do not invent topics.**

7. Ask the user for constraints the code cannot express (e.g. "never quote a price").
   Constraints reach every generation prompt and usually become custom metrics later.

8. Record the map through the CLI (each list fully replaces its section, and every
   entry is shape-validated):

   ```bash
   evalbuilder agent-map update eval/agent-map.json --intents @intents.json --scenarios @scenarios.json --failures @failures.json --topics @topics.json --constraints @constraints.json
   ```

9. Present the map and its `decisions_needed`, then **stop. Do not generate cases
   until the user confirms the intent/scenario map or explicitly authorizes
   assumptions.** Hand off to agent-eval-dataset.

## Rules

- Never hand-edit `eval/agent-map.json` — the CLI validates every mutation.
- Heuristics are hypotheses. Label them as such when presenting.
- The agent map records `source_sha256`; if the agent code changed since the last
  discover run, re-run discovery before trusting the map.
