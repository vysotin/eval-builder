# Evaluator selection

Translate each requirement into the **cheapest reliable evaluator**, in this order:

1. **Deterministic** — `expected_tools` (ordered subsequence, args subset),
   `contains`, `json_valid`. No LLM, no variance. Use whenever the contract is
   mechanical.
2. **Trajectory match** — `trajectory_match` (agentevals): `match_mode` strict /
   unordered / subset / superset; `args_match` exact / ignore / subset. Needs a
   reference `trajectory` on the case.
3. **LLM judge** — `correctness` (OpenEvals CORRECTNESS_PROMPT), `contract`
   (judges against the case's `reference_outputs.contract`), any OpenEvals rubric by
   name (`hallucination`, `conciseness`, `tool_selection`, `task_completion`, … or
   `type: openevals` + `prompt: NAME_PROMPT`), and `trajectory_llm` (AgentEvals
   trajectory judge over the captured trajectory). Judge models are `provider:model`
   specs: `claude-cli:sonnet` runs through the Claude Code CLI on the user's
   subscription (no key); `anthropic:…`/`openai:…` need keys. A missing key/binary
   is reported as an evaluator error, never as an agent failure; a case without the
   needed reference (`contains`, `contract`, `trajectory`) is `skipped`.

**One metric per evaluator.** Infrastructure (did it run), topology (which tools),
parsing (is it JSON), and semantics (is it right) stay separate metrics so a failure
localizes itself.

## evaluators.yaml

```yaml
evaluators:
  - type: expected_tools
  - type: contains
  - type: trajectory_match
    match_mode: unordered
    args_match: subset
  - type: contract
    model: "claude-cli:sonnet"
  - type: correctness
    model: "claude-cli:sonnet"
  - type: hallucination            # any openevals.prompts rubric by name
  - type: trajectory_llm           # agentevals trajectory judge
```

`model` defaults to `EVALBUILDER_JUDGE_MODEL`; `prompt` overrides the built-in judge
prompt when the rubric is domain-specific. `custom` type takes `ref: module:function`
with signature `fn(case, case_run) -> {key, score, comment}`.

## Thresholds

Scores are reported raw; pass/fail thresholds are a reporting decision. An
uncalibrated threshold is theatre — start binary (judges return 0/1), inspect the
distribution and slices, and only then negotiate thresholds with the user.
