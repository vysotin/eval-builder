---
name: agent-eval-dataset
description: Use when creating or extending an eval dataset for a LangGraph agent after the agent map is reviewed — generates coverage-driven golden cases in OpenEvals/LangSmith-compatible JSON with metadata, imports existing goldens, and routes everything through the pending-review gate.
---

# Generate a coverage-driven golden dataset

You are the generator. The CLI owns validation, IDs, coverage math, and the review
state machine; you own case content and quality.

## Preconditions

- A reviewed `eval/agent-map.json` exists and the user has confirmed its intents and
  scenarios (or explicitly authorized assumptions). If not, route to
  agent-eval-discover first.
- Ask whether existing golden cases exist. Import them **before** generating:

  ```bash
  evalbuilder dataset import eval/datasets/NAME.json --from path/to/goldens.json
  ```

## Workflow

1. Create the dataset, recording the target factory:

   ```bash
   evalbuilder dataset init eval/datasets/NAME.json --name NAME --type final_response --target MODULE:build_agent
   ```

2. Check what the coverage grid requires and allocate cases across it:

   ```bash
   evalbuilder dataset gaps eval/datasets/NAME.json --agent-map eval/agent-map.json
   ```

3. Write each case with `evalbuilder dataset add PATH --case @case.json`. Every case
   carries (see `references/generation-guide.md` for variants, distribution, and
   golden-answer modes):
   - `inputs.messages` — the exact user input;
   - `reference_outputs` — a testable contract: a `response` description,
     `expected_tools` with args, optionally a reference `trajectory`, a `contains`
     substring, and a `contract` sentence derived from the agent map's constraints;
   - `metadata` — the full coverage cell (`intent`, `topic`, `scenario`,
     `failure_mode`), a `variant`, and `evidence` naming the agent-map entry it tests.

4. Quality-filter your own cases: reject ambiguous or unanswerable inputs — rewrite
   rather than discard. Then apply input evolutions (concretizing, constrained,
   comparative, multicontext) without changing the answerability contract. An
   unevolved synthetic dataset is trivially easy and will pass regardless of whether
   the agent is any good.

5. Re-run `evalbuilder dataset gaps` and fill what is missing, then
   `evalbuilder dataset validate` the result.

6. Present the pending cases to the user (id, input, expected result, coverage cell,
   source) and ask whether to review one-by-one or in a batch.

## The approval gate

Every generated or imported case lands as `pending`. Only this command records a
decision, and only after the user has explicitly approved or rejected **specific
cases in this conversation**:

```bash
evalbuilder review eval/datasets/NAME.json --approve "id1,id2" --note "approved by user <date>"
```

- **Never infer approval from silence, general enthusiasm, or a successful run.**
- Never approve cases you generated yourself without that explicit decision.
- Do not delete rejected cases; they are decision history.
- Old approval cannot authorize new content: imported and regenerated cases return
  to pending, by design.
