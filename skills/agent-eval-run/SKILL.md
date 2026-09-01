---
name: agent-eval-run
description: Use when running, scoring, or re-scoring an eval experiment for a LangGraph agent, publishing a dataset to LangSmith, or simulating multi-turn conversation scenarios — validates the dataset, executes approved cases with mocks, and reports metrics with coverage slices.
---

# Validate, execute, score, report

Execution and scoring are separate phases sharing one run artifact: re-score with
different evaluators without re-running the agent. A successful API call is not
semantic success — only evaluators decide that.

## Workflow

1. Validate first; a dataset with pending cases will not run:

   ```bash
   evalbuilder dataset validate eval/datasets/NAME.json
   ```

2. Execute approved cases (mocks from the dataset are installed with `--mock`; the
   miss policy defaults to the dataset's `mocks.on_miss`, and under `llm` the LLM mock
   engine needs a model — `mocks.llm.model` or `--mock-model`):

   ```bash
   evalbuilder run eval/datasets/NAME.json --mock --out eval/results
   evalbuilder run eval/datasets/NAME.json --mock --on-miss llm --mock-model claude-cli:claude-sonnet-5 --strategy degraded
   ```

   The run artifact records `mocking` (policy, model, calls answered per layer) and each
   case's `mock_calls` ledger (which layer answered, validation, repairs, fallbacks).

3. Choose evaluators per `references/evaluator-selection.md`, write
   `eval/evaluators.yaml`, then score the run artifact:

   ```bash
   evalbuilder score eval/results/run-ID.json --dataset eval/datasets/NAME.json --evaluators eval/evaluators.yaml
   ```

4. Report per-metric stats **and the slices** (by intent, failure_mode, variant) —
   never only the aggregate. A defect usually lives in one slice; the aggregate
   hides it.

5. Keep failure classes separate in your report: agent errors, infrastructure
   errors (including an LLM mock the engine could not make schema-conformant),
   evaluator errors (`report.cases[].errors`) and mock-layer facts (`mocking.calls`:
   `invalid`, `fallback`, `error`) are different things.
   **An evaluator or mock problem must never be read as the agent behaving badly.**

## Optional: LangSmith

When `evalbuilder check` shows langsmith capability, publish approved cases
(idempotent; verified by read-back) so experiments are tracked server-side:

```bash
evalbuilder publish eval/datasets/NAME.json --dataset-name NAME
```

## Optional: multi-turn simulation

For conversation scenarios (persona, goal, opening, followups, stop conditions),
write a scenarios YAML and run:

```bash
evalbuilder simulate eval/datasets/NAME.json --scenarios eval/scenarios.yaml
evalbuilder simulate eval/datasets/NAME.json --scenarios eval/scenarios.yaml --mock   # dataset mocks; a scenario's mock_strategy selects the engine's strategy
```

Stopping at `max_turns` is a truncation, not a pass. Only expectation-violating
runs are mined back into the dataset — as **pending** cases with
`metadata.source=simulation` for the user to review.

## Rules

- Never re-run the agent just to re-score; `evalbuilder score` works on the stored
  run artifact.
- Never run with pending or rejected cases; get an explicit review decision first
  (see agent-eval-dataset's approval gate).
- Report score distributions honestly, including error counts per metric.
