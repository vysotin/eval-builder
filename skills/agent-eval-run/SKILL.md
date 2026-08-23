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

2. Execute approved cases (mocks from the dataset are installed with `--mock`):

   ```bash
   evalbuilder run eval/datasets/NAME.json --mock --out eval/results
   ```

3. Choose evaluators per `references/evaluator-selection.md`, write
   `eval/evaluators.yaml`, then score the run artifact:

   ```bash
   evalbuilder score eval/results/run-ID.json --dataset eval/datasets/NAME.json --evaluators eval/evaluators.yaml
   ```

4. Report per-metric stats **and the slices** (by intent, failure_mode, variant) —
   never only the aggregate. A defect usually lives in one slice; the aggregate
   hides it.

5. Keep failure classes separate in your report: agent errors, infrastructure
   errors, and evaluator errors (`report.cases[].errors`) are different facts.
   **An evaluator problem must never be read as the agent behaving badly.**

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
