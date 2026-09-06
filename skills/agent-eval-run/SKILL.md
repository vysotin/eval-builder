---
name: agent-eval-run
description: Use when running (inference: in-process or against a deployed agent), scoring, or re-scoring an eval experiment for a LangGraph agent, publishing a dataset to LangSmith, or simulating multi-turn conversation scenarios — validates the dataset, executes approved cases with mocks in parallel, and reports metrics with coverage slices.
---

# Validate, infer, evaluate, report

Inference and evaluation are separate phases sharing one run artifact: re-score with
different evaluators without re-running the agent. A successful API call is not
semantic success — only evaluators decide that. Inference runs cases (and, with
`--scenarios`, the multi-turn simulations) in parallel with joblib — in this process by
default, or against an agent deployed behind an HTTP endpoint.

## Workflow

1. Validate first; a dataset with pending cases will not run:

   ```bash
   evalbuilder dataset validate eval/datasets/NAME.json
   ```

2. Infer: execute the approved cases (the dataset's mock layers are installed per
   request — `--mock` is the default, `--no-mock` calls real tools; the miss policy
   defaults to the dataset's `mocks.on_miss`, and under `llm` the LLM mock engine needs
   a model — `mocks.llm.model` or `--mock-model`):

   ```bash
   evalbuilder infer eval/datasets/NAME.json --out eval/results                       # in this process
   evalbuilder infer eval/datasets/NAME.json --repeats 2 --workers 4 --backend threads   # repeats → one run artifact each
   evalbuilder infer eval/datasets/NAME.json --on-miss llm --mock-model claude-cli:claude-sonnet-5 --strategy degraded
   evalbuilder infer eval/datasets/NAME.json --scenarios eval/scenarios.yaml           # + the multi-turn simulations
   ```

   To run against the agent in a container (the same image the pipeline builds), deploy
   it first and point `infer` at the deployment (or any endpoint serving
   `evalbuilder serve`); `--model` is ignored there — the deployment decides the agent
   model:

   ```bash
   evalbuilder deploy up eval/pipeline.yaml [--target docker|kubernetes]   # → <output dir>/deployment.json
   evalbuilder infer eval/datasets/NAME.json --deployment eval/pipeline/NAME --out eval/results
   evalbuilder infer eval/datasets/NAME.json --endpoint http://127.0.0.1:8080
   evalbuilder deploy down eval/pipeline/NAME
   ```

   Every run artifact records `execution` (`mode` local | remote, `endpoint`, `workers`,
   `backend`), `mocking` (policy, model, calls answered per layer) and, per case, the
   `mock_calls` ledger (which layer answered, validation, repairs, fallbacks) and a
   per-turn `log` (`turn`, `seconds`, `tool_calls`, `error`). `evalbuilder run` is the
   older name of `infer` and behaves identically.

3. Evaluate: choose evaluators per `references/evaluator-selection.md`, write
   `eval/evaluators.yaml`, then score the stored run artifacts (case runs are split
   across joblib workers; several runs are aggregated into pass rates, slices,
   stability and a verdict — `--config` reuses a pipeline config's judge model and
   thresholds):

   ```bash
   evalbuilder eval eval/results/run-ID.json --dataset eval/datasets/NAME.json --evaluators eval/evaluators.yaml
   evalbuilder eval eval/results/run-*.json --dataset eval/datasets/NAME.json --evaluators eval/evaluators.yaml --workers 4 --aggregate --config eval/pipeline.yaml
   evalbuilder score eval/results/run-ID.json --dataset eval/datasets/NAME.json --evaluators eval/evaluators.yaml   # one run, prints its report
   ```

4. Report per-metric stats **and the slices** (by intent, failure_mode, variant) —
   never only the aggregate. A defect usually lives in one slice; the aggregate
   hides it.

5. Keep failure classes separate in your report: agent errors, infrastructure
   errors (including an LLM mock the engine could not make schema-conformant, and an
   endpoint that did not answer — `execution.endpoint` and the per-turn `log` say
   which), evaluator errors (`report.cases[].errors`) and mock-layer facts
   (`mocking.calls`: `invalid`, `fallback`, `error`) are different things.
   **An evaluator or mock problem must never be read as the agent behaving badly.**

## Optional: LangSmith

When `evalbuilder check` shows langsmith capability, publish approved cases
(idempotent; verified by read-back) so experiments are tracked server-side:

```bash
evalbuilder publish eval/datasets/NAME.json --dataset-name NAME
```

## Optional: multi-turn simulation

For conversation scenarios (persona, goal, opening, followups, stop conditions),
write a scenarios YAML and run them in the inference phase (against the same agent,
in-process or deployed, with the same workers) or on their own:

```bash
evalbuilder infer eval/datasets/NAME.json --scenarios eval/scenarios.yaml --out eval/results   # runs + simulation-<id>.json
evalbuilder simulate eval/datasets/NAME.json --scenarios eval/scenarios.yaml                   # scenarios only, real tools
evalbuilder simulate eval/datasets/NAME.json --scenarios eval/scenarios.yaml --mock            # dataset mocks; a scenario's mock_strategy selects the engine's strategy
```

Stopping at `max_turns` is a truncation, not a pass. Only expectation-violating
runs are mined back into the dataset — as **pending** cases with
`metadata.source=simulation` for the user to review.

## Rules

- Never re-run the agent just to re-score; `evalbuilder eval` (and `score`) work on
  the stored run artifacts.
- Never leave a deployment you started running without saying so; `evalbuilder deploy
  status DIR` names it and `evalbuilder deploy down DIR` removes it.
- Never run with pending or rejected cases; get an explicit review decision first
  (see agent-eval-dataset's approval gate).
- Report score distributions honestly, including error counts per metric.
