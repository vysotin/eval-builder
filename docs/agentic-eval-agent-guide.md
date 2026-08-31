# Agentic Eval Builder: Direct-Language Guide

Use this guide when you create or review an eval for an agent. Follow the steps in order. Do not collapse everything into one score.

## 1. Define the test contract

Write down:

- who uses the agent and what they need;
- what the agent must do, may do, and must never do;
- its tools, data, permissions, memory, subagents, and environment;
- the final state that proves success;
- invariants that must hold during the whole run;
- the impact and recoverability of each failure;
- the release or rollback decision this eval will support.

Test the whole deployed system: model, prompts, orchestration, tools, retrieval, memory, permissions, retries, and handoffs.

## 2. Map the evaluation surface

Map this flow:

`input → interpret → plan/route → retrieve/use tools/delegate → change state → verify → answer/escalate`

At every step, list:

- decisions and constraints;
- possible wrong actions;
- tool and environment failures;
- safety and privacy hazards;
- evidence you can capture.

## 3. Select suites

Create separate suites:

- **Component:** prompt, router, retriever, memory, and tool contracts. Run on every change.
- **Capability:** hard, realistic tasks. Expect headroom.
- **Regression:** solved tasks and past incidents. Expect almost 100% on critical cases.
- **Safety/security:** authorization, privacy, injection, harmful actions, and fail-safe behavior.
- **Robustness:** malformed data, tool faults, timeouts, partial state, and recovery.
- **Reliability:** repeated trials of important cases.
- **Operational:** latency, cost, tokens, calls, retries, and rate limits.
- **Production:** shadow traffic, sampled reviews, incidents, drift, and user outcomes.

## 4. Choose tests by agent type

| If the agent… | Always test… |
|---|---|
| talks or routes | intent, clarification, policy, topic changes, handoff, tone, multilingual input |
| calls tools/APIs | tool and arguments, durable state, ordering invariants, idempotency, timeouts before/after side effects |
| retrieves or researches | retrieval, authority, freshness, conflicting evidence, grounded claims, citations, abstention, injected content |
| writes code | hidden functional tests, non-regression, security/static checks, scope control, reproducible environment |
| controls a browser/computer | durable outcome, visual/DOM variants, session expiry, navigation recovery, safe side effects, injection |
| analyzes data | data selection, numeric correctness, lineage, leakage, missing data, statistical validity, reproducibility |
| stores memory | correct write/read/delete, correction, time decay, cross-user isolation, sensitive-data retention |
| delegates to agents | routing, handoff integrity, conflict, provenance, loops, trust boundaries, coordination cost |
| runs for a long time | milestones, checkpoint recovery, constraint retention, progress checks, budgets, safe stop |
| works in a high-stakes domain | all applicable tests plus expert review, calibration, fairness, privacy, oversight, escalation, fail-safe behavior |

## 5. Cover edge-case families

For every applicable family, add at least one direct case and more for high-risk boundaries:

- **Task:** ambiguous, contradictory, impossible, out of scope, multiple valid outcomes.
- **Input:** empty, long, typo, dialect, multilingual, structured, multimodal, multiple intents, correction.
- **Evidence:** missing, conflicting, stale, low-quality, hard to retrieve, citation mismatch, injected instruction.
- **Planning:** wrong order, lost constraint, premature completion, loop, repeated bad retry, overreach, bad escalation.
- **Tool:** wrong tool, missing/wrong/boundary argument, malformed output, timeout, rate limit, schema drift, partial commit, race, duplicate retry.
- **State/time:** leaked state, cross-user data, stale memory, wrong deletion, long-horizon drift, replay, DST/date boundary.
- **Coordination:** wrong delegate, lost instruction, conflicting edits, circular handoff, unverified worker claim, privilege crossing.
- **Safety:** direct/indirect injection, jailbreak, unauthorized action, data exposure, excessive agency, argument injection, exfiltration, unfairness, missing override, unsafe failure.
- **Output:** wrong/incomplete answer, hallucinated action, invalid format, bad uncertainty, misleading explanation, inaccessible tone/locale.
- **Operations/eval:** latency/cost blowout, resource exhaustion, missing trace, mock mismatch, flaky fixture, bad grader, reward hacking, contamination, distribution shift.

## 6. Build a compact dataset

Start with 20–50 clear cases if the agent is new. Expand when failures become rarer or differences become smaller.

Collect candidates from requirements, manual checks, incidents, support tickets, production traces, domain experts, threat models, and adapted benchmarks. You may use a model to generate variants, but a qualified reviewer must validate the ground truth.

Use this coverage grid:

`intent × risk × input/user variant × conversation state × tool/environment condition × failure mode × expected behavior`

Do not build the full Cartesian product. Instead:

1. include every catastrophic hazard and mandatory invariant;
2. include valid, invalid, missing, minimum, maximum, and just-outside-boundary values;
3. cover ordinary dimension pairs; add 3-way cases only where risk or incidents justify them;
4. cover multi-turn state transitions and fault timing;
5. add metamorphic variants: paraphrase, irrelevant context, reordering, lower permission, less evidence, replay, and decisive-fact changes;
6. prioritize `impact × likelihood × novelty × uncertainty ÷ cost`;
7. remove redundant cases that add no signal.

Keep separate development, regression, private holdout, red-team, and production-shadow sets. Split by time/source/entity/template family. Protect the holdout from prompt tuning and training leakage.

## 7. Make every case valid

Reject a case unless:

- two qualified reviewers can infer the same success criteria;
- a human or reference solution proves it is solvable;
- every grader assertion follows from the written contract;
- valid alternate solutions can pass;
- shortcut and negative-control solutions fail;
- the initial state is clean and reproducible;
- it covers a named behavior, risk, boundary, or incident.

Store the case purpose, source, risk, tags, initial state, conversation, allowed/prohibited actions, expected final state, invariants, reference solution, grader versions, and environment version.

## 8. Select graders

Use this order:

1. Check the durable outcome or state.
2. Check deterministic invariants, schemas, permissions, citations, cost, and latency.
3. Use a model judge only for semantic or subjective criteria.
4. Use experts for high-stakes decisions, judge calibration, new rubrics, and appeals.

Grade outcomes before paths. Do not require one exact tool sequence when several safe solutions exist. Require a path only when the path is the policy—for example, verify identity before disclosure or obtain consent before purchase.

For a model judge:

- use atomic criteria and pass/fail or pairwise decisions;
- include positive, negative, and borderline examples;
- blind candidate identity and randomize pair order;
- test position and verbosity bias;
- compare against expert labels and track false passes/fails;
- version model, prompt, and settings;
- treat judge errors as errors, not agent failures.

## 9. Run correctly

- Match production model settings, prompts, tools, permissions, and orchestration.
- Reset files, database, browser, caches, clock, and session for every trial.
- Use deterministic mocks for fast tests. Also run smaller contract/staging tests against real integrations.
- Capture messages, observations, tool calls/results, handoffs, state before/after, timing, errors, tokens, and cost.
- Enforce step, time, cost, and side-effect budgets.
- Repeat stochastic and critical cases. Use more repeats for borderline or high-variance cases.
- Separate `agent failure`, `grader error`, and `infrastructure error`.

## 10. Report and gate

Report:

- end-to-end task success;
- hard-invariant and safety violations;
- per-task and per-assertion results;
- pass@k when one of several attempts may succeed;
- pass^k when every attempt must be reliable;
- confidence intervals and paired change versus baseline;
- slices by intent, risk, tool, failure mode, language, turns, user group, and environment version;
- latency, cost, tokens, calls, retries, and recovery rate.

Never ship from the average alone. Use hard gates for critical safety, privacy, authorization, state integrity, and regression cases. Require acceptable lower confidence bounds and no material high-risk slice regression.

## 11. Tune the eval

When the eval is noisy or misleading:

- replace exact paths with outcome checks;
- split vague rubrics into atomic criteria;
- balance “should act” with “should not act” cases;
- inspect durable state, not a success message or button click;
- isolate trial state and separate infrastructure errors;
- add repeats to high-variance cells, not easy duplicates;
- add private tasks when public tests saturate or leak;
- attack graders with shortcuts and reward-hacking attempts;
- recalibrate judges against fresh expert labels.

Do not lower a threshold or rewrite a case just to make the current agent pass.

## 12. Turn failures into coverage

For each production incident or eval failure:

1. save the trace and state safely;
2. find the first causal wrong decision;
3. minimize it into a reproducible case;
4. add it to regression;
5. add boundary and metamorphic neighbors;
6. update the risk and coverage map;
7. verify the fix against all old capability and safety suites.

## Final review command

Before approving an eval, ask:

> Does this suite prove the intended outcome, catch unacceptable paths, cover the highest-risk interactions efficiently, estimate stochastic reliability, resist grader shortcuts, and predict behavior in the real deployment environment?

If any part is unknown, state the limitation and add evidence. Do not convert missing evidence into a pass.

For rationale, examples, sources, and the complete taxonomy, read [Evaluating Agentic Systems: Research Report and Practical Framework](agentic-eval-research-report.md).
