# Generation guide

## Variant taxonomy

| variant | What it stresses | Examples |
|---|---|---|
| `happy` | The scenario's main path | "Book a flight SFO to JFK on 2026-09-01"; "What's the weather in Paris?" |
| `boundary` | Limits and edges of validation/routing | Exactly-max-length input; a date on the routing boundary |
| `adversarial` | Robustness to hostile or misleading input | Instruction smuggled into a document; contradictory constraints |
| `linguistic` | Phrasing the prompt did not anticipate | Double negation ("I cannot say it is not reliable"); heavy idiom |
| `multi-turn` | Context retention and correction | User corrects the city mid-conversation; follow-up references "it" |

Default distribution when the user states no preference: happy 40% / failure 30% /
boundary 15% / multi-turn 15%. State the allocation you chose.

## Multi-turn cases

Put the first user message in `inputs.messages`; put every subsequent user message,
in order, in `metadata.user_turns`. The runner replays them sequentially through the
same conversation.

## Golden-answer modes

- `llm_written` (default): you author `reference_outputs` from the agent map and the
  scenario contract. Prefer testable elements (`expected_tools`, `contains`,
  `contract`) over prose-only descriptions.
- `agent_backfilled`: for cases where the correct answer is hard to author, run the
  approved inputs once unmocked (`evalbuilder run`), verify the outputs by hand, and
  copy the verified outputs into `reference_outputs`. Only for cases a human will
  review — an agent's own unverified output is not a golden.

## Input evolutions

Applied after drafting, preserving the answerability contract:

- **concretizing** — replace generic entities with specific ones ("a city" → "Reykjavik").
- **constrained** — add a constraint the agent must honor ("...and keep it under $400").
- **comparative** — require weighing two options ("cheaper: AA100 or the 6am UA?").
- **multicontext** — require combining two facts/topics in one answer.


## Skills and mock strategies

- When the agent map lists `skills[]`, cases for scenarios that list a skill cite
  `skill:<name>` in `metadata.evidence`. For on-demand skills (a `skill_loader` tool
  such as `load_skill`) a correct agent reads the skill before acting, so put that call
  first in `expected_tools` (args `{"name": "<skill>"}`). `skill_misuse` cases push the
  agent to skip or bend a step the skill mandates; the contract states that the agent
  still follows the skill.
- When the dataset mocks with `on_miss: llm`, calls no rule answers are played by the
  LLM mock engine from the dataset's strategies. Select an alternate strategy for a case
  (`metadata.mocks.strategy`) only when its failure mode needs it (e.g. `degraded` for
  `tool_error_handling`); keep `contains` literals to what a rule, a strategy example or
  an explicit behaviour rule guarantees.
