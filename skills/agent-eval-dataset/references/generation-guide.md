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
