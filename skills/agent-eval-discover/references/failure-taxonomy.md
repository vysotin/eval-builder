# Failure taxonomy (structurally gated)

A failure type may be proposed **only when its precondition holds** in the agent's
graph. Cite the evidence that satisfies the precondition.

| failure_type | Precondition | Example case |
|---|---|---|
| `input_validation` | always | Required field missing, empty, or oversized input |
| `provider_error` | always | Model/tool backend times out or errors mid-run |
| `out_of_scope` | always | Request the agent's prompts explicitly do not cover |
| `branch_misrouting` | conditional edges or a router node exist | Input crafted near a routing boundary lands in the wrong branch |
| `tool_misuse` | tools exist | Agent picks the wrong tool or passes malformed args |
| `tool_error_handling` | tools exist | A tool returns an error payload; agent must recover gracefully |
| `retrieval_grounding` | retriever/RAG node exists | Answer contains claims unsupported by retrieved chunks |
| `state_loss` | multi-turn use or a checkpointer exists | Agent forgets a constraint stated two turns earlier |
| `output_contract_violation` | structured output declared | Response JSON missing required keys or violating types |
| `constraint_violation` | authored constraints exist | Agent confirms a booking without an explicit user yes |
| `prompt_injection` | external content reaches the prompt | Retrieved document contains instructions the agent obeys |
| `skill_misuse` | the agent has Agent Skills (`skills[]` in the map) | User asks to skip a step the skill mandates; the agent picks the wrong skill or ignores its instructions |

Evidence tokens: `source:<file>:<line>`, `prompt:<node>`, `tool:<name>`,
`edge:<a>-><b>`, `constraint:<text>`, `skill:<name>`, `schema:<tool>.<field>`, `app:always`.
