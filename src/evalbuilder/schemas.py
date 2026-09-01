"""Pydantic models for all evalbuilder artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DATASET_SCHEMA = "evalbuilder/dataset/v1"
AGENT_MAP_SCHEMA = "evalbuilder/agent-map/v1"
RUN_SCHEMA = "evalbuilder/run/v1"
REPORT_SCHEMA = "evalbuilder/score-report/v1"  # per-run evaluator scores (was evalbuilder/report/v1)


class MockRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    matchArgs: dict = Field(default_factory=dict)
    response: Any = None


class Review(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    note: str = ""


class Publication(BaseModel):
    langsmith_example_id: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    inputs: dict
    reference_outputs: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    review: Review = Field(default_factory=Review)
    publication: Publication = Field(default_factory=Publication)


class Target(BaseModel):
    framework: str = "langgraph"
    module: str
    factory: str = "build_agent"


class AgentMap(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_: str = Field(AGENT_MAP_SCHEMA, alias="schema")
    framework: str = "langgraph"
    source_sha256: str = ""
    app: dict = Field(default_factory=dict)
    graph: dict = Field(
        default_factory=lambda: {"nodes": [], "edges": [], "conditional_edges": []}
    )
    tools: list[dict] = Field(default_factory=list)
    skills: list[dict] = Field(default_factory=list)  # Agent Skills (SKILL.md) the agent embeds or loads on demand
    constraints: list[str] = Field(default_factory=list)
    data_domains: dict = Field(default_factory=lambda: {"topics": [], "sources": []})
    intents: list[dict] = Field(default_factory=list)
    scenarios: list[dict] = Field(default_factory=list)
    failure_scenarios: list[dict] = Field(default_factory=list)
    decisions_needed: list[str] = Field(default_factory=list)


class CaseRun(BaseModel):
    case_id: str
    outputs: dict = Field(default_factory=dict)
    trajectory: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    node_path: list[str] = Field(default_factory=list)
    error: str | None = None
    error_class: Literal["none", "agent", "infrastructure"] = "none"


class RunArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(RUN_SCHEMA, alias="schema")
    run_id: str
    dataset_path: str
    dataset_name: str
    mocked: bool
    case_runs: list[CaseRun] = Field(default_factory=list)
    timestamp: str = ""
    agent_model: str | None = None


class Report(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(REPORT_SCHEMA, alias="schema")
    run_id: str
    dataset_name: str
    metrics: dict = Field(default_factory=dict)
    slices: dict = Field(default_factory=dict)
    cases: list[dict] = Field(default_factory=list)


class Dataset(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_: str = Field(DATASET_SCHEMA, alias="schema")
    name: str
    dataset_type: Literal["final_response", "trajectory"]
    target: Target
    mocks: dict = Field(default_factory=dict)
    cases: list[Case] = Field(default_factory=list)
    langsmith: dict = Field(
        default_factory=lambda: {"dataset_id": None, "dataset_name": None}
    )
