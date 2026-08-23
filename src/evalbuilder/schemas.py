"""Pydantic models for all evalbuilder artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DATASET_SCHEMA = "evalbuilder/dataset/v1"
AGENT_MAP_SCHEMA = "evalbuilder/agent-map/v1"
RUN_SCHEMA = "evalbuilder/run/v1"
REPORT_SCHEMA = "evalbuilder/report/v1"


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
