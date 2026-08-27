"""Pipeline config — `evalbuilder/pipeline-config/v1` (YAML or JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PIPELINE_CONFIG_SCHEMA = "evalbuilder/pipeline-config/v1"

STAGE_NAMES = (
    "preflight", "discover", "map", "mocks", "dataset", "review", "verify",
    "run", "score", "aggregate", "simulate", "publish", "analyze", "report",
)

DEFAULT_EVALUATORS: list[dict] = [
    {"type": "expected_tools"},
    {"type": "contains"},
    {"type": "contract"},
    {"type": "correctness"},
]


class TargetConfig(BaseModel):
    source: str
    module: str
    factory: str = "build_agent"
    root_node: str | None = None


class ModelsConfig(BaseModel):
    agent: str | None = None
    judge: str = "claude-cli:sonnet"
    generator: str = "claude-cli:sonnet"


class PerIntent(BaseModel):
    happy: int = 2
    failure: int = 1


class CoverageConfig(BaseModel):
    total_cases: int = 20
    per_intent: PerIntent = Field(default_factory=PerIntent)
    per_failure_category: int = 1
    out_of_intent: int = 2
    multi_turn_share: float = 0.15


class ThresholdsConfig(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    default: float = 0.8
    slice_min: float = 0.5
    overall_pass: float = 0.8

    def for_metric(self, name: str) -> float:
        return self.metrics.get(name, self.default)


class RunsConfig(BaseModel):
    repeats: int = 3


class MockingConfig(BaseModel):
    required: bool = True
    on_miss: Literal["real", "fallback", "strict"] = "strict"


class StagesConfig(BaseModel):
    skip: list[str] = Field(default_factory=list)
    max_retries: int = 1
    simulate: bool = True
    publish: Literal["auto", "never", "always"] = "auto"


class ReviewConfig(BaseModel):
    auto_approve: bool = False
    approved_by: str = ""
    note: str = ""


class OutputConfig(BaseModel):
    dir: str | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_: str = Field(PIPELINE_CONFIG_SCHEMA, alias="schema")
    name: str
    target: TargetConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    constraints: list[str] = Field(default_factory=list)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    evaluators: list[dict] = Field(default_factory=lambda: [dict(e) for e in DEFAULT_EVALUATORS])
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    runs: RunsConfig = Field(default_factory=RunsConfig)
    mocking: MockingConfig = Field(default_factory=MockingConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    langsmith: dict = Field(default_factory=lambda: {"dataset_name": None})

    @property
    def output_dir(self) -> Path:
        return Path(self.output.dir or f"eval/pipeline/{self.name}")

    def problems(self) -> list[str]:
        """Semantic checks beyond the schema; empty list means usable."""
        errors: list[str] = []
        if self.schema_ != PIPELINE_CONFIG_SCHEMA:
            errors.append(f"schema must be {PIPELINE_CONFIG_SCHEMA}")
        if not Path(self.target.source).exists():
            errors.append(f"target.source not found: {self.target.source}")
        if self.runs.repeats < 1:
            errors.append("runs.repeats must be >= 1")
        if self.coverage.total_cases < 1:
            errors.append("coverage.total_cases must be >= 1")
        if not 0 <= self.coverage.multi_turn_share <= 1:
            errors.append("coverage.multi_turn_share must be within [0, 1]")
        for name, value in [
            ("thresholds.default", self.thresholds.default),
            ("thresholds.slice_min", self.thresholds.slice_min),
            ("thresholds.overall_pass", self.thresholds.overall_pass),
            *[(f"thresholds.metrics.{k}", v) for k, v in self.thresholds.metrics.items()],
        ]:
            if not 0 <= value <= 1:
                errors.append(f"{name} must be within [0, 1]")
        unknown = [s for s in self.stages.skip if s not in STAGE_NAMES]
        if unknown:
            errors.append(f"stages.skip has unknown stages: {unknown}")
        if "report" in self.stages.skip:
            errors.append("stages.skip cannot include report")
        if not self.evaluators:
            errors.append("evaluators must not be empty")
        for i, spec in enumerate(self.evaluators):
            if not isinstance(spec, dict) or not spec.get("type"):
                errors.append(f"evaluators[{i}] needs a type")
        for field in ("judge", "generator"):
            spec = getattr(self.models, field)
            if ":" not in spec:
                errors.append(f"models.{field} must look like provider:model")
        if self.models.agent and ":" not in self.models.agent:
            errors.append("models.agent must look like provider:model")
        if self.review.auto_approve and not self.review.approved_by:
            errors.append("review.auto_approve requires review.approved_by")
        return errors


def load_config(path: Path) -> PipelineConfig:
    path = Path(path)
    text = path.read_text()
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a mapping")
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as e:
        lines = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ValueError(f"{path}: invalid pipeline config\n  " + "\n  ".join(lines)) from e


def template(name: str, source: str, module: str) -> str:
    """A commented starter config with every default spelled out."""
    return f"""schema: {PIPELINE_CONFIG_SCHEMA}
name: {name}
target:
  source: {source}            # agent source file (AST discovery)
  module: {module}            # importable module exposing TOOLS + build_agent
  factory: build_agent        # graph factory ("root node" of the agent)
models:
  agent: claude-cli:sonnet    # injected into build_agent(model=...); omit to use the target's default
  judge: claude-cli:sonnet    # LLM-as-judge (claude-cli:*, anthropic:*, openai:*)
  generator: claude-cli:sonnet  # authors intents, scenarios, cases, mocks, analysis
constraints: []               # free-text rules the agent must honor, e.g. "Never quote a refund before lookup_order"
coverage:
  total_cases: 20             # floor on the dataset size
  per_intent: {{happy: 2, failure: 1}}
  per_failure_category: 1     # per structurally-applicable failure type
  out_of_intent: 2
  multi_turn_share: 0.15
evaluators:                   # deterministic first, then judges (openevals/agentevals)
  - type: expected_tools
  - type: contains
  - type: contract
  - type: correctness
  # - type: trajectory_llm
  # - type: openevals
  #   prompt: CONCISENESS_PROMPT
thresholds:
  default: 0.8                # per-metric pass rate
  metrics: {{expected_tools: 0.9}}
  slice_min: 0.5              # any intent/failure_mode/variant slice below this fails
  overall_pass: 0.8           # mean of metric pass rates
runs:
  repeats: 3                  # repeated runs to detect unstable cases/evaluators
mocking:
  required: true              # every introspected tool gets a mock rule
  on_miss: strict             # unmatched tool call -> error, never a real call
stages:
  skip: []                    # e.g. [simulate, publish]
  max_retries: 1
  simulate: true
  publish: auto               # auto = when LANGSMITH_API_KEY is configured
review:
  auto_approve: false         # set true (with approved_by) to authorize the pipeline to approve generated cases
  approved_by: ""
  note: ""
output:
  dir: null                   # default eval/pipeline/{name}
"""
