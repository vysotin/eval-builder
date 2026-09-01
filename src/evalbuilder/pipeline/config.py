"""Pipeline config — `evalbuilder/pipeline-config/v1` (YAML or JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PIPELINE_CONFIG_SCHEMA = "evalbuilder/pipeline-config/v1"
DEFAULT_MODEL = "claude-cli:claude-sonnet-5"  # Claude Sonnet 5 through the Claude Code subscription CLI

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
    judge: str = DEFAULT_MODEL
    generator: str = DEFAULT_MODEL
    mock: str | None = None  # drives LLM mock responses (mocking.on_miss: llm); None = the generator model


class PerIntent(BaseModel):
    happy: int = 2
    failure: int = 1


class CoverageConfig(BaseModel):
    total_cases: int = 20
    per_intent: PerIntent = Field(default_factory=PerIntent)
    per_failure_category: int = 1
    out_of_intent: int = 2
    multi_turn_share: float = 0.15
    per_tool_edge_cases: int = 2  # schema-derived edge cases (missing/wrong/out-of-range input, malformed output) per tool


class ThresholdsConfig(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    default: float = 0.8
    slice_min: float = 0.5
    overall_pass: float = 0.8

    def for_metric(self, name: str) -> float:
        return self.metrics.get(name, self.default)


class RunsConfig(BaseModel):
    repeats: int = 3
    parallel_intents: int = 4  # intent groups executed concurrently within each repeat (1 = sequential)
    parallel_scoring: int = 4  # case runs scored concurrently within each run report (1 = sequential)
    parallel_simulations: int = 4  # simulation scenarios executed concurrently (1 = sequential)


class MockingConfig(BaseModel):
    required: bool = True
    on_miss: Literal["real", "fallback", "strict", "llm"] = "strict"
    strategies: bool = True  # generate mock-strategies.json (the LLM mock engine's backend behaviours) in the mocks stage
    strategy: str = "default"  # dataset-level strategy id; cases / scenarios may select another
    on_invalid: Literal["fallback", "strict"] = "fallback"  # after the repair round: schema-conformant fallback, or an error
    max_repairs: int = 1


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


class FeedbackEntry(BaseModel):
    at: str = ""
    note: str
    from_stage: str = "dataset"  # the generation stage the reviewer wants rerun


SECTION_COMMENTS = {
    "target": "agent source file (AST discovery) + importable module exposing TOOLS and build_agent",
    "models": "provider:model[@effort] — claude-cli (Claude Code subscription), anthropic|claude, openai, gemini|google, scripted; mock drives LLM mock responses (null = generator)",
    "constraints": "rules the agent must honor; every generation prompt sees them and judges check them",
    "instructions": "free-text general rules for the generator (domain notes, what to emphasise, what to avoid)",
    "feedback": "reviewer comments appended after partial runs; the generation stages read them on rerun",
    "coverage": "case counts: per intent, per failure category, out-of-intent, multi-turn share, schema edge cases per tool",
    "evaluators": "deterministic first (expected_tools, contains), then judges (contract, correctness, openevals, trajectory_llm)",
    "thresholds": "pass rate per metric / per slice / overall",
    "runs": "repeats detect unstable cases and evaluators; parallel_intents / parallel_scoring / parallel_simulations size the run, scoring and simulation thread pools (1 = sequential)",
    "mocking": "layer 1 = deterministic rules; on_miss strict = unmatched call is an error, llm = the LLM mock engine answers from pre-generated strategies (validated against the tool's output schema; on_invalid fallback|strict, max_repairs)",
    "stages": "skip list, retries, optional simulate/publish",
    "review": "auto_approve + approved_by is the explicit human authorization to approve generated cases",
    "output": "artifact directory (default eval/pipeline/<name>)",
    "langsmith": "dataset name for publish (defaults to name)",
}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_: str = Field(PIPELINE_CONFIG_SCHEMA, alias="schema")
    name: str
    target: TargetConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    constraints: list[str] = Field(default_factory=list)
    instructions: str = ""
    feedback: list[FeedbackEntry] = Field(default_factory=list)
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

    @property
    def mock_model_spec(self) -> str:
        """The model behind the LLM mock engine (`models.mock`, else the generator)."""
        return self.models.mock or self.models.generator

    @property
    def llm_mocking(self) -> bool:
        return self.mocking.on_miss == "llm"

    def add_feedback(self, note: str, from_stage: str = "dataset") -> FeedbackEntry:
        entry = FeedbackEntry(at=datetime.now(timezone.utc).isoformat(timespec="seconds"), note=note.strip(), from_stage=from_stage)
        self.feedback.append(entry)
        return entry

    def guidance(self) -> str:
        """The user's free-text instructions and reviewer feedback as one prompt block."""
        parts = []
        if self.instructions.strip():
            parts.append("USER INSTRUCTIONS (general rules and constraints from the user):\n" + self.instructions.strip())
        if self.feedback:
            lines = [f"- [{f.at}] (rerun from {f.from_stage}) {f.note}" for f in self.feedback if f.note.strip()]
            if lines:
                parts.append("REVIEWER FEEDBACK (apply every item; later items refine earlier ones):\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def to_yaml(self) -> str:
        """The config as commented YAML (what `evalbuilder pipeline run` accepts)."""
        data = self.model_dump(by_alias=True)
        lines = [f"schema: {data.pop('schema')}", f"name: {data.pop('name')}"]
        for key, value in data.items():
            comment = SECTION_COMMENTS.get(key)
            if comment:
                lines.append(f"# {comment}")
            lines.append(yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=100).rstrip())
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        return path

    def problems(self) -> list[str]:
        """Semantic checks beyond the schema; empty list means usable."""
        errors: list[str] = []
        if self.schema_ != PIPELINE_CONFIG_SCHEMA:
            errors.append(f"schema must be {PIPELINE_CONFIG_SCHEMA}")
        if not Path(self.target.source).exists():
            errors.append(f"target.source not found: {self.target.source}")
        if self.runs.repeats < 1:
            errors.append("runs.repeats must be >= 1")
        if self.runs.parallel_intents < 1:
            errors.append("runs.parallel_intents must be >= 1")
        if self.runs.parallel_scoring < 1:
            errors.append("runs.parallel_scoring must be >= 1")
        if self.runs.parallel_simulations < 1:
            errors.append("runs.parallel_simulations must be >= 1")
        if self.coverage.total_cases < 1:
            errors.append("coverage.total_cases must be >= 1")
        if not 0 <= self.coverage.multi_turn_share <= 1:
            errors.append("coverage.multi_turn_share must be within [0, 1]")
        if self.coverage.per_tool_edge_cases < 0:
            errors.append("coverage.per_tool_edge_cases must be >= 0")
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
        if self.models.mock and ":" not in self.models.mock:
            errors.append("models.mock must look like provider:model")
        if self.mocking.max_repairs < 0:
            errors.append("mocking.max_repairs must be >= 0")
        if not self.mocking.strategy.strip():
            errors.append("mocking.strategy must be a non-empty strategy id")
        if self.review.auto_approve and not self.review.approved_by:
            errors.append("review.auto_approve requires review.approved_by")
        return errors


def parse_config(text: str, source: str = "<text>") -> PipelineConfig:
    """Validate YAML/JSON text; raises ValueError with one line per problem."""
    data = yaml.safe_load(text) if text.strip() else None
    if not isinstance(data, dict):
        raise ValueError(f"{source}: config must be a mapping")
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as e:
        lines = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ValueError(f"{source}: invalid pipeline config\n  " + "\n  ".join(lines)) from e


def load_config(path: Path) -> PipelineConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: config must be a mapping")
        try:
            return PipelineConfig.model_validate(data)
        except ValidationError as e:
            lines = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            raise ValueError(f"{path}: invalid pipeline config\n  " + "\n  ".join(lines)) from e
    return parse_config(text, str(path))


def template(name: str, source: str, module: str) -> str:
    """A commented starter config with every default spelled out."""
    return f"""schema: {PIPELINE_CONFIG_SCHEMA}
name: {name}
target:
  source: {source}            # agent source file (AST discovery)
  module: {module}            # importable module exposing TOOLS + build_agent
  factory: build_agent        # graph factory ("root node" of the agent)
models:                       # provider:model[@effort]; providers: claude-cli (Claude Code subscription),
                              # anthropic|claude, openai, gemini|google (API keys + `uv sync --extra llm`)
  agent: {DEFAULT_MODEL}    # injected into build_agent(model=...); omit to use the target's default
  judge: {DEFAULT_MODEL}    # LLM-as-judge, e.g. anthropic:claude-sonnet-5, openai:gpt-5, gemini:gemini-2.5-pro
  generator: {DEFAULT_MODEL}  # authors intents, scenarios, cases, mocks, strategies, analysis
  mock: null                  # drives LLM mock responses when mocking.on_miss is llm (null = generator)
constraints: []               # free-text rules the agent must honor, e.g. "Never quote a refund before lookup_order"
instructions: ""              # free-text general rules for the generator (domain notes, emphasis, exclusions)
feedback: []                  # reviewer comments appended after a partial run: [{{at, note, from_stage}}]
coverage:
  total_cases: 20             # floor on the dataset size
  per_intent: {{happy: 2, failure: 1}}
  per_failure_category: 1     # per structurally-applicable failure type
  out_of_intent: 2
  multi_turn_share: 0.15
  per_tool_edge_cases: 2      # schema-derived edge cases per tool (missing/wrong/out-of-range input, malformed output)
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
  parallel_intents: 4         # intent groups run concurrently within each repeat (1 = sequential)
  parallel_scoring: 4         # case runs scored concurrently within each run report (1 = sequential)
  parallel_simulations: 4     # simulation scenarios run concurrently (1 = sequential)
mocking:
  required: true              # every mockable tool gets a fixture (skill loaders are never mocked)
  on_miss: strict             # layer 1 rules miss -> strict: error | llm: the LLM mock engine answers | fallback | real
  strategies: true            # pre-generate mock-strategies.json (backend world + per-tool behaviours for the engine)
  strategy: default           # dataset-level strategy; cases and simulation scenarios may pick another id
  on_invalid: fallback        # engine answer still invalid after the repair round -> fallback (schema sample) | strict (error)
  max_repairs: 1              # repair rounds against the tool's output schema
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
