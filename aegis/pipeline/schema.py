"""Pipeline schema definitions using Pydantic for validation.

This module defines the structure of Aegis pipelines, including:
- Pipeline configuration and metadata
- Pipeline steps (role-based, tool-based)
- Gating conditions for conditional execution
- Consensus strategies for merging findings
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class StepKind(str, Enum):
    """Type of pipeline step."""
    ROLE = "role"              # Execute models by role (triage, deep_scan, judge)
    MODEL = "model"            # Execute specific model(s)
    TOOL = "tool"              # Execute external SAST tool
    CONSENSUS = "consensus"    # Merge findings using strategy
    GATE = "gate"              # Conditional branching


class ConsensusStrategy(str, Enum):
    """Strategy for merging findings from multiple sources."""
    UNION = "union"                    # All findings (default)
    MAJORITY = "majority"              # Findings seen by >50% of models
    WEIGHTED = "weighted"              # Weighted by model confidence/weight
    JUDGE = "judge"                    # Use judge model to resolve conflicts
    INTERSECTION = "intersection"      # Only findings all models agree on


class GatingOperator(str, Enum):
    """Comparison operators for gating conditions."""
    GT = ">"      # Greater than
    GTE = ">="    # Greater than or equal
    LT = "<"      # Less than
    LTE = "<="    # Less than or equal
    EQ = "=="     # Equal
    NEQ = "!="    # Not equal
    IN = "in"     # Value in list
    NOT_IN = "not_in"  # Value not in list


class GatingCondition(BaseModel):
    """Condition for gating/branching logic.

    Examples:
        # Skip deep scan if triage found no high-severity issues
        field: "findings.high_count"
        operator: ">"
        value: 0

        # Only escalate to judge if models disagree
        field: "findings.agreement_score"
        operator: "<"
        value: 0.7
    """
    field: str = Field(..., description="Field path to evaluate (e.g., 'findings.high_count')")
    operator: GatingOperator = Field(..., description="Comparison operator")
    value: Union[int, float, str, bool, List] = Field(..., description="Value to compare against")

    # Optional: combine multiple conditions
    and_conditions: Optional[List["GatingCondition"]] = Field(None, description="AND these conditions")
    or_conditions: Optional[List["GatingCondition"]] = Field(None, description="OR these conditions")


class PipelineStep(BaseModel):
    """A single step in a pipeline.

    Each step can:
    - Execute models by role or specific model IDs
    - Run external SAST tools
    - Apply consensus strategies
    - Gate/branch based on previous step outputs
    """
    id: str = Field(..., description="Unique step identifier (e.g., 'triage', 'deep_scan')")
    kind: StepKind = Field(..., description="Type of step")

    # Role-based step (kind=role)
    role: Optional[str] = Field(None, description="Role to execute (triage, deep_scan, judge, explain)")

    # Model-based step (kind=model)
    models: Optional[List[str]] = Field(None, description="Specific model IDs to execute")

    # Tool-based step (kind=tool)
    tool_id: Optional[str] = Field(None, description="External tool identifier (e.g., 'bandit', 'sysevr')")
    tool_config: Optional[Dict[str, Any]] = Field(None, description="Tool-specific configuration")

    # Consensus step (kind=consensus)
    strategy: Optional[ConsensusStrategy] = Field(None, description="Consensus strategy")
    sources: Optional[List[str]] = Field(None, description="Step IDs to merge findings from")

    # Gating step (kind=gate)
    condition: Optional[GatingCondition] = Field(None, description="Condition to evaluate")
    on_true: Optional[str] = Field(None, description="Step ID to execute if condition is true")
    on_false: Optional[str] = Field(None, description="Step ID to execute if condition is false")

    # Common fields
    depends_on: Optional[List[str]] = Field(None, description="Step IDs this step depends on (for future DAG support)")
    enabled: bool = Field(True, description="Whether this step is enabled")
    timeout_seconds: Optional[int] = Field(None, description="Override default timeout for this step")

    @model_validator(mode='after')
    def validate_step_kind(self):
        """Validate step has required fields for its kind."""
        if self.kind == StepKind.ROLE and not self.role:
            raise ValueError("Step with kind='role' must have 'role' field")

        if self.kind == StepKind.MODEL and not self.models:
            raise ValueError("Step with kind='model' must have 'models' field")

        if self.kind == StepKind.TOOL and not self.tool_id:
            raise ValueError("Step with kind='tool' must have 'tool_id' field")

        if self.kind == StepKind.CONSENSUS:
            if not self.strategy:
                raise ValueError("Step with kind='consensus' must have 'strategy' field")
            if not self.sources:
                raise ValueError("Step with kind='consensus' must have 'sources' field")

        if self.kind == StepKind.GATE:
            if not self.condition:
                raise ValueError("Step with kind='gate' must have 'condition' field")
            if not self.on_true and not self.on_false:
                raise ValueError("Step with kind='gate' must have at least one of 'on_true' or 'on_false'")

        return self


class PipelineConfig(BaseModel):
    """Complete pipeline configuration.

    Defines a workflow of steps to execute for security analysis.
    """
    name: str = Field(..., description="Pipeline name (e.g., 'triage_then_deep')")
    version: str = Field("1.0", description="Pipeline version for tracking changes")
    description: Optional[str] = Field(None, description="Human-readable description")

    # Steps execute in order (linear for Phase B, DAG later)
    steps: List[PipelineStep] = Field(..., description="Ordered list of pipeline steps")

    # Metadata
    author: Optional[str] = Field(None, description="Pipeline author")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    is_preset: bool = Field(False, description="Whether this is a built-in preset pipeline")

    # Feature flags
    enable_parallelization: bool = Field(True, description="Allow parallel chunk processing within steps")
    store_intermediate_results: bool = Field(False, description="Store findings from each step separately")

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, steps: List[PipelineStep]):
        """Validate step list has at least one step and unique IDs."""
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        step_ids = [step.id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Step IDs must be unique")

        return steps

    @model_validator(mode='after')
    def validate_step_references(self):
        """Validate step references (depends_on, sources, on_true, on_false) exist."""
        step_ids = {step.id for step in self.steps}

        for step in self.steps:
            # Check depends_on references
            if step.depends_on:
                for dep_id in step.depends_on:
                    if dep_id not in step_ids:
                        raise ValueError(f"Step '{step.id}' depends on unknown step '{dep_id}'")

            # Check consensus sources
            if step.kind == StepKind.CONSENSUS and step.sources:
                for source_id in step.sources:
                    if source_id not in step_ids:
                        raise ValueError(f"Consensus step '{step.id}' references unknown source '{source_id}'")

            # Check gate targets
            if step.kind == StepKind.GATE:
                if step.on_true and step.on_true not in step_ids:
                    raise ValueError(f"Gate step '{step.id}' on_true references unknown step '{step.on_true}'")
                if step.on_false and step.on_false not in step_ids:
                    raise ValueError(f"Gate step '{step.id}' on_false references unknown step '{step.on_false}'")

        return self


class PipelineExecutionContext(BaseModel):
    """Runtime context for pipeline execution.

    Tracks state as pipeline progresses through steps.
    """
    pipeline_id: str = Field(..., description="Pipeline identifier")
    scan_id: str = Field(..., description="Scan ID this execution belongs to")

    # Step outputs (step_id -> findings + metadata)
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Output from each completed step")

    # Execution state
    current_step_id: Optional[str] = Field(None, description="Currently executing step")
    completed_steps: List[str] = Field(default_factory=list, description="IDs of completed steps")
    failed_steps: List[str] = Field(default_factory=list, description="IDs of failed steps")

    # Metadata
    started_at: Optional[str] = Field(None, description="ISO timestamp of execution start")
    completed_at: Optional[str] = Field(None, description="ISO timestamp of execution completion")

    model_config = {"arbitrary_types_allowed": True}


# Update forward references for recursive models
GatingCondition.model_rebuild()
