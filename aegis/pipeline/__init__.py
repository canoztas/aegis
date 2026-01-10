"""Pipeline execution framework for Aegis.

This package provides:
- Pipeline schema definitions (schema.py)
- Pipeline loader and validator (loader.py)
- Pipeline executor with event system (executor.py - Week 2)
- Gating and escalation logic (gating.py - Week 3)
- External tool integration (tools.py - Week 4)
"""

from aegis.pipeline.schema import (
    PipelineConfig,
    PipelineStep,
    StepKind,
    GatingCondition,
    GatingOperator,
    ConsensusStrategy,
    PipelineExecutionContext,
)
from aegis.pipeline.loader import (
    PipelineLoader,
    PipelineRegistry,
)
from aegis.pipeline.executor import (
    PipelineExecutor,
)

__all__ = [
    "PipelineConfig",
    "PipelineStep",
    "StepKind",
    "GatingCondition",
    "GatingOperator",
    "ConsensusStrategy",
    "PipelineExecutionContext",
    "PipelineLoader",
    "PipelineRegistry",
    "PipelineExecutor",
]
