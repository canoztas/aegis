"""Model management system for Aegis.

This package provides:
- Model discovery (Ollama, HuggingFace)
- Model registration and lifecycle
- Parser system for heterogeneous outputs
- Role-based runners (triage, deep_scan, judge, explain)
"""

from aegis.models.schema import (
    ModelType,
    ModelRole,
    ModelRecord,
    DiscoveredModel,
    FindingCandidate,
    ModelAvailability,
)
from aegis.models.registry import ModelRegistryV2

__all__ = [
    "ModelType",
    "ModelRole",
    "ModelRecord",
    "DiscoveredModel",
    "FindingCandidate",
    "ModelAvailability",
    "ModelRegistryV2",
]
