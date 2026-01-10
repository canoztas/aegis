"""Backward-compatible alias for ModelExecutionEngine (deprecated)."""

from aegis.models.engine import ModelExecutionEngine
from aegis.models.engine import _candidate_to_finding
from aegis.models.engine import create_provider
from aegis.models.engine import ProviderCreationError

__all__ = [
    "ModelExecutionEngine",
    "ProviderCreationError",
    "create_provider",
    "_candidate_to_finding",
]
