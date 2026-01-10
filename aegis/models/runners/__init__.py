"""Role-based runners for model execution."""

from aegis.models.runners.base import BaseRunner
from aegis.models.runners.triage import TriageRunner
from aegis.models.runners.deep_scan import DeepScanRunner

__all__ = [
    "BaseRunner",
    "TriageRunner",
    "DeepScanRunner",
]
