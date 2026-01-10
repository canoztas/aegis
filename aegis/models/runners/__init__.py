"""Role-based runners for model execution."""

from aegis.models.runners.base import BaseRunner
from aegis.models.runners.triage import TriageRunner
from aegis.models.runners.deep_scan import DeepScanRunner
from aegis.models.runners.judge import JudgeRunner
from aegis.models.runners.explain import ExplainRunner

__all__ = [
    "BaseRunner",
    "TriageRunner",
    "DeepScanRunner",
    "JudgeRunner",
    "ExplainRunner",
]
