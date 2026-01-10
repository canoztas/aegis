"""Model execution helpers wiring providers, runners, and parsers together."""

import asyncio
import hashlib
from typing import Dict, List, Optional

from aegis.data_models import Finding
from aegis.models.parser_factory import get_parser
from aegis.models.provider_factory import create_provider, ProviderCreationError
from aegis.models.registry import ModelRegistryV2
from aegis.models.runners import TriageRunner, DeepScanRunner
from aegis.models.schema import (
    ModelRecord,
    ModelRole,
    ParserResult,
    FindingCandidate,
)


def _candidate_to_finding(candidate: FindingCandidate) -> Finding:
    """Convert FindingCandidate (parser-level) to core Finding dataclass."""
    fingerprint_src = (
        f"{candidate.file_path}|{candidate.line_start}|{candidate.line_end}|"
        f"{candidate.category}|{candidate.description}"
    )
    fingerprint = hashlib.sha1(fingerprint_src.encode("utf-8")).hexdigest()

    return Finding(
        name=candidate.title or candidate.category,
        severity=str(candidate.severity).lower(),
        cwe=candidate.cwe or candidate.metadata.get("cwe", "CWE-000"),
        file=candidate.file_path,
        start_line=int(candidate.line_start or 0),
        end_line=int(candidate.line_end or candidate.line_start or 0),
        message=candidate.description,
        confidence=float(candidate.confidence or 0.0),
        fingerprint=fingerprint,
    )


class ModelExecutionEngine:
    """Executes registered models using providers, runners, and parsers."""

    def __init__(self, registry: Optional[ModelRegistryV2] = None):
        self.registry = registry or ModelRegistryV2()

    def resolve_model(self, model_id: str) -> Optional[ModelRecord]:
        return self.registry.get_model(model_id)

    def resolve_role(self, role: ModelRole) -> Optional[ModelRecord]:
        return self.registry.get_best_model_for_role(role)

    def _build_runner(self, model: ModelRecord, role: Optional[ModelRole] = None):
        parser_cfg = model.settings.get("parser_config", {})
        parser = get_parser(model.parser_id or "json_schema", parser_cfg)
        try:
            provider = create_provider(model)
        except Exception as exc:
            raise ProviderCreationError(str(exc))

        target_role = role or (model.roles[0] if model.roles else ModelRole.DEEP_SCAN)
        if target_role == ModelRole.TRIAGE:
            return TriageRunner(provider, parser, config=model.settings)
        return DeepScanRunner(provider, parser, config=model.settings)

    def run_model_sync(
        self,
        model: ModelRecord,
        code: str,
        file_path: str,
        role: Optional[ModelRole] = None,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> ParserResult:
        """Run a model synchronously on provided code."""
        context = {
            "code": code,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "snippet": code,
        }
        runner = self._build_runner(model, role)
        prompt = code  # Triage runner uses it directly; deep scan builds template internally
        return asyncio.run(runner.run(prompt, context))

    def run_model_to_findings(
        self,
        model: ModelRecord,
        code: str,
        file_path: str,
        role: Optional[ModelRole] = None,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> List[Finding]:
        """Execute model and convert parser output to Finding dataclasses."""
        result = self.run_model_sync(
            model=model,
            code=code,
            file_path=file_path,
            role=role,
            line_start=line_start,
            line_end=line_end,
        )
        return [_candidate_to_finding(c) for c in result.findings]
