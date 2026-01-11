"""Parser that accepts tool outputs directly."""

from typing import Any, Dict, List, Optional

from aegis.models.parsers.base import BaseParser
from aegis.models.schema import ParserResult, FindingCandidate


class ToolResultParser(BaseParser):
    """Pass-through parser for tool outputs."""

    def _normalize_findings(self, items: List[Any]) -> List[FindingCandidate]:
        findings: List[FindingCandidate] = []
        for item in items or []:
            if isinstance(item, FindingCandidate):
                findings.append(item)
            elif isinstance(item, dict):
                try:
                    findings.append(FindingCandidate(**item))
                except Exception:
                    continue
        return findings

    def parse(self, raw_output: Any, context: Optional[Dict[str, Any]] = None) -> ParserResult:
        if isinstance(raw_output, ParserResult):
            return raw_output

        if isinstance(raw_output, dict):
            findings = self._normalize_findings(raw_output.get("findings", []))
            parse_errors = raw_output.get("parse_errors", [])
            raw = raw_output.get("raw_output")
            return ParserResult(findings=findings, parse_errors=parse_errors, raw_output=raw)

        if isinstance(raw_output, list):
            findings = self._normalize_findings(raw_output)
            return ParserResult(findings=findings)

        return ParserResult(
            findings=[],
            parse_errors=["Unsupported tool output"],
            raw_output=str(raw_output),
        )
