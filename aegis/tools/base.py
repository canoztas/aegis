"""Base interfaces for tool plugins."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from aegis.models.schema import ParserResult


class ToolPlugin(ABC):
    """Base class for external tool plugins."""

    tool_id: str = "tool"
    name: str = "Tool"
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def analyze_snippet(
        self,
        code: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> ParserResult:
        """Analyze a single code snippet."""

    def analyze_project(
        self,
        source_files: Dict[str, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> ParserResult:
        """Analyze a whole project (defaults to per-file scan)."""
        config = config or {}
        from aegis.models.schema import ParserResult

        findings = []
        parse_errors = []
        for file_path, content in source_files.items():
            try:
                context = {
                    "code": content,
                    "file_path": file_path,
                    "line_start": 1,
                    "line_end": None,
                    "snippet": content,
                }
                result = self.analyze_snippet(content, context, config)
                findings.extend(result.findings)
                parse_errors.extend(result.parse_errors)
            except Exception as exc:
                parse_errors.append(f"{file_path}: {exc}")

        return ParserResult(findings=findings, parse_errors=parse_errors)
