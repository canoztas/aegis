"""Judge runner for reviewing and scoring findings from other models."""

from typing import Any, Dict, List, Optional
from aegis.models.runners.base import BaseRunner
from aegis.models.schema import ParserResult, FindingCandidate


class JudgeRunner(BaseRunner):
    """
    Runner for judge models that review findings from other models.

    Judge models receive all candidate findings and determine:
    - Which findings are valid
    - Confidence scores for each finding
    - Final severity levels
    - Consolidated descriptions

    Use case: GPT-4, Claude, or specialized judge models
    """

    DEFAULT_PROMPT_TEMPLATE = """You are a security code review expert. Review the following vulnerability findings from multiple security analysis models.

For each finding, determine:
1. Is it a real vulnerability? (true/false)
2. Confidence level (0.0 to 1.0)
3. Severity (critical/high/medium/low/info)
4. Consolidated description

CODE FILE: {file_path}

FINDINGS TO REVIEW:
{findings_json}

Return ONLY a JSON object with this structure:
{{
  "findings": [
    {{
      "original_id": "finding_1",
      "is_valid": true,
      "confidence": 0.95,
      "severity": "high",
      "cwe": "CWE-89",
      "title": "SQL Injection vulnerability",
      "description": "Consolidated description of the vulnerability",
      "recommendation": "Use parameterized queries",
      "line_start": 42,
      "line_end": 45
    }}
  ]
}}

Only include findings that are valid (is_valid: true). Be conservative - it's better to reject false positives.
"""

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParserResult:
        """
        Run judge model to review findings.

        Args:
            prompt: Formatted judge prompt with findings to review
            context: Additional context (file_path, findings_json, etc.)
            **kwargs: Additional provider-specific arguments

        Returns:
            ParserResult with validated findings
        """
        # Build judge prompt with findings context
        if context:
            findings_json = context.get("findings_json", "")
            file_path = context.get("file_path", "unknown")

            formatted_prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
                file_path=file_path,
                findings_json=findings_json
            )
        else:
            formatted_prompt = prompt

        # Call provider (typically a powerful LLM)
        raw_output = await self._call_provider(formatted_prompt, context, **kwargs)

        # Parse response (expects JSON with findings array)
        result = self.parser.parse(raw_output, context)

        return result

    def _build_findings_context(self, findings: List[FindingCandidate]) -> str:
        """
        Build JSON representation of findings for judge prompt.

        Args:
            findings: List of candidate findings to review

        Returns:
            JSON string representation
        """
        import json

        findings_data = []
        for idx, finding in enumerate(findings, 1):
            findings_data.append({
                "id": f"finding_{idx}",
                "cwe": finding.cwe or "UNKNOWN",
                "severity": finding.severity or "medium",
                "confidence": finding.confidence or 0.5,
                "title": finding.title or "Untitled",
                "description": finding.description or "",
                "line_start": finding.line_start,
                "line_end": finding.line_end,
                "source_model": getattr(finding, "source_model", "unknown")
            })

        return json.dumps(findings_data, indent=2)
