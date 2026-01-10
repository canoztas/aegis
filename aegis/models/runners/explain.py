"""Explain runner for generating human-readable explanations of findings."""

from typing import Any, Dict, List, Optional
from aegis.models.runners.base import BaseRunner
from aegis.models.schema import ParserResult, FindingCandidate


class ExplainRunner(BaseRunner):
    """
    Runner for explain models that generate detailed explanations of findings.

    Explain models receive findings and generate:
    - Human-readable explanations of vulnerabilities
    - Attack scenarios and exploit paths
    - Remediation guidance with code examples
    - CVSS scoring and risk assessment
    - References to CWE/OWASP documentation

    Use case: GPT-4, Claude, or specialized explanation models
    """

    DEFAULT_PROMPT_TEMPLATE = """You are a security expert explaining vulnerabilities to developers. Generate a detailed, educational explanation for the following security finding.

CODE FILE: {file_path}

FINDING:
- Type: {cwe} - {title}
- Severity: {severity}
- Location: Lines {line_start}-{line_end}
- Description: {description}

CODE CONTEXT:
{code}

Provide a comprehensive explanation including:

1. **What is this vulnerability?**
   - Clear description in plain language
   - Why this pattern is dangerous

2. **How can it be exploited?**
   - Realistic attack scenario
   - Example exploit payload (if applicable)

3. **Impact Assessment**
   - What data/systems are at risk?
   - Business impact and compliance concerns

4. **How to fix it**
   - Specific code changes with examples
   - Best practices to prevent recurrence

5. **References**
   - CWE/OWASP links
   - Security guidelines

Return ONLY a JSON object with this structure:
{{
  "finding_id": "{finding_id}",
  "explanation": "Detailed explanation text...",
  "attack_scenario": "How an attacker would exploit this...",
  "impact": "Business and technical impact...",
  "remediation": "Step-by-step fix with code examples...",
  "references": ["https://cwe.mitre.org/...", "https://owasp.org/..."],
  "cvss_score": 7.5,
  "risk_level": "high"
}}
"""

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParserResult:
        """
        Run explain model to generate detailed explanation.

        Args:
            prompt: Formatted explain prompt with finding details
            context: Additional context (finding details, code, file_path, etc.)
            **kwargs: Additional provider-specific arguments

        Returns:
            ParserResult with explanation metadata
        """
        # Build explain prompt with finding context
        if context:
            finding = context.get("finding", {})
            code = context.get("code", "")
            file_path = context.get("file_path", "unknown")

            formatted_prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
                file_path=file_path,
                cwe=finding.get("cwe", "UNKNOWN"),
                title=finding.get("title", "Security Issue"),
                severity=finding.get("severity", "medium"),
                line_start=finding.get("line_start", 0),
                line_end=finding.get("line_end", 0),
                description=finding.get("description", "No description provided"),
                code=code,
                finding_id=finding.get("id", "unknown")
            )
        else:
            formatted_prompt = prompt

        # Call provider (typically a powerful LLM like GPT-4 or Claude)
        raw_output = await self._call_provider(formatted_prompt, context, **kwargs)

        # Parse response (expects JSON with explanation fields)
        result = self.parser.parse(raw_output, context)

        # Attach explanation metadata to findings
        if result.findings and context:
            for candidate in result.findings:
                # Store explanation data in metadata
                if not hasattr(candidate, 'metadata'):
                    candidate.metadata = {}

                candidate.metadata['explained'] = True
                candidate.metadata['explanation_model'] = context.get('model_id', 'unknown')

        return result

    def build_explain_context(self, finding: Dict[str, Any], code: str, file_path: str) -> Dict[str, Any]:
        """
        Build context dictionary for explain prompt.

        Args:
            finding: Finding dictionary with cwe, title, severity, etc.
            code: Source code context
            file_path: Path to the source file

        Returns:
            Context dictionary for prompt formatting
        """
        return {
            "finding": finding,
            "code": code,
            "file_path": file_path,
            "finding_id": finding.get("id", "unknown")
        }
