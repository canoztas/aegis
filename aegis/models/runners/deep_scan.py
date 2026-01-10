"""Deep scan runner for detailed vulnerability analysis."""

import logging
from typing import Any, Dict, Optional

from aegis.models.schema import ModelRole, ParserResult
from aegis.models.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class DeepScanRunner(BaseRunner):
    """
    Runner for deep_scan role - detailed vulnerability analysis.

    Typically uses:
    - Generative models (CodeAstra, GPT-4, etc.)
    - Returns structured findings with details
    - Higher latency than triage
    """

    DEFAULT_PROMPT_TEMPLATE = """Analyze the following code for security vulnerabilities.
Return ONLY valid JSON matching exactly this structure:
{{
  "findings": [
    {{
      "file_path": "{file_path}",
      "line_start": <number>,
      "line_end": <number>,
      "snippet": "<code snippet>",
      "cwe": "<CWE-id or null>",
      "severity": "critical|high|medium|low|info",
      "confidence": <0.0-1.0>,
      "title": "<short title>",
      "category": "<vulnerability type/category>",
      "description": "<detailed explanation>",
      "recommendation": "<how to fix>"
    }}
  ]
}}

Code to analyze:
```
{code}
```

Return only the JSON payload. No prose."""

    def __init__(self, provider: Any, parser: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize deep scan runner."""
        super().__init__(provider, parser, ModelRole.DEEP_SCAN, config)
        tmpl = self.config.get("prompt_template")
        # Some configs may pass None; fall back to default
        self.prompt_template = tmpl if tmpl else self.DEFAULT_PROMPT_TEMPLATE

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParserResult:
        """
        Execute deep scan analysis.

        Args:
            prompt: Code snippet to analyze (or use context['code'])
            context: Context with file_path, code, etc.
            **kwargs: Additional model arguments

        Returns:
            ParserResult with detailed findings
        """
        context = dict(context or {})

        # Build structured prompt
        code = context.get("code", prompt)
        file_path = context.get("file_path", "unknown")

        formatted_prompt = self._build_prompt(
            self.prompt_template,
            code=code,
            file_path=file_path
        )
        context["prompt"] = formatted_prompt

        try:
            logger.debug(f"Running deep scan on {file_path}")

            # Execute model
            if hasattr(self.provider, 'analyze'):
                # Async provider (HF)
                raw_output = await self.provider.analyze(formatted_prompt, context, **kwargs)
            elif hasattr(self.provider, 'generate'):
                # Sync provider (Ollama)
                raw_output = self.provider.generate(formatted_prompt)
            else:
                raise ValueError(f"Provider {self.provider} has no analyze() or generate() method")

            if raw_output is None:
                return ParserResult(findings=[], parse_errors=["Empty model response"], raw_output=None)

            # Parse output
            result = self.parser.parse(raw_output, context)

            logger.debug(f"Deep scan complete: {len(result.findings)} findings")

            return result

        except Exception as e:
            logger.error(f"Deep scan execution failed: {e}")
            return ParserResult(
                findings=[],
                parse_errors=[f"Execution error: {str(e)}"]
            )
