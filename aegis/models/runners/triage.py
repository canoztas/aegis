"""Triage runner for initial code classification."""

import logging
from typing import Any, Dict, Optional

from aegis.models.schema import ModelRole, ParserResult
from aegis.models.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class TriageRunner(BaseRunner):
    """
    Runner for triage role - initial vulnerability classification.

    Typically uses:
    - Classification models (CodeBERT, etc.)
    - Fast, lightweight models
    - Returns triage signals or low-detail findings
    """

    def __init__(self, provider: Any, parser: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize triage runner."""
        super().__init__(provider, parser, ModelRole.TRIAGE, config)

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParserResult:
        """
        Execute triage analysis.

        Args:
            prompt: Code snippet to analyze
            context: Context with file_path, line numbers, etc.
            **kwargs: Additional model arguments

        Returns:
            ParserResult with triage signal and/or findings
        """
        context = context or {}

        try:
            # Execute model
            logger.debug(f"Running triage on {context.get('file_path', 'unknown')}")

            if hasattr(self.provider, 'analyze'):
                # Async provider (HF)
                raw_output = await self.provider.analyze(prompt, context, **kwargs)
            elif hasattr(self.provider, 'generate'):
                # Sync provider (Ollama)
                raw_output = self.provider.generate(prompt)
            else:
                raise ValueError(f"Provider {self.provider} has no analyze() or generate() method")

            # Parse output
            result = self.parser.parse(raw_output, context)

            logger.debug(
                f"Triage complete: {len(result.findings)} findings, "
                f"suspicious={result.triage_signal.is_suspicious if result.triage_signal else 'N/A'}"
            )

            return result

        except Exception as e:
            logger.error(f"Triage execution failed: {e}")
            return ParserResult(
                findings=[],
                parse_errors=[f"Execution error: {str(e)}"]
            )
