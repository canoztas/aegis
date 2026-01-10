"""Base runner interface for role-based model execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from aegis.models.schema import ModelRole, ParserResult


class BaseRunner(ABC):
    """
    Abstract base class for role-based runners.

    Runners orchestrate:
    1. Model execution via provider
    2. Output parsing via parser
    3. Result structuring
    """

    def __init__(
        self,
        provider: Any,
        parser: Any,
        role: ModelRole,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize runner.

        Args:
            provider: Model provider instance
            parser: Output parser instance
            role: Role this runner fulfills
            config: Optional configuration
        """
        self.provider = provider
        self.parser = parser
        self.role = role
        self.config = config or {}

    @abstractmethod
    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParserResult:
        """
        Execute model and parse output.

        Args:
            prompt: Input prompt
            context: Execution context (file_path, snippet, etc.)
            **kwargs: Additional arguments for model

        Returns:
            ParserResult with findings and/or triage signals
        """
        pass

    def _build_prompt(self, template: str, **variables) -> str:
        """
        Build prompt from template and variables.

        Args:
            template: Prompt template with {placeholders}
            **variables: Variables to substitute

        Returns:
            Formatted prompt
        """
        return template.format(**variables)
