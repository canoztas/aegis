"""Base parser interface for model outputs."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from aegis.models.schema import ParserResult


class BaseParser(ABC):
    """
    Abstract base class for output parsers.

    Parsers convert heterogeneous model outputs into structured FindingCandidates.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parser with optional configuration.

        Args:
            config: Parser-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    def parse(self, raw_output: Any, context: Optional[Dict[str, Any]] = None) -> ParserResult:
        """
        Parse model output into structured findings.

        Args:
            raw_output: Raw output from model (str, dict, list, etc.)
            context: Optional context (file_path, snippet, etc.)

        Returns:
            ParserResult with findings, triage signals, and errors
        """
        pass

    def validate_output_size(self, raw_output: str, max_length: int = 100000) -> bool:
        """
        Validate output size to prevent pathological inputs.

        Args:
            raw_output: Raw output string
            max_length: Maximum allowed length

        Returns:
            True if valid, False otherwise
        """
        if isinstance(raw_output, str) and len(raw_output) > max_length:
            return False
        return True
