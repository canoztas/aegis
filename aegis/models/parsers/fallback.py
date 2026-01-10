"""Fallback parser that returns no findings but records parse failure."""

from typing import Any, Dict, Optional

from aegis.models.parsers.base import BaseParser
from aegis.models.schema import ParserResult


class FallbackParser(BaseParser):
    """Parser used when no parsing strategy is configured."""

    def parse(self, raw_output: Any, context: Optional[Dict[str, Any]] = None) -> ParserResult:
        return ParserResult(
            findings=[],
            parse_errors=["No parser configured for this model"],
            raw_output=str(raw_output) if raw_output is not None else None,
        )
