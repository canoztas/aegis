"""Factory helpers for creating output parsers by ID."""

from typing import Optional, Dict, Type
import importlib

from aegis.models.parsers import (
    BaseParser,
    JSONFindingsParser,
    JSONSchemaParser,
    HFTextClassificationParser,
    FallbackParser,
    ToolResultParser,
    ClaudeCodeSecurityParser,
)

# Built-in parser registry
_PARSER_CLASSES: Dict[str, Type[BaseParser]] = {
    "json_schema": JSONSchemaParser,
    "json_findings": JSONFindingsParser,
    "hf_classification": HFTextClassificationParser,
    "fallback": FallbackParser,
    "tool_result": ToolResultParser,
    "claude_code_security": ClaudeCodeSecurityParser,
}


def _load_from_path(path: str) -> Optional[Type[BaseParser]]:
    """Dynamically load a parser class from a full dotted path."""
    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name, None)
        if klass and issubclass(klass, BaseParser):
            return klass
    except Exception:
        return None
    return None


def get_parser(parser_id: Optional[str], config: Optional[Dict] = None) -> BaseParser:
    """
    Instantiate a parser for the given parser ID.

    Falls back to a no-op parser when the ID is missing or unknown.
    """
    config = config or {}
    if not parser_id:
        return FallbackParser(config)

    parser_cls = _PARSER_CLASSES.get(parser_id)
    if not parser_cls:
        # Support fully qualified class path
        parser_cls = _load_from_path(parser_id)

    if not parser_cls:
        return FallbackParser(config)

    return parser_cls(config)
