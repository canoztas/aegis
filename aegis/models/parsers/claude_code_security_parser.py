"""Parser for Claude Code Security provider output.

Handles the two-layer JSON structure produced by the Claude Code CLI:

1. **CLI envelope** — ``--output-format json`` wraps Claude's response in
   ``{"type": "result", "result": "...", ...}``.  The provider normally
   unwraps this before handing text to the parser, but if the raw envelope
   is passed through we handle it here as well.
2. **Findings JSON** — Claude's response text contains a JSON object (or
   array) with vulnerability findings.  This layer is extracted with the
   same robust logic used by :class:`JSONFindingsParser` (fenced code blocks,
   balanced-brace extraction, etc.).
"""

import json
import logging
from typing import Any, Dict, Optional

from aegis.models.parsers.json_schema import JSONFindingsParser
from aegis.models.schema import ParserResult

logger = logging.getLogger(__name__)


class ClaudeCodeSecurityParser(JSONFindingsParser):
    """Parser specialised for Claude Code Security output.

    Extends :class:`JSONFindingsParser` with:

    * Automatic unwrapping of the Claude Code CLI JSON envelope when the
      raw subprocess output is passed directly.
    * Graceful handling of empty / ``"No vulnerabilities found"`` responses.
    """

    def parse(
        self,
        raw_output: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ParserResult:
        """Parse Claude Code Security output into findings.

        Args:
            raw_output: Either the ``result`` text already extracted by the
                provider, or the full CLI JSON envelope (dict or string).
            context: Execution context (``file_path``, ``snippet``, etc.).

        Returns:
            :class:`ParserResult` with findings and any parse errors.
        """
        # ── Step 1: Unwrap CLI envelope if present ──────────────────────
        text = self._unwrap_envelope(raw_output)

        logger.debug(
            "[claude-code-parser] unwrapped text (first 500 chars): %s",
            str(text)[:500],
        )

        # ── Step 2: Fast-path for clearly empty responses ───────────────
        if self._is_empty_response(text):
            return ParserResult(findings=[], parse_errors=[])

        # ── Step 3: Delegate to the robust JSON extraction in the parent
        return super().parse(text, context)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _unwrap_envelope(raw_output: Any) -> Any:
        """If *raw_output* is a CLI JSON envelope, extract the ``result``.

        Supports both ``dict`` (already parsed) and ``str`` (raw JSON) forms.
        Returns the original value unchanged if it is not a CLI envelope.
        """
        if isinstance(raw_output, dict):
            if "result" in raw_output and raw_output.get("type") == "result":
                inner = raw_output["result"]
                if raw_output.get("is_error"):
                    logger.warning(
                        "Claude Code returned an error result: %s",
                        str(inner)[:200],
                    )
                return str(inner) if inner is not None else ""
            return raw_output

        if not isinstance(raw_output, str):
            return raw_output

        text = raw_output.strip()
        if not text:
            return text

        # Try to detect a CLI envelope in the string
        if text.startswith("{") and '"type"' in text[:80]:
            try:
                envelope = json.loads(text)
                if isinstance(envelope, dict) and envelope.get("type") == "result":
                    inner = envelope.get("result", "")
                    if envelope.get("is_error"):
                        logger.warning(
                            "Claude Code returned an error result: %s",
                            str(inner)[:200],
                        )
                    return str(inner) if inner is not None else ""
            except json.JSONDecodeError:
                pass

        return text

    @staticmethod
    def _is_empty_response(text: Any) -> bool:
        """Return *True* for responses that indicate no findings."""
        if not isinstance(text, str):
            return False
        stripped = text.strip().lower()
        if not stripped:
            return True

        # Common "nothing found" phrasings from Claude
        no_finding_phrases = (
            "no vulnerabilities",
            "no security issues",
            "no findings",
            "code appears secure",
            "no issues found",
            "no security vulnerabilities",
        )
        # Only match if the entire response is a short "nothing found" message
        if len(stripped) < 200:
            for phrase in no_finding_phrases:
                if phrase in stripped:
                    return True

        # Also handle {"findings": []} fast-path
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                findings = obj.get("findings", obj.get("vulnerabilities"))
                if isinstance(findings, list) and len(findings) == 0:
                    return True
            if isinstance(obj, list) and len(obj) == 0:
                return True
        except (json.JSONDecodeError, TypeError):
            pass

        return False
