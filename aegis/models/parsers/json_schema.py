"""JSON findings parser for structured model outputs."""

import json
import re
import logging
from typing import Any, Dict, Optional, List

from aegis.models.schema import FindingCandidate, ParserResult
from aegis.models.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class JSONFindingsParser(BaseParser):
    """
    Parser for JSON-formatted outputs from generative models.

    Handles:
    - Fenced code blocks (```json ... ```)
    - Plain JSON objects
    - Brace matching for extraction
    - Size guardrails
    """

    def parse(self, raw_output: Any, context: Optional[Dict[str, Any]] = None) -> ParserResult:
        """
        Parse JSON output into FindingCandidates.

        Expected format:
        {
          "findings": [
            {
              "file_path": "...",
              "line_start": 10,
              "line_end": 12,
              "snippet": "...",
              "category": "sql_injection",
              "severity": "high",
              "description": "...",
              "confidence": 0.95
            }
          ]
        }
        """
        context = context or {}
        errors: List[str] = []
        findings: List[FindingCandidate] = []

        # Convert to string if needed
        if isinstance(raw_output, dict):
            try:
                raw_output = json.dumps(raw_output)
            except Exception as e:
                errors.append(f"Failed to serialize dict: {e}")
                return ParserResult(findings=[], parse_errors=errors, raw_output=str(raw_output))

        if not isinstance(raw_output, str):
            errors.append(f"Unexpected output type: {type(raw_output)}")
            return ParserResult(findings=[], parse_errors=errors)

        raw_text = raw_output
        prompt = context.get("prompt")
        if prompt and isinstance(prompt, str):
            prompt_stripped = prompt.strip()
            if raw_output.strip() == prompt_stripped:
                errors.append("Model returned prompt without generating output")
                return ParserResult(findings=[], parse_errors=errors, raw_output=raw_output)
            if raw_output.strip().startswith(prompt_stripped):
                raw_text = raw_output.strip()[len(prompt_stripped):].lstrip()

        if not raw_text.strip():
            errors.append("Empty model output after removing prompt")
            return ParserResult(findings=[], parse_errors=errors, raw_output=raw_output)

        # Validate size
        max_len = self.config.get("max_length", 100_000)
        if not self.validate_output_size(raw_text, max_length=max_len):
            errors.append(f"Output too large (> {max_len} chars)")
            return ParserResult(findings=[], parse_errors=errors, raw_output=raw_text[:1000] + "...")

        # Extract JSON
        extracted_json = self._extract_json(raw_text)
        if not extracted_json:
            errors.append("No valid JSON found in output")
            return ParserResult(findings=[], parse_errors=errors, raw_output=raw_output)

        # Parse findings
        try:
            data = json.loads(extracted_json)
            findings = self._extract_findings(data, context)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return ParserResult(
            findings=findings,
            parse_errors=errors,
            raw_output=raw_output if errors else None,
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text, handling fenced code blocks and plain JSON.
        """
        # Prefer fenced JSON blocks first
        json_fence_pattern = r"```json\s*([\s\S]*?)```"
        for match in re.finditer(json_fence_pattern, text, flags=re.IGNORECASE):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if candidate[0] not in "{[":
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        # Try any fenced block that looks like JSON
        fenced_pattern = r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```"
        for match in re.finditer(fenced_pattern, text):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if candidate[0] not in "{[":
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        # Try to find first balanced braces
        brace_json = self._extract_balanced_braces(text)
        if brace_json:
            return brace_json

        # Last resort: try the whole text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        return None

    def _extract_balanced_braces(self, text: str) -> Optional[str]:
        """Extract first balanced {...} JSON blob."""
        start_idx = text.find("{")
        while start_idx != -1:
            depth = 0
            for idx in range(start_idx, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx : idx + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
            start_idx = text.find("{", start_idx + 1)
        return None

    def _extract_findings(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[FindingCandidate]:
        """
        Extract FindingCandidates from parsed JSON.
        """
        findings = []
        findings_list = data.get("findings", data if isinstance(data, list) else [])

        if isinstance(findings_list, dict):
            findings_list = findings_list.get("findings", [])

        if not isinstance(findings_list, list):
            logger.warning("'findings' field is not a list")
            return []

        default_file = context.get("file_path", "unknown")
        default_snippet = context.get("snippet", context.get("code", ""))

        for item in findings_list:
            if not isinstance(item, dict):
                continue

            try:
                finding = FindingCandidate(
                    file_path=item.get("file_path", default_file),
                    line_start=item.get("line_start", item.get("line", 1)),
                    line_end=item.get("line_end"),
                    snippet=item.get("snippet", default_snippet),
                    title=item.get("title"),
                    category=item.get("category", item.get("type", "unknown")),
                    cwe=item.get("cwe") or item.get("cwe_id"),
                    severity=item.get("severity", "medium"),
                    description=item.get("description", item.get("message", "")),
                    recommendation=item.get("recommendation"),
                    confidence=float(item.get("confidence", 1.0)),
                    metadata=item.get("metadata", {}),
                )
                findings.append(finding)
            except Exception as e:
                logger.warning(f"Failed to parse finding: {e}")
                continue

        return findings


# Backward-compatible alias
JSONSchemaParser = JSONFindingsParser
