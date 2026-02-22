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

    def _sanitize_json_string(self, text: str) -> str:
        r"""
        Sanitize JSON string by fixing common invalid escape sequences.

        Models sometimes output invalid JSON escapes like \\0 (null char).
        JSON only supports: \" \\ \/ \b \f \n \r \t \uXXXX

        Also handles actual control characters (null bytes, etc.) that models
        may include in their output.
        """
        # First, handle actual control characters (not escape sequences)
        # Replace actual null bytes with escaped representation
        text = text.replace('\x00', '\\u0000')

        # Replace other control characters (0x01-0x1F except \t \n \r)
        def replace_control_char(match):
            char = match.group(0)
            return f'\\u{ord(char):04x}'

        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', replace_control_char, text)

        # Replace invalid \0 escape with literal backslash-zero
        text = re.sub(r'(?<!\\)\\0', r'\\\\0', text)

        # Replace other invalid single-char escapes (but preserve valid ones)
        # Valid: " \ / b f n r t u
        def fix_invalid_escape(match):
            char = match.group(1)
            if char in 'bfnrtu"\\/':
                return match.group(0)  # Keep valid escapes
            return '\\\\' + char  # Convert to literal

        text = re.sub(r'\\([^bfnrtu"\\/])', fix_invalid_escape, text)

        return text

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text, handling fenced code blocks and plain JSON.
        """
        # Sanitize common JSON issues first
        text = self._sanitize_json_string(text)
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

        # Handle unclosed fenced blocks (truncated output)
        unclosed_fence = re.search(r"```(?:json|[a-zA-Z0-9_-]+)?\s*([\s\S]+)", text, flags=re.IGNORECASE)
        if unclosed_fence:
            inner = unclosed_fence.group(1).strip().rstrip("`").strip()
            if inner and inner[0] in "{[":
                brace_json = self._extract_balanced_braces(inner)
                if brace_json:
                    return brace_json

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
        findings_list: Any = None

        if isinstance(data, list):
            findings_list = data
        elif isinstance(data, dict):
            if "findings" in data:
                findings_list = data.get("findings")
            else:
                for key in ("vulnerabilities", "issues", "issue", "vulnerability", "finding", "result"):
                    if key in data:
                        findings_list = data.get(key)
                        break

            if findings_list is None:
                candidate_keys = {"severity", "description", "message", "category", "type", "title"}
                if candidate_keys.intersection(data.keys()):
                    findings_list = [data]

        if isinstance(findings_list, dict):
            if "findings" in findings_list and len(findings_list) == 1:
                findings_list = findings_list.get("findings", [])
            else:
                findings_list = [findings_list]

        # Handle string values: try to parse as JSON or treat as empty
        if isinstance(findings_list, str):
            stripped = findings_list.strip()
            if not stripped or any(p in stripped.lower() for p in ("no vulnerabilit", "no issue", "no finding")):
                logger.info("'findings' is an empty/no-findings string, returning []")
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    findings_list = parsed
                elif isinstance(parsed, dict):
                    findings_list = [parsed]
                else:
                    logger.warning("'findings' field is a string that parsed to non-list/dict: %s", type(parsed))
                    return []
            except json.JSONDecodeError:
                logger.warning("'findings' field is an unparseable string: %s", stripped[:200])
                return []

        if not isinstance(findings_list, list):
            logger.warning("'findings' field is not a list (type=%s): %s", type(findings_list).__name__, str(findings_list)[:200])
            return []

        default_file = context.get("file_path", "unknown")
        default_snippet = context.get("snippet", context.get("code", ""))

        for item in findings_list:
            if not isinstance(item, dict):
                continue

            try:
                severity_value = str(item.get("severity", "medium")).lower()
                if severity_value not in {"critical", "high", "medium", "low", "info"}:
                    severity_value = "medium"
                category_value = (
                    item.get("category")
                    or item.get("type")
                    or item.get("issue_type")
                    or item.get("vulnerability_type")
                    or "unknown"
                )
                description_value = (
                    item.get("description")
                    or item.get("message")
                    or item.get("summary")
                    or ""
                )
                confidence_value = item.get("confidence", 1.0)
                try:
                    confidence_value = float(confidence_value)
                except (TypeError, ValueError):
                    confidence_value = 1.0

                finding = FindingCandidate(
                    file_path=item.get("file_path", item.get("file", default_file)),
                    line_start=item.get("line_start", item.get("start_line", item.get("line", 1))),
                    line_end=item.get("line_end", item.get("end_line")),
                    snippet=item.get("snippet", default_snippet),
                    title=item.get("title") or item.get("type"),
                    category=category_value,
                    cwe=item.get("cwe") or item.get("cwe_id"),
                    severity=severity_value,
                    description=description_value,
                    recommendation=item.get("recommendation"),
                    confidence=confidence_value,
                    metadata=item.get("metadata", {}),
                )
                findings.append(finding)
            except Exception as e:
                logger.warning(f"Failed to parse finding: {e}")
                continue

        return findings


# Backward-compatible alias
JSONSchemaParser = JSONFindingsParser
