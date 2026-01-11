"""Simple regex-based tool plugin (example)."""

import re
from typing import Dict, Any, Optional, List

from aegis.models.schema import FindingCandidate, ParserResult
from aegis.tools.base import ToolPlugin


DEFAULT_PATTERNS = [
    {
        "name": "Use of eval",
        "pattern": r"\beval\s*\(",
        "severity": "high",
        "category": "unsafe_eval",
        "cwe": "CWE-95",
        "description": "Use of eval() can lead to code execution.",
        "recommendation": "Avoid eval(); use safer parsing or explicit handlers.",
    },
    {
        "name": "Hardcoded password",
        "pattern": r"password\s*=\s*['\"]",
        "severity": "medium",
        "category": "hardcoded_secret",
        "cwe": "CWE-798",
        "description": "Hardcoded password detected.",
        "recommendation": "Move secrets to a secure vault or environment variables.",
    },
]


class RegexTool(ToolPlugin):
    tool_id = "regex_basic"
    name = "Regex Scanner"
    description = "Simple regex-based scanner (example tool plugin)."

    def analyze_snippet(
        self,
        code: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> ParserResult:
        cfg = {}
        cfg.update(self.config or {})
        cfg.update(config or {})

        patterns = cfg.get("patterns") or DEFAULT_PATTERNS
        file_path = context.get("file_path", "unknown")
        base_line = int(context.get("line_start") or 1)
        lines = code.splitlines() or [""]

        findings: List[FindingCandidate] = []
        for entry in patterns:
            try:
                pattern = entry.get("pattern")
                if not pattern:
                    continue
                for match in re.finditer(pattern, code, re.MULTILINE):
                    line_offset = code[:match.start()].count("\n")
                    line_start = base_line + line_offset
                    line_text = lines[line_offset] if line_offset < len(lines) else ""
                    findings.append(
                        FindingCandidate(
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_start,
                            snippet=line_text.strip() or line_text,
                            title=entry.get("name") or "Pattern match",
                            category=entry.get("category") or "pattern_match",
                            cwe=entry.get("cwe"),
                            severity=entry.get("severity") or "medium",
                            description=entry.get("description") or "Pattern match detected.",
                            recommendation=entry.get("recommendation"),
                            confidence=float(entry.get("confidence", 0.5)),
                            metadata={"pattern": pattern},
                        )
                    )
            except Exception:
                continue

        return ParserResult(findings=findings)
