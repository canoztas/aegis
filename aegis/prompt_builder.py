"""Prompt builder with CWE injection."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from aegis.data_models import ModelRequest


class PromptBuilder:
    """Builds prompts with CWE context and structured output."""

    def __init__(self, cwe_data_path: Optional[str] = None):
        """Initialize prompt builder."""
        if cwe_data_path is None:
            # Default to data/cwe.json relative to project root
            project_root = Path(__file__).parent.parent
            cwe_data_path = project_root / "data" / "cwe.json"
        
        self.cwe_data_path = Path(cwe_data_path)
        self._cwe_data: Optional[Dict[str, Any]] = None
        self._templates: Dict[str, str] = {}

    def _load_cwe_data(self) -> Dict[str, Any]:
        """Load CWE data from JSON file."""
        if self._cwe_data is None:
            if self.cwe_data_path.exists():
                with open(self.cwe_data_path, "r") as f:
                    self._cwe_data = json.load(f)
            else:
                self._cwe_data = {}
        return self._cwe_data

    def _load_template(self, template_id: str) -> str:
        """Load prompt template."""
        if template_id not in self._templates:
            template_path = Path(__file__).parent / "templates" / "llm" / f"{template_id}.txt"
            if template_path.exists():
                with open(template_path, "r") as f:
                    self._templates[template_id] = f.read()
            else:
                # Fallback to default template
                self._templates[template_id] = self._get_default_template()
        return self._templates[template_id]

    def _get_default_template(self) -> str:
        """Get default SAST prompt template."""
        return """SYSTEM: You are a code security scanner. Output only JSON that conforms to the schema. No commentary.

USER:
Language: {LANG}
File: {FILE_PATH}
Snippet (lines {LINE_START}-{LINE_END}):
\"\"\"
{SNIPPET}
\"\"\"

Relevant CWEs (short descriptions):
{CWE_BLOCK}

JSON schema:
{OUTPUT_SCHEMA}

Task:
- Detect vulnerabilities in the snippet.
- If none, return {"findings": []}.
- If any, emit findings with precise line ranges and a stable fingerprint (e.g., sha1 of normalized (CWE + file + lines + short message)).
- Keep messages short and actionable.

Return only JSON."""

    def _get_output_schema(self) -> str:
        """Get JSON output schema for findings."""
        schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "name",
                            "severity",
                            "cwe",
                            "file",
                            "start_line",
                            "end_line",
                            "message",
                            "confidence",
                            "fingerprint",
                        ],
                        "properties": {
                            "name": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "cwe": {"type": "string", "pattern": "^CWE-\\d+$"},
                            "file": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "message": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "fingerprint": {"type": "string"},
                        },
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["findings"],
        }
        # Return JSON schema as-is (we use replace() instead of format() for template substitution)
        return json.dumps(schema, indent=2)

    def _build_cwe_block(self, cwe_ids: Optional[List[str]], language: str) -> str:
        """Build CWE context block for prompt."""
        cwe_data = self._load_cwe_data()
        
        # Auto-select CWEs if none provided
        if not cwe_ids:
            cwe_ids = self._auto_select_cwes(language)
        
        if not cwe_ids:
            return "No specific CWEs selected."
        
        blocks = []
        for cwe_id in cwe_ids:
            if cwe_id in cwe_data:
                cwe_info = cwe_data[cwe_id]
                block = f"{cwe_id}: {cwe_info.get('name', 'Unknown')}\n"
                block += f"  Description: {cwe_info.get('description', '')}\n"
                block += f"  Common patterns: {', '.join(cwe_info.get('patterns', []))}\n"
                block += f"  Remediation: {cwe_info.get('remediation', '')}"
                blocks.append(block)
            else:
                blocks.append(f"{cwe_id}: CWE not found in database")
        
        return "\n\n".join(blocks)

    def _auto_select_cwes(self, language: str) -> List[str]:
        """Auto-select relevant CWEs based on language."""
        language_lower = language.lower()
        
        # Language-specific CWE mappings
        mappings = {
            "python": [
                "CWE-78", "CWE-22", "CWE-502", "CWE-20", "CWE-94",  # injection, path, deserialization
                "CWE-918", "CWE-639", "CWE-862", "CWE-295", "CWE-532",  # SSRF, IDOR, auth, certs, logs
            ],
            "javascript": [
                "CWE-79", "CWE-89", "CWE-352", "CWE-20",  # XSS, SQLi, CSRF, input
                "CWE-601", "CWE-918", "CWE-639", "CWE-862", "CWE-94",  # redirect, SSRF, IDOR, auth, eval
            ],
            "typescript": [
                "CWE-79", "CWE-89", "CWE-352", "CWE-20",  # XSS, SQLi, CSRF, input
                "CWE-601", "CWE-918", "CWE-639", "CWE-862", "CWE-94",  # redirect, SSRF, IDOR, auth, eval
            ],
            "java": [
                "CWE-89", "CWE-502", "CWE-22", "CWE-20",  # SQLi, deserialization, path, input
                "CWE-611", "CWE-918", "CWE-639", "CWE-862", "CWE-287",  # XXE, SSRF, IDOR, auth
            ],
            "php": [
                "CWE-79", "CWE-89", "CWE-434", "CWE-20", "CWE-78",  # XSS, SQLi, upload, input, cmd
                "CWE-502", "CWE-601", "CWE-918", "CWE-862", "CWE-639",  # deser, redirect, SSRF, auth, IDOR
            ],
            "go": [
                "CWE-78", "CWE-22", "CWE-20",  # cmd injection, path, input
                "CWE-918", "CWE-295", "CWE-362", "CWE-862", "CWE-639",  # SSRF, certs, race, auth, IDOR
            ],
            "rust": [
                "CWE-22", "CWE-400", "CWE-20",  # path, resource, input
                "CWE-362", "CWE-918", "CWE-862", "CWE-639",  # race, SSRF, auth, IDOR
            ],
            "c": [
                "CWE-119", "CWE-120", "CWE-125", "CWE-787",  # buffer overflows, OOB read/write
                "CWE-190", "CWE-416", "CWE-476", "CWE-78", "CWE-22",  # integer, UAF, NULL, cmd, path
            ],
            "cpp": [
                "CWE-119", "CWE-120", "CWE-125", "CWE-787",  # buffer overflows, OOB read/write
                "CWE-190", "CWE-416", "CWE-415", "CWE-476", "CWE-401",  # integer, UAF, double-free, NULL, leak
            ],
            "csharp": [
                "CWE-89", "CWE-502", "CWE-22", "CWE-79",  # SQLi, deserialization, path, XSS
                "CWE-611", "CWE-918", "CWE-639", "CWE-862", "CWE-287",  # XXE, SSRF, IDOR, auth
            ],
            "ruby": [
                "CWE-78", "CWE-89", "CWE-79", "CWE-20", "CWE-94",  # cmd, SQLi, XSS, input, eval
                "CWE-502", "CWE-918", "CWE-639", "CWE-862",  # deserialization, SSRF, IDOR, auth
            ],
            "kotlin": [
                "CWE-89", "CWE-502", "CWE-22", "CWE-20",  # SQLi, deserialization, path, input
                "CWE-611", "CWE-918", "CWE-639", "CWE-862",  # XXE, SSRF, IDOR, auth
            ],
        }
        
        # Try exact match first
        if language_lower in mappings:
            return mappings[language_lower]
        
        # Try partial matches
        for lang_key, cwes in mappings.items():
            if lang_key in language_lower or language_lower in lang_key:
                return cwes
        
        # Default to common CWEs (input validation, XSS, SQLi, auth, IDOR)
        return ["CWE-20", "CWE-79", "CWE-89", "CWE-862", "CWE-639"]

    def build_prompt(self, request: ModelRequest) -> str:
        """Build complete prompt from request."""
        template = self._load_template(request.prompt_template_id)
        
        # Extract snippet if line ranges provided
        snippet = request.code_context
        if request.line_start and request.line_end:
            lines = request.code_context.split("\n")
            snippet_lines = lines[request.line_start - 1 : request.line_end]
            snippet = "\n".join(snippet_lines)
            line_start = request.line_start
            line_end = request.line_end
        else:
            line_start = 1
            line_end = len(request.code_context.split("\n"))
        
        # Build CWE block
        cwe_block = self._build_cwe_block(request.cwe_ids, request.language)
        
        # Get output schema
        output_schema = self._get_output_schema()
        
        # Format template - use replace instead of format to avoid issues with JSON schema braces
        prompt = template
        prompt = prompt.replace("{LANG}", request.language)
        prompt = prompt.replace("{FILE_PATH}", request.file_path)
        prompt = prompt.replace("{LINE_START}", str(line_start))
        prompt = prompt.replace("{LINE_END}", str(line_end))
        prompt = prompt.replace("{SNIPPET}", snippet)
        prompt = prompt.replace("{CWE_BLOCK}", cwe_block)
        prompt = prompt.replace("{OUTPUT_SCHEMA}", output_schema)
        
        return prompt

    def build_judge_prompt(
        self,
        candidate_findings: List[Dict[str, Any]],
        file_path: str,
        language: str,
        repo_name: Optional[str] = None,
    ) -> str:
        """Build prompt for judge model in consensus mode."""
        template_path = Path(__file__).parent / "templates" / "llm" / "judge_consensus.txt"
        
        if template_path.exists():
            with open(template_path, "r") as f:
                template = f.read()
        else:
            template = self._get_default_judge_template()
        
        # Format candidate findings as table
        candidate_table = json.dumps(candidate_findings, indent=2)
        
        output_schema = self._get_output_schema()
        
        # Use replace instead of format to avoid issues with JSON schema braces
        prompt = template
        prompt = prompt.replace("{REPO_NAME}", repo_name or "unknown")
        prompt = prompt.replace("{FILE}", file_path)
        prompt = prompt.replace("{CANDIDATE_TABLE}", candidate_table)
        prompt = prompt.replace("{OUTPUT_SCHEMA}", output_schema)
        prompt = prompt.replace("{LANG}", language)
        
        return prompt

    def _get_default_judge_template(self) -> str:
        """Get default judge consensus template."""
        return """SYSTEM: You are a strict security adjudicator. Combine overlapping or duplicate findings from multiple models into a single consistent JSON. Only output valid JSON matching the provided schema. Do not add explanations.

USER:
Repository: {REPO_NAME}
File: {FILE}

Candidate findings from N models (JSON arrays, already normalized):
{CANDIDATE_TABLE}

Schema to follow exactly:
{OUTPUT_SCHEMA}

Rules:
- Merge duplicates (same CWE and overlapping line ranges) into one.
- Prefer more precise line ranges and clearer messages.
- Set confidence = max of contributing findings.
- Keep only plausible CWEs given the language: {LANG}.
- Return only the JSON object, nothing else."""

