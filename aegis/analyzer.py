import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from flask import current_app


@dataclass
class Vulnerability:
    file_path: str
    line_number: int
    # low, medium, high, critical
    severity: str
    # 0.0 to 10.0
    score: float
    category: str
    description: str
    recommendation: str
    code_snippet: str


class SecurityAnalyzer:
    def __init__(self) -> None:
        self.ollama_url = current_app.config.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model = current_app.config.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")

    def analyze_files(self, source_files: Dict[str, str]) -> List[Dict[str, Any]]:
        all_vulnerabilities = []

        for file_path, content in source_files.items():
            try:
                vulnerabilities = self._analyze_single_file(file_path, content)
                all_vulnerabilities.extend(vulnerabilities)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        return all_vulnerabilities

    def _analyze_single_file(
        self, file_path: str, content: str
    ) -> List[Dict[str, Any]]:
        file_extension = file_path.split(".")[-1].lower()
        language = self._get_language_from_extension(file_extension)

        prompt = self._create_analysis_prompt(language, content, file_path)

        try:
            response = self._query_ollama(prompt)

            vulnerabilities = self._parse_ollama_response(response, file_path, content)

            return vulnerabilities

        except Exception as e:
            print(f"Error querying Ollama for {file_path}: {e}")
            return []

    def _get_language_from_extension(self, extension: str) -> str:
        language_map = {
            "py": "Python",
            "js": "JavaScript",
            "ts": "TypeScript",
            "java": "Java",
            "cpp": "C++",
            "c": "C",
            "cs": "C#",
            "php": "PHP",
            "rb": "Ruby",
            "go": "Go",
            "rs": "Rust",
            "sql": "SQL",
            "sh": "Shell",
            "bash": "Bash",
            "ps1": "PowerShell",
        }
        return language_map.get(extension, "Unknown")

    def _create_analysis_prompt(
        self, language: str, content: str, file_path: str
    ) -> str:
        return f"""You are a security expert analyzing {language} code for vulnerabilities. 
Analyze the following code from file '{file_path}' and identify security vulnerabilities.

For each vulnerability found, provide a JSON response with this exact structure:
{{
    "vulnerabilities": [
        {{
            "line_number": <integer>,
            "severity": "<low|medium|high|critical>",
            "score": <float between 0.0 and 10.0>,
            "category": "<vulnerability category>",
            "description": "<detailed description>",
            "recommendation": "<how to fix>",
            "code_snippet": "<relevant code lines>"
        }}
    ]
}}

Focus on common security issues like:
- SQL injection
- Cross-site scripting (XSS)
- Authentication/authorization flaws
- Input validation issues
- Cryptographic vulnerabilities
- Information disclosure
- Buffer overflows
- Command injection
- Path traversal
- Insecure deserialization

Code to analyze:
```{language}
{content}
```

Respond ONLY with valid JSON. If no vulnerabilities are found, return {{"vulnerabilities": []}}.
"""

    def _query_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 2000},
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _parse_ollama_response(
        self, response: str, file_path: str, content: str
    ) -> List[Dict[str, Any]]:
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                print(f"No JSON found in Ollama response for {file_path}")
                return []

            json_str = response[json_start:json_end]
            parsed_response = json.loads(json_str)

            vulnerabilities = []

            if "vulnerabilities" in parsed_response:
                for vuln_data in parsed_response["vulnerabilities"]:
                    vuln_dict = {
                        "file_path": file_path,
                        "line_number": vuln_data.get("line_number", 0),
                        "severity": vuln_data.get("severity", "medium"),
                        "score": float(vuln_data.get("score", 5.0)),
                        "category": vuln_data.get("category", "Unknown"),
                        "description": vuln_data.get(
                            "description", "No description provided"
                        ),
                        "recommendation": vuln_data.get(
                            "recommendation", "No recommendation provided"
                        ),
                        "code_snippet": vuln_data.get("code_snippet", ""),
                    }
                    vulnerabilities.append(vuln_dict)

            return vulnerabilities

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Error processing response for {file_path}: {e}")
            return []
