"""Ollama adapter for local LLM models."""
import json
import time
import requests
from typing import Any, Dict, Optional
from aegis.adapters.base import AbstractAdapter
from aegis.models import ModelRequest, ModelResponse, HealthResponse, Finding


class OllamaAdapter(AbstractAdapter):
    """Adapter for Ollama local LLM models."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        display_name: Optional[str] = None,
        connector: Optional[Any] = None,
    ):
        """
        Initialize Ollama adapter.

        Args:
            model_name: Ollama model name
            base_url: Ollama API base URL
            display_name: Display name for the model
            connector: Optional OllamaConnector instance for rate limiting/retry
        """
        adapter_id = f"ollama:{model_name}"
        super().__init__(
            adapter_id=adapter_id,
            display_name=display_name or f"Ollama {model_name}",
            provider="ollama",
            supports_stream=False,
            supports_json=True,
        )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.connector = connector  # Optional: provides rate limiting, retry, circuit breaker

    def predict(self, request: ModelRequest) -> ModelResponse:
        """Run prediction using Ollama API."""
        start_time = time.time()

        try:
            # Build prompt from request
            prompt = request.code_context

            # Use connector if available (provides rate limiting, retry, circuit breaker)
            if self.connector:
                result = self.connector.generate(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=0.1,
                )
                raw_response = result.get("text", "")
            else:
                # Fallback to direct API call (legacy behavior)
                url = f"{self.base_url}/api/chat"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a security scanner. Output only valid JSON matching the provided schema. No commentary."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                    "format": "json"  # Request JSON format
                }

                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=request.timeout,
                )
                response.raise_for_status()
                result = response.json()
                raw_response = result.get("message", {}).get("content", "")

            # Parse JSON response
            findings = self._parse_response(raw_response, request.file_path)

            latency_ms = int((time.time() - start_time) * 1000)

            return ModelResponse(
                model_id=self.id,
                findings=findings,
                usage={},
                raw=raw_response,
                latency_ms=latency_ms,
            )

        except requests.exceptions.RequestException as e:
            return ModelResponse(
                model_id=self.id,
                findings=[],
                error=f"Ollama request failed: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            return ModelResponse(
                model_id=self.id,
                findings=[],
                error=f"Unexpected error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    def _parse_response(self, response_text: str, file_path: str) -> list[Finding]:
        """Parse JSON response into Findings."""
        findings = []
        
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                return findings
            
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            
            if "findings" in parsed:
                for finding_data in parsed["findings"]:
                    try:
                        finding = Finding(
                            name=finding_data.get("name", "Security Issue"),
                            severity=finding_data.get("severity", "medium"),
                            cwe=finding_data.get("cwe", "CWE-20"),
                            file=finding_data.get("file", file_path),
                            start_line=finding_data.get("start_line", 0),
                            end_line=finding_data.get("end_line", 0),
                            message=finding_data.get("message", ""),
                            confidence=finding_data.get("confidence", 0.5),
                            fingerprint=finding_data.get("fingerprint", ""),
                        )
                        findings.append(finding)
                    except Exception as e:
                        # Skip invalid findings - log for debugging
                        import logging
                        logging.debug(f"Failed to parse finding: {e}, data: {finding_data}")
                        continue
            
        except json.JSONDecodeError as e:
            # Try to retry with a simpler extraction
            import logging
            logging.debug(f"JSON decode error in Ollama response: {e}, response_text: {response_text[:200]}")
            pass
        except Exception as e:
            import logging
            logging.debug(f"Error parsing Ollama response: {e}")
            pass
        
        return findings

    def health(self) -> HealthResponse:
        """Check Ollama health."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            return HealthResponse(
                healthy=True,
                message="Ollama is healthy",
                details={"models": response.json().get("models", [])},
            )
        except Exception as e:
            return HealthResponse(
                healthy=False,
                message=f"Ollama health check failed: {str(e)}",
            )

