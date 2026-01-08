"""LLM adapter for cloud providers (OpenAI, Anthropic, Azure)."""
import json
import time
import requests
from typing import Any, Dict, Optional
from aegis.adapters.base import AbstractAdapter
from aegis.models import ModelRequest, ModelResponse, HealthResponse, Finding


class LLMAdapter(AbstractAdapter):
    """Adapter for cloud LLM providers."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,  # Deprecated: use base_url
        display_name: Optional[str] = None,
        connector: Optional[Any] = None,
    ):
        """
        Initialize LLM adapter.

        Args:
            provider: Provider name (openai, anthropic, etc.)
            model_name: Model name
            api_key: API key
            base_url: API base URL (preferred)
            api_base: API base URL (deprecated, use base_url)
            display_name: Display name for the model
            connector: Optional OpenAIConnector instance for rate limiting/retry
        """
        adapter_id = f"{provider}:{model_name}"
        super().__init__(
            adapter_id=adapter_id,
            display_name=display_name or f"{provider.title()} {model_name}",
            provider=provider,  # type: ignore
            supports_stream=False,
            supports_json=True,
        )
        self.model_name = model_name
        self.api_key = api_key
        # Support both base_url (new) and api_base (legacy)
        self.api_base = base_url or api_base or self._get_default_base(provider)
        self.connector = connector  # Optional: provides rate limiting, retry, circuit breaker

    def _get_default_base(self, provider: str) -> str:
        """Get default API base URL for provider."""
        bases = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "azure": "",  # Must be provided
        }
        return bases.get(provider, "")

    def predict(self, request: ModelRequest) -> ModelResponse:
        """Run prediction using cloud LLM API."""
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                return self._predict_openai(request, start_time)
            elif self.provider == "anthropic":
                return self._predict_anthropic(request, start_time)
            elif self.provider == "azure":
                return self._predict_azure(request, start_time)
            else:
                return ModelResponse(
                    model_id=self.id,
                    findings=[],
                    error=f"Unsupported provider: {self.provider}",
                    latency_ms=int((time.time() - start_time) * 1000),
                )
        except Exception as e:
            return ModelResponse(
                model_id=self.id,
                findings=[],
                error=f"Unexpected error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    def _predict_openai(self, request: ModelRequest, start_time: float) -> ModelResponse:
        """Predict using OpenAI API."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a security scanner. Output only valid JSON matching the provided schema. No commentary."
                },
                {
                    "role": "user",
                    "content": request.code_context
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, json=payload, headers=headers, timeout=request.timeout)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"]
        findings = self._parse_response(raw_response, request.file_path)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return ModelResponse(
            model_id=self.id,
            findings=findings,
            usage=result.get("usage"),
            raw=raw_response,
            latency_ms=latency_ms,
        )

    def _predict_anthropic(self, request: ModelRequest, start_time: float) -> ModelResponse:
        """Predict using Anthropic API."""
        url = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "system": "You are a security scanner. Output only valid JSON matching the provided schema. No commentary.",
            "messages": [
                {
                    "role": "user",
                    "content": request.code_context
                }
            ],
        }

        response = requests.post(url, json=payload, headers=headers, timeout=request.timeout)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result["content"][0]["text"]
        findings = self._parse_response(raw_response, request.file_path)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return ModelResponse(
            model_id=self.id,
            findings=findings,
            usage=result.get("usage"),
            raw=raw_response,
            latency_ms=latency_ms,
        )

    def _predict_azure(self, request: ModelRequest, start_time: float) -> ModelResponse:
        """Predict using Azure OpenAI API."""
        # Similar to OpenAI but with different endpoint structure
        url = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version=2024-02-15-preview"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a security scanner. Output only valid JSON matching the provided schema. No commentary."
                },
                {
                    "role": "user",
                    "content": request.code_context
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, json=payload, headers=headers, timeout=request.timeout)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"]
        findings = self._parse_response(raw_response, request.file_path)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return ModelResponse(
            model_id=self.id,
            findings=findings,
            usage=result.get("usage"),
            raw=raw_response,
            latency_ms=latency_ms,
        )

    def _parse_response(self, response_text: str, file_path: str) -> list[Finding]:
        """Parse JSON response into Findings."""
        findings = []
        
        try:
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
                    except Exception:
                        continue
            
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
        
        return findings

    def health(self) -> HealthResponse:
        """Check LLM provider health."""
        try:
            # Simple health check - try a minimal request
            if self.provider == "openai":
                url = f"{self.api_base}/models"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
            elif self.provider == "anthropic":
                # Anthropic doesn't have a simple health endpoint
                return HealthResponse(
                    healthy=True,
                    message="Anthropic adapter ready (health check not available)",
                )
            
            return HealthResponse(
                healthy=True,
                message=f"{self.provider} is healthy",
            )
        except Exception as e:
            return HealthResponse(
                healthy=False,
                message=f"{self.provider} health check failed: {str(e)}",
            )

