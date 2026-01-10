"""OpenAI-compatible connector for cloud providers."""
import json
import requests
import aiohttp
from typing import Dict, Any, Optional, AsyncIterator
from aegis.connectors.base import BaseConnector


class OpenAIConnector(BaseConnector):
    """Connector for OpenAI-compatible APIs (OpenAI, Anthropic, etc)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider_type: str = "openai",
        rate_limit_per_second: float = 5.0,
        timeout_seconds: int = 60,
        retry_max_attempts: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """
        Initialize OpenAI-compatible connector.

        Args:
            base_url: Base URL for the API
            api_key: API key
            provider_type: Provider type (openai, anthropic, custom)
            rate_limit_per_second: Maximum requests per second
            timeout_seconds: Request timeout in seconds
            retry_max_attempts: Maximum retry attempts
            retry_backoff_factor: Exponential backoff factor
        """
        super().__init__(
            base_url=base_url,
            rate_limit_per_second=rate_limit_per_second,
            timeout_seconds=timeout_seconds,
            retry_max_attempts=retry_max_attempts,
            retry_backoff_factor=retry_backoff_factor,
        )
        self.api_key = api_key
        self.provider_type = provider_type.lower()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        if self.provider_type == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

    def _format_request(
        self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], **kwargs
    ) -> Dict[str, Any]:
        """Format request for provider."""
        if self.provider_type == "anthropic":
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
        else:
            # OpenAI-compatible
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens

        # Add provider-specific kwargs
        for key, value in kwargs.items():
            payload[key] = value

        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse provider response."""
        if self.provider_type == "anthropic":
            content = response_data.get("content", [])
            text = content[0].get("text", "") if content else ""
            usage = response_data.get("usage", {})
            return {
                "text": text,
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                "model": response_data.get("model", ""),
                "finish_reason": response_data.get("stop_reason", "stop"),
            }
        else:
            # OpenAI-compatible
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = response_data.get("usage", {})
            return {
                "text": message.get("content", ""),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "model": response_data.get("model", ""),
                "finish_reason": choice.get("finish_reason", "stop"),
            }

    def _get_endpoint(self) -> str:
        """Get API endpoint."""
        if self.provider_type == "anthropic":
            return f"{self.base_url}/messages"
        else:
            return f"{self.base_url}/chat/completions"

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion (synchronous)."""

        def _request():
            url = self._get_endpoint()
            headers = self._get_headers()
            payload = self._format_request(prompt, model, temperature, max_tokens, **kwargs)

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return self._parse_response(response.json())

        return self.retry_with_backoff(_request)

    async def generate_async(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion (async)."""
        url = self._get_endpoint()
        headers = self._get_headers()
        payload = self._format_request(prompt, model, temperature, max_tokens, **kwargs)

        # Apply rate limiting
        self.rate_limiter.acquire()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return self._parse_response(result)

    async def stream_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming (async)."""
        url = self._get_endpoint()
        headers = self._get_headers()
        payload = self._format_request(prompt, model, temperature, max_tokens, **kwargs)
        payload["stream"] = True

        # Apply rate limiting
        self.rate_limiter.acquire()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)

                                if self.provider_type == "anthropic":
                                    # Anthropic streaming format
                                    if chunk.get("type") == "content_block_delta":
                                        delta = chunk.get("delta", {})
                                        if "text" in delta:
                                            yield delta["text"]
                                else:
                                    # OpenAI streaming format
                                    choice = chunk.get("choices", [{}])[0]
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
