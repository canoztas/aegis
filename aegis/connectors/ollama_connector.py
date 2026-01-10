"""Ollama connector with streaming support."""
import json
import requests
import aiohttp
from typing import Dict, Any, Optional, AsyncIterator
from aegis.connectors.base import BaseConnector


class OllamaConnector(BaseConnector):
    """Connector for Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        rate_limit_per_second: float = 10.0,
        timeout_seconds: int = 600,
        retry_max_attempts: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """Initialize Ollama connector."""
        super().__init__(
            base_url=base_url,
            rate_limit_per_second=rate_limit_per_second,
            timeout_seconds=timeout_seconds,
            retry_max_attempts=retry_max_attempts,
            retry_backoff_factor=retry_backoff_factor,
        )

    def _make_request(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ):
        """Make request to Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Add any extra options
        for key, value in kwargs.items():
            payload["options"][key] = value

        response = requests.post(
            url, json=payload, timeout=self.timeout, stream=stream
        )
        response.raise_for_status()
        return response

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
            response = self._make_request(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )
            result = response.json()

            return {
                "text": result.get("response", ""),
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0),
                },
                "model": model,
                "finish_reason": "stop" if result.get("done") else "length",
            }

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
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        for key, value in kwargs.items():
            payload["options"][key] = value

        # Apply rate limiting
        self.rate_limiter.acquire()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                return {
                    "text": result.get("response", ""),
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0)
                        + result.get("eval_count", 0),
                    },
                    "model": model,
                    "finish_reason": "stop" if result.get("done") else "length",
                }

    async def stream_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming (async)."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        for key, value in kwargs.items():
            payload["options"][key] = value

        # Apply rate limiting
        self.rate_limiter.acquire()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from Ollama."""
        url = f"{self.base_url}/api/pull"

        def _pull():
            response = requests.post(
                url,
                json={"name": model_name, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        return self.retry_with_backoff(_pull)

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        url = f"{self.base_url}/api/tags"

        def _list():
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()

        return self.retry_with_backoff(_list)
