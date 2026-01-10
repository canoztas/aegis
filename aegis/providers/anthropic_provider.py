"""Anthropic API provider for Claude models."""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Provider for Anthropic API (Claude 3 Opus, Sonnet, Haiku)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model_name: Claude model name (claude-3-opus-20240229, etc.)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: Custom base URL (default: https://api.anthropic.com)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or "https://api.anthropic.com"
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        # Lazy import to avoid requiring anthropic package if not used
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except ImportError:
            raise ImportError("Anthropic package required. Install with: pip install anthropic")

        logger.info(f"Initialized Anthropic provider: model={model_name}, base_url={self.base_url}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate completion from Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Enable streaming (not yet implemented)
            **kwargs: Additional Anthropic API parameters

        Returns:
            Generated text
        """
        try:
            # Anthropic uses system parameter separately
            create_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **kwargs,
            }

            if system_prompt:
                create_params["system"] = system_prompt

            response = await self.client.messages.create(**create_params)

            content = response.content[0].text

            # Log token usage for cost tracking
            usage = response.usage
            logger.debug(
                f"Anthropic API call: model={self.model_name}, "
                f"input_tokens={usage.input_tokens}, "
                f"output_tokens={usage.output_tokens}"
            )

            return content

        except self.anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except self.anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit exceeded: {e}")
            raise
        except self.anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Anthropic provider error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate streaming completion from Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional Anthropic API parameters

        Yields:
            Generated text chunks
        """
        try:
            create_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **kwargs,
            }

            if system_prompt:
                create_params["system"] = system_prompt

            async with self.client.messages.stream(**create_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    async def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current token usage and cost estimates.

        Returns:
            Dictionary with usage information
        """
        return {
            "provider": "anthropic",
            "model": self.model_name,
            "message": "Token usage tracked per request in logs",
        }

    def close(self):
        """Close the Anthropic client."""
        # AsyncAnthropic client handles cleanup automatically
        pass


# Model pricing (USD per 1M tokens) - Updated 2025-01
ANTHROPIC_PRICING = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},

    # Claude 3 Opus
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},

    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},

    # Claude 3 Haiku
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Legacy models
    "claude-2.1": {"input": 8.0, "output": 24.0},
    "claude-2.0": {"input": 8.0, "output": 24.0},
    "claude-instant-1.2": {"input": 0.8, "output": 2.4},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for Anthropic API call.

    Args:
        model_name: Anthropic model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = ANTHROPIC_PRICING.get(model_name)
    if not pricing:
        logger.warning(f"No pricing data for model: {model_name}")
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
