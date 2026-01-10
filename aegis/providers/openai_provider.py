"""OpenAI API provider for GPT models."""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Provider for OpenAI API (GPT-3.5, GPT-4, GPT-4-Turbo)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model_name: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Custom base URL (default: https://api.openai.com/v1)
            organization: OpenAI organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Lazy import to avoid requiring openai package if not used
        try:
            import openai
            self.openai = openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        logger.info(f"Initialized OpenAI provider: model={model_name}, base_url={self.base_url}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate completion from OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Enable streaming (not yet implemented)
            **kwargs: Additional OpenAI API parameters

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

            content = response.choices[0].message.content

            # Log token usage for cost tracking
            usage = response.usage
            logger.debug(
                f"OpenAI API call: model={self.model_name}, "
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}, "
                f"total_tokens={usage.total_tokens}"
            )

            return content

        except self.openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except self.openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except self.openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI provider error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate streaming completion from OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional OpenAI API parameters

        Yields:
            Generated text chunks
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current token usage and cost estimates.

        Returns:
            Dictionary with usage information
        """
        # Note: OpenAI doesn't provide a direct usage API
        # This would need to be tracked separately via database
        return {
            "provider": "openai",
            "model": self.model_name,
            "message": "Token usage tracked per request in logs",
        }

    def close(self):
        """Close the OpenAI client."""
        # AsyncOpenAI client handles cleanup automatically
        pass


# Model pricing (USD per 1K tokens) - Updated 2025-01
OPENAI_PRICING = {
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},

    # GPT-4
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},

    # GPT-3.5 Turbo
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost for OpenAI API call.

    Args:
        model_name: OpenAI model name
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = OPENAI_PRICING.get(model_name)
    if not pricing:
        logger.warning(f"No pricing data for model: {model_name}")
        return 0.0

    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost
