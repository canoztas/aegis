"""OpenAI API provider for GPT models."""

import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from aegis.providers.base import CloudProviderBase
from aegis.utils.provider_errors import handle_openai_error

logger = logging.getLogger(__name__)


class OpenAIProvider(CloudProviderBase):
    """Provider for OpenAI API (GPT-3.5, GPT-4, GPT-4-Turbo)."""

    provider_name = "OpenAI"

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
        super().__init__(model_name, api_key, timeout, max_retries, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization

        self._validate_api_key()
        self._create_client()

        logger.info(f"Initialized OpenAI provider: model={model_name}, base_url={self.base_url}")

    def _validate_api_key(self) -> None:
        """Validate API key is present."""
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    def _create_client(self) -> None:
        """Create the OpenAI async client."""
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

    async def _make_generate_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs,
    ) -> str:
        """Make the OpenAI chat completion request."""
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

        except Exception as e:
            handle_openai_error(e, self.openai, self.provider_name)

    async def _make_stream_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs,
    ) -> Any:
        """Make the OpenAI streaming chat completion request."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                **kwargs,
            )

        except Exception as e:
            handle_openai_error(e, self.openai, self.provider_name)

    async def _iterate_stream(self, stream: Any) -> AsyncIterator[str]:
        """Iterate over the OpenAI stream and yield text chunks."""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


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
