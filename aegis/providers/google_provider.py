"""Google Generative AI provider for Gemini models."""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from aegis.providers.base import CloudProviderBase
from aegis.utils.provider_errors import handle_google_error

logger = logging.getLogger(__name__)


class GoogleProvider(CloudProviderBase):
    """Provider for Google Generative AI (Gemini Pro, Gemini 1.5)."""

    provider_name = "Google"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ):
        """
        Initialize Google provider.

        Args:
            model_name: Gemini model name (gemini-pro, gemini-1.5-pro, etc.)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, api_key, timeout, **kwargs)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        self._validate_api_key()
        self._create_client()

        logger.info(f"Initialized Google provider: model={model_name}")

    def _validate_api_key(self) -> None:
        """Validate API key is present."""
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

    def _create_client(self) -> None:
        """Create the Google Generative AI client."""
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Google Generative AI package required. Install with: pip install google-generativeai")

    async def _make_generate_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int = 40,
        **kwargs,
    ) -> str:
        """Make the Google Generative AI request."""
        # Google Generative AI doesn't have separate system prompt
        # So we prepend it to the user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

        try:
            # Run sync method in executor to avoid blocking
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config,
            )

            content = response.text

            # Log token usage for cost tracking
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                logger.debug(
                    f"Google API call: model={self.model_name}, "
                    f"prompt_tokens={usage.prompt_token_count}, "
                    f"candidates_tokens={usage.candidates_token_count}, "
                    f"total_tokens={usage.total_token_count}"
                )

            return content

        except Exception as e:
            handle_google_error(e, self.provider_name)

    async def _make_stream_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int = 40,
        **kwargs,
    ) -> Any:
        """Make the Google Generative AI streaming request."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

        try:
            # Run sync streaming in executor
            return await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )

        except Exception as e:
            handle_google_error(e, self.provider_name)

    async def _iterate_stream(self, stream: Any) -> AsyncIterator[str]:
        """Iterate over the Google stream and yield text chunks."""
        for chunk in stream:
            if chunk.text:
                yield chunk.text


# Model pricing (USD per 1M tokens) - Updated 2025-01
# Note: Gemini pricing varies by context length
GOOGLE_PRICING = {
    # Gemini 1.5 Pro
    "gemini-1.5-pro": {
        "input_short": 1.25,  # ≤128K tokens
        "input_long": 2.50,   # >128K tokens
        "output": 5.0,
    },
    "gemini-1.5-pro-latest": {
        "input_short": 1.25,
        "input_long": 2.50,
        "output": 5.0,
    },

    # Gemini 1.5 Flash
    "gemini-1.5-flash": {
        "input_short": 0.075,  # ≤128K tokens
        "input_long": 0.15,    # >128K tokens
        "output": 0.30,
    },
    "gemini-1.5-flash-latest": {
        "input_short": 0.075,
        "input_long": 0.15,
        "output": 0.30,
    },

    # Gemini Pro (legacy)
    "gemini-pro": {
        "input_short": 0.50,
        "input_long": 0.50,
        "output": 1.50,
    },
}


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    use_long_context: bool = False
) -> float:
    """
    Calculate cost for Google Generative AI call.

    Args:
        model_name: Google model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_long_context: Whether input exceeds 128K tokens

    Returns:
        Cost in USD
    """
    pricing = GOOGLE_PRICING.get(model_name)
    if not pricing:
        logger.warning(f"No pricing data for model: {model_name}")
        return 0.0

    input_rate = pricing["input_long"] if use_long_context else pricing["input_short"]
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
