"""Google Generative AI provider for Gemini models."""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class GoogleProvider:
    """Provider for Google Generative AI (Gemini Pro, Gemini 1.5)."""

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
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.timeout = timeout
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        # Lazy import to avoid requiring google-generativeai package if not used
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Google Generative AI package required. Install with: pip install google-generativeai")

        logger.info(f"Initialized Google provider: model={model_name}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate completion from Google Generative AI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (prepended to prompt)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            stream: Enable streaming (not yet implemented)
            **kwargs: Additional Google API parameters

        Returns:
            Generated text
        """
        try:
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
            logger.error(f"Google provider error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate streaming completion from Google Generative AI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            **kwargs: Additional Google API parameters

        Yields:
            Generated text chunks
        """
        try:
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

            # Run sync streaming in executor
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google streaming error: {e}")
            raise

    async def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current token usage and cost estimates.

        Returns:
            Dictionary with usage information
        """
        return {
            "provider": "google",
            "model": self.model_name,
            "message": "Token usage tracked per request in logs",
        }

    def close(self):
        """Close the Google client."""
        # No cleanup needed for Google Generative AI
        pass


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
