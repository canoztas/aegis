"""Base class for cloud API providers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional

from aegis.utils.retry import (
    retry_async,
    RateLimitError,
    ServiceUnavailableError,
    DEFAULT_CLOUD_RETRY_CONFIG,
)

logger = logging.getLogger(__name__)


class CloudProviderBase(ABC):
    """
    Abstract base class for cloud LLM providers.

    Provides common functionality for:
    - Retry configuration
    - Error logging
    - Usage info
    - Cleanup

    Subclasses must implement:
    - _create_client(): Initialize the provider client
    - _make_generate_request(): Make the actual API call
    - _make_stream_request(): Make the streaming API call
    """

    provider_name: str = "CloudProvider"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize cloud provider base.

        Args:
            model_name: Model identifier at the provider
            api_key: API key (or set via environment variable)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Provider-specific options
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs

    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate API key is present. Raises ValueError if missing."""
        pass

    @abstractmethod
    def _create_client(self) -> None:
        """Create the provider client. Called during __init__ in subclass."""
        pass

    @abstractmethod
    async def _make_generate_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs,
    ) -> str:
        """
        Make the actual API request for text generation.

        This method should handle all provider-specific API interactions
        and error conversion.

        Returns:
            Generated text content
        """
        pass

    @abstractmethod
    async def _make_stream_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs,
    ) -> Any:
        """
        Make the actual API request for streaming generation.

        Returns:
            Stream/iterator object from the provider
        """
        pass

    @abstractmethod
    async def _iterate_stream(self, stream: Any) -> AsyncIterator[str]:
        """
        Iterate over the stream and yield text chunks.

        Args:
            stream: The stream object from _make_stream_request

        Yields:
            Text chunks
        """
        pass

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
        Generate completion from the cloud API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Enable streaming (use generate_stream instead)
            **kwargs: Additional API parameters

        Returns:
            Generated text
        """
        async def _request():
            return await self._make_generate_request(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

        try:
            return await retry_async(_request, config=DEFAULT_CLOUD_RETRY_CONFIG)
        except (RateLimitError, ServiceUnavailableError) as e:
            logger.error(f"{self.provider_name} request failed after retries: {e}")
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} provider error: {e}")
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
        Generate streaming completion from the cloud API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional API parameters

        Yields:
            Generated text chunks
        """
        async def _request():
            return await self._make_stream_request(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

        try:
            stream = await retry_async(_request, config=DEFAULT_CLOUD_RETRY_CONFIG)
            async for chunk in self._iterate_stream(stream):
                yield chunk
        except (RateLimitError, ServiceUnavailableError) as e:
            logger.error(f"{self.provider_name} streaming failed after retries: {e}")
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise

    async def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current token usage and cost estimates.

        Returns:
            Dictionary with usage information
        """
        return {
            "provider": self.provider_name.lower(),
            "model": self.model_name,
            "message": "Token usage tracked per request in logs",
        }

    def close(self) -> None:
        """Close the provider client. Most async clients handle cleanup automatically."""
        pass
