"""Shared error handling utilities for cloud providers."""

import logging
from typing import Any, Optional

from aegis.utils.retry import RateLimitError, ServiceUnavailableError

logger = logging.getLogger(__name__)


def extract_retry_after(error: Any) -> Optional[float]:
    """
    Extract retry-after value from an error's response headers.

    Args:
        error: Exception with potential response.headers.retry-after

    Returns:
        Float seconds to wait, or None if not available
    """
    if hasattr(error, 'response') and error.response and hasattr(error.response, 'headers'):
        retry_after_header = error.response.headers.get('retry-after')
        if retry_after_header:
            try:
                return float(retry_after_header)
            except ValueError:
                pass
    # Also check direct retry_after attribute (Google provider)
    if hasattr(error, 'retry_after'):
        return error.retry_after
    return None


def handle_openai_error(error: Any, openai_module: Any, provider_name: str = "OpenAI") -> None:
    """
    Handle OpenAI SDK errors and convert to standard exceptions.

    Args:
        error: The caught exception
        openai_module: The openai module (for exception type checking)
        provider_name: Name for logging (OpenAI or compatible)

    Raises:
        RateLimitError: For rate limit errors
        ServiceUnavailableError: For service unavailable errors
        The original error: For non-retryable errors
    """
    retry_after = extract_retry_after(error)

    if isinstance(error, openai_module.RateLimitError):
        logger.warning(f"{provider_name} rate limit exceeded: {error}")
        raise RateLimitError(f"{provider_name} rate limit: {error}", retry_after=retry_after)

    if isinstance(error, (openai_module.APIConnectionError, openai_module.APITimeoutError)):
        logger.warning(f"{provider_name} connection/timeout error: {error}")
        raise ServiceUnavailableError(f"{provider_name} service unavailable: {error}")

    if isinstance(error, openai_module.APIStatusError):
        if error.status_code in (503, 504):
            logger.warning(f"{provider_name} service unavailable (status {error.status_code}): {error}")
            raise ServiceUnavailableError(f"{provider_name} service unavailable: {error}", retry_after=retry_after)
        else:
            logger.error(f"{provider_name} API status error: {error}")
            raise

    if isinstance(error, openai_module.AuthenticationError):
        logger.error(f"{provider_name} authentication failed: {error}")
        raise

    if isinstance(error, openai_module.APIError):
        logger.error(f"{provider_name} API error: {error}")
        raise

    # Unknown error type, re-raise
    raise


def handle_anthropic_error(error: Any, anthropic_module: Any, provider_name: str = "Anthropic") -> None:
    """
    Handle Anthropic SDK errors and convert to standard exceptions.

    Args:
        error: The caught exception
        anthropic_module: The anthropic module (for exception type checking)
        provider_name: Name for logging

    Raises:
        RateLimitError: For rate limit errors
        ServiceUnavailableError: For service unavailable errors
        The original error: For non-retryable errors
    """
    retry_after = extract_retry_after(error)

    if isinstance(error, anthropic_module.RateLimitError):
        logger.warning(f"{provider_name} rate limit exceeded: {error}")
        raise RateLimitError(f"{provider_name} rate limit: {error}", retry_after=retry_after)

    if isinstance(error, (anthropic_module.APIConnectionError, anthropic_module.APITimeoutError)):
        logger.warning(f"{provider_name} connection/timeout error: {error}")
        raise ServiceUnavailableError(f"{provider_name} service unavailable: {error}")

    if isinstance(error, anthropic_module.APIStatusError):
        if error.status_code in (503, 504):
            logger.warning(f"{provider_name} service unavailable (status {error.status_code}): {error}")
            raise ServiceUnavailableError(f"{provider_name} service unavailable: {error}", retry_after=retry_after)
        else:
            logger.error(f"{provider_name} API status error: {error}")
            raise

    if isinstance(error, anthropic_module.AuthenticationError):
        logger.error(f"{provider_name} authentication failed: {error}")
        raise

    if isinstance(error, anthropic_module.APIError):
        logger.error(f"{provider_name} API error: {error}")
        raise

    # Unknown error type, re-raise
    raise


def handle_google_error(error: Any, provider_name: str = "Google") -> None:
    """
    Handle Google Generative AI errors and convert to standard exceptions.

    Google uses string-based error detection since exceptions come from google.api_core.

    Args:
        error: The caught exception
        provider_name: Name for logging

    Raises:
        RateLimitError: For rate limit errors
        ServiceUnavailableError: For service unavailable errors
        The original error: For non-retryable errors
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Check for rate limit errors (429 or ResourceExhausted)
    if '429' in error_str or 'rate limit' in error_str or 'ResourceExhausted' in error_type or 'quota' in error_str:
        retry_after = getattr(error, 'retry_after', None)
        logger.warning(f"{provider_name} rate limit exceeded: {error}")
        raise RateLimitError(f"{provider_name} rate limit: {error}", retry_after=retry_after)

    # Check for service unavailable errors (503, 504, or Unavailable)
    if any(x in error_str for x in ['503', '504', 'unavailable', 'deadline exceeded', 'timeout']):
        logger.warning(f"{provider_name} service unavailable: {error}")
        raise ServiceUnavailableError(f"{provider_name} service unavailable: {error}")

    # For other errors, log and re-raise
    logger.error(f"{provider_name} provider error: {error}")
    raise
