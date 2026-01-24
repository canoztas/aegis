"""Retry logic with exponential backoff for cloud API calls."""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: Optional[Tuple[int, ...]] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff (2.0 = double each time)
            jitter: Add random jitter to delays
            retryable_status_codes: HTTP status codes that should trigger retry (429, 503, etc.)
            retryable_exceptions: Exception types that should trigger retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes or (429, 503, 504)
        self.retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Current retry attempt (0-indexed)
            retry_after: Optional Retry-After header value in seconds

        Returns:
            Delay in seconds
        """
        if retry_after is not None:
            # Respect Retry-After header
            delay = min(retry_after, self.max_delay)
        else:
            # Exponential backoff
            delay = min(
                self.base_delay * (self.exponential_base ** attempt),
                self.max_delay,
            )

        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.25  # +/- 25%
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure non-negative

        return delay


async def retry_async(
    func: Callable[..., Any],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt >= config.max_retries:
                logger.error(f"Retry exhausted after {attempt + 1} attempts: {e}")
                raise

            # Check if exception has retry_after attribute (from HTTP response)
            retry_after = getattr(e, "retry_after", None)
            delay = config.calculate_delay(attempt, retry_after)

            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)

        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable error: {e}")
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable[..., Any],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Retry sync function with exponential backoff.

    Args:
        func: Function to retry
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt >= config.max_retries:
                logger.error(f"Retry exhausted after {attempt + 1} attempts: {e}")
                raise

            retry_after = getattr(e, "retry_after", None)
            delay = config.calculate_delay(attempt, retry_after)

            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Non-retryable error: {e}")
            raise

    if last_exception:
        raise last_exception


class RateLimitError(Exception):
    """Exception for rate limit errors."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServiceUnavailableError(Exception):
    """Exception for service unavailable errors."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


# Default retry configuration for cloud providers
DEFAULT_CLOUD_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    retryable_exceptions=(RateLimitError, ServiceUnavailableError),
)
