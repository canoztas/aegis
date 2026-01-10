"""Base connector with rate limiting, retry logic, and timeout handling."""
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate_per_second: float):
        """
        Initialize token bucket.

        Args:
            rate_per_second: Maximum requests per second
        """
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.capacity = rate_per_second

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            float: Time waited in seconds
        """
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0

        wait_time = (tokens - self.tokens) / self.rate
        time.sleep(wait_time)
        self.tokens = 0.0
        self.last_update = time.time()
        return wait_time


class BaseConnector(ABC):
    """Base connector for LLM providers."""

    def __init__(
        self,
        base_url: str,
        rate_limit_per_second: float = 10.0,
        timeout_seconds: int = 300,
        retry_max_attempts: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """
        Initialize base connector.

        Args:
            base_url: Base URL for the provider API
            rate_limit_per_second: Maximum requests per second
            timeout_seconds: Request timeout in seconds
            retry_max_attempts: Maximum retry attempts for failed requests
            retry_backoff_factor: Exponential backoff factor
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_factor = retry_backoff_factor

        # Rate limiter
        self.rate_limiter = TokenBucket(rate_limit_per_second)

        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_timeout = 60  # seconds

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False

        if self.circuit_breaker_last_failure is None:
            return False

        elapsed = (datetime.now() - self.circuit_breaker_last_failure).total_seconds()
        if elapsed > self.circuit_breaker_reset_timeout:
            # Reset circuit breaker
            self.circuit_breaker_failures = 0
            self.circuit_breaker_last_failure = None
            return False

        return True

    def record_failure(self):
        """Record a failure for circuit breaker."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.now()

    def record_success(self):
        """Record a success, reset circuit breaker."""
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None

    def retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_max_attempts):
            try:
                # Apply rate limiting
                self.rate_limiter.acquire()

                # Check circuit breaker
                if self.is_circuit_open():
                    raise Exception("Circuit breaker is open - too many failures")

                result = func(*args, **kwargs)
                self.record_success()
                return result

            except Exception as e:
                last_exception = e
                self.record_failure()

                if attempt < self.retry_max_attempts - 1:
                    # Calculate backoff time
                    backoff = self.retry_backoff_factor ** attempt
                    print(
                        f"Retry {attempt + 1}/{self.retry_max_attempts} "
                        f"after {backoff}s due to: {str(e)}"
                    )
                    time.sleep(backoff)
                else:
                    print(f"All retries exhausted. Last error: {str(e)}")

        raise last_exception

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion (synchronous).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with 'text', 'usage', 'model' keys
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion (async).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with 'text', 'usage', 'model' keys
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate completion with streaming (async).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            str: Text chunks
        """
        pass
