"""Rate limiting with token bucket algorithm for API providers."""

import asyncio
import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, rate: float, capacity: float):
        """
        Initialize token bucket.

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from bucket (async).

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"Acquired {tokens} tokens, {self.tokens:.2f} remaining")
                    return True

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                    raise asyncio.TimeoutError(f"Rate limit timeout after {elapsed:.2f}s")

            # Wait before retry (adaptive backoff)
            wait_time = min(1.0, tokens / self.rate)
            await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without waiting (sync).

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimiter:
    """Rate limiter manager for multiple providers."""

    def __init__(self):
        """Initialize rate limiter."""
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def get_bucket(self, provider_key: str, rate: float, capacity: float) -> TokenBucket:
        """
        Get or create token bucket for provider.

        Args:
            provider_key: Unique key for provider (e.g., "openai:gpt-4")
            rate: Token refill rate (requests per second)
            capacity: Maximum burst capacity

        Returns:
            TokenBucket instance
        """
        with self.lock:
            if provider_key not in self.buckets:
                self.buckets[provider_key] = TokenBucket(rate, capacity)
                logger.info(f"Created rate limiter: {provider_key} (rate={rate}/s, capacity={capacity})")
            return self.buckets[provider_key]

    async def acquire(
        self,
        provider_key: str,
        tokens: float = 1.0,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire tokens for provider (creates bucket if needed).

        Args:
            provider_key: Unique key for provider
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens acquired

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            ValueError: If bucket not configured for provider
        """
        if provider_key not in self.buckets:
            raise ValueError(f"Rate limiter not configured for: {provider_key}")

        bucket = self.buckets[provider_key]
        return await bucket.acquire(tokens, timeout)

    def clear_bucket(self, provider_key: str):
        """Remove bucket for provider."""
        with self.lock:
            if provider_key in self.buckets:
                del self.buckets[provider_key]
                logger.info(f"Cleared rate limiter: {provider_key}")

    def clear_all(self):
        """Remove all buckets."""
        with self.lock:
            self.buckets.clear()
            logger.info("Cleared all rate limiters")


# Global rate limiter instance
DEFAULT_RATE_LIMITER = RateLimiter()


# Provider-specific rate limit configurations (requests per minute)
PROVIDER_RATE_LIMITS = {
    # OpenAI rate limits (Tier 1 defaults)
    "openai:gpt-4": {"rpm": 500, "tpm": 10_000},
    "openai:gpt-4-turbo": {"rpm": 500, "tpm": 30_000},
    "openai:gpt-3.5-turbo": {"rpm": 3500, "tpm": 60_000},

    # Anthropic rate limits (Build tier defaults)
    "anthropic:claude-3-opus": {"rpm": 50, "tpm": 40_000},
    "anthropic:claude-3-sonnet": {"rpm": 50, "tpm": 40_000},
    "anthropic:claude-3-haiku": {"rpm": 50, "tpm": 50_000},

    # Google Generative AI (free tier)
    "google:gemini-pro": {"rpm": 60, "tpm": 32_000},
    "google:gemini-1.5-pro": {"rpm": 60, "tpm": 1_000_000},
    "google:gemini-1.5-flash": {"rpm": 60, "tpm": 1_000_000},
}


def configure_rate_limiter(
    rate_limiter: RateLimiter,
    provider_type: str,
    model_name: str,
    rpm: Optional[int] = None,
    burst_multiplier: float = 1.5
):
    """
    Configure rate limiter for provider/model.

    Args:
        rate_limiter: RateLimiter instance
        provider_type: Provider type (openai, anthropic, google)
        model_name: Model name
        rpm: Requests per minute (overrides default)
        burst_multiplier: Burst capacity multiplier
    """
    provider_key = f"{provider_type}:{model_name}"
    limits = PROVIDER_RATE_LIMITS.get(provider_key)

    if limits:
        default_rpm = limits["rpm"]
    else:
        # Fallback defaults
        default_rpm = 60
        logger.warning(f"No rate limit config for {provider_key}, using default: {default_rpm} RPM")

    # Use custom RPM if provided
    rpm = rpm or default_rpm

    # Convert RPM to RPS (rate)
    rate = rpm / 60.0

    # Burst capacity (allow temporary bursts above rate)
    capacity = rate * burst_multiplier

    rate_limiter.get_bucket(provider_key, rate, capacity)
    logger.info(f"Configured rate limiter: {provider_key} (RPM={rpm}, burst={capacity:.2f})")
