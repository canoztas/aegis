"""Utility modules for Aegis.

This package contains shared utilities:
- helpers: File handling, device resolution, and general utilities
- retry: Retry logic and error handling for API calls
- provider_errors: Shared error handling for cloud providers
"""

# Re-export from helpers (formerly utils.py)
from aegis.utils.helpers import (
    _detect_cuda_available,
    _detect_mps_available,
    allowed_file,
    chunk_file_lines,
    debug_scan_log,
    detect_language,
    extract_source_files,
    format_score,
    get_severity_color,
    resolve_device,
    resolve_dtype,
)

# Re-export from retry
from aegis.utils.retry import (
    DEFAULT_CLOUD_RETRY_CONFIG,
    RateLimitError,
    RetryConfig,
    ServiceUnavailableError,
    retry_async,
)

# Re-export from provider_errors
from aegis.utils.provider_errors import (
    extract_retry_after,
    handle_anthropic_error,
    handle_google_error,
    handle_openai_error,
)

# Re-export from cwe_lookup
from aegis.utils.cwe_lookup import (
    format_cwe,
    format_cwe_full,
    get_cwe_category,
    get_cwe_info,
    get_cwe_name,
    get_cwe_severity,
    normalize_cwe_id,
)

__all__ = [
    # helpers
    "allowed_file",
    "chunk_file_lines",
    "debug_scan_log",
    "detect_language",
    "extract_source_files",
    "format_score",
    "get_severity_color",
    "resolve_device",
    "resolve_dtype",
    # retry
    "DEFAULT_CLOUD_RETRY_CONFIG",
    "RateLimitError",
    "RetryConfig",
    "ServiceUnavailableError",
    "retry_async",
    # provider_errors
    "extract_retry_after",
    "handle_anthropic_error",
    "handle_google_error",
    "handle_openai_error",
    # cwe_lookup
    "format_cwe",
    "format_cwe_full",
    "get_cwe_category",
    "get_cwe_info",
    "get_cwe_name",
    "get_cwe_severity",
    "normalize_cwe_id",
]
