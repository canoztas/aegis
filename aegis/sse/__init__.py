"""Server-Sent Events (SSE) infrastructure for real-time updates."""

from .stream import SSEManager, SSEConnection

__all__ = ["SSEManager", "SSEConnection"]
