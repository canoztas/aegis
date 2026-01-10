"""Server-Sent Events (SSE) manager for real-time progress updates.

Provides:
- SSEConnection: Individual SSE connection to a client
- SSEManager: Manages multiple connections and broadcasts events
- Thread-safe event broadcasting
- Automatic connection cleanup
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from queue import Queue, Empty
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=False)
class SSEConnection:
    """Individual SSE connection to a client."""

    scan_id: str
    client_id: str
    queue: Queue = field(default_factory=Queue, compare=False, hash=False)
    connected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(), compare=False)
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat(), compare=False)

    def __hash__(self):
        """Make connection hashable based on scan_id + client_id."""
        return hash((self.scan_id, self.client_id))

    def __eq__(self, other):
        """Compare connections by scan_id + client_id."""
        if not isinstance(other, SSEConnection):
            return False
        return self.scan_id == other.scan_id and self.client_id == other.client_id

    def send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Send an event to this connection.

        Args:
            event_type: Type of event (e.g., 'step_start', 'progress')
            data: Event data payload
        """
        event = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.queue.put(event)
        self.last_activity = datetime.utcnow().isoformat()

    def get_events(self, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        Get pending events from queue.

        Args:
            timeout: Maximum time to wait for events (seconds)

        Returns:
            List of events
        """
        events = []
        deadline = time.time() + timeout

        try:
            # Wait for first event (with timeout for keepalive)
            wait_time = min(timeout, 15.0)  # Max 15s between keepalives
            event = self.queue.get(timeout=wait_time)
            events.append(event)

            # Drain remaining events (non-blocking)
            while time.time() < deadline:
                try:
                    event = self.queue.get_nowait()
                    events.append(event)
                except Empty:
                    break

        except Empty:
            # Timeout - send keepalive comment
            pass

        return events


class SSEManager:
    """
    Manages SSE connections and event broadcasting.

    Thread-safe manager for broadcasting events to multiple clients.
    Each scan can have multiple connected clients.

    Usage:
        manager = SSEManager()

        # Client connects
        connection = manager.connect(scan_id="scan-123", client_id="client-1")

        # Broadcast events
        manager.broadcast(scan_id="scan-123", event_type="progress", data={"percent": 50})

        # Client disconnects
        manager.disconnect(scan_id="scan-123", client_id="client-1")
    """

    def __init__(self):
        """Initialize SSE manager."""
        # scan_id -> set of SSEConnection objects
        self._connections: Dict[str, Set[SSEConnection]] = {}
        self._lock = threading.RLock()

    def connect(self, scan_id: str, client_id: Optional[str] = None) -> SSEConnection:
        """
        Register a new SSE connection.

        Args:
            scan_id: Scan ID to connect to
            client_id: Optional client identifier (auto-generated if not provided)

        Returns:
            SSEConnection object
        """
        if client_id is None:
            client_id = f"client-{int(time.time() * 1000)}"

        connection = SSEConnection(scan_id=scan_id, client_id=client_id)

        with self._lock:
            if scan_id not in self._connections:
                self._connections[scan_id] = set()
            self._connections[scan_id].add(connection)

        logger.info(f"SSE connection established: scan_id={scan_id}, client_id={client_id}")

        # Send initial connection event
        connection.send_event("connected", {
            "scan_id": scan_id,
            "client_id": client_id,
            "message": "SSE connection established",
        })

        return connection

    def disconnect(self, scan_id: str, client_id: str) -> None:
        """
        Disconnect a client.

        Args:
            scan_id: Scan ID
            client_id: Client identifier
        """
        with self._lock:
            if scan_id in self._connections:
                # Find and remove connection
                self._connections[scan_id] = {
                    conn for conn in self._connections[scan_id]
                    if conn.client_id != client_id
                }

                # Clean up empty scan entries
                if not self._connections[scan_id]:
                    del self._connections[scan_id]

        logger.info(f"SSE connection closed: scan_id={scan_id}, client_id={client_id}")

    def broadcast(self, scan_id: str, event_type: str, data: Dict[str, Any]) -> int:
        """
        Broadcast an event to all connections for a scan.

        Args:
            scan_id: Scan ID
            event_type: Type of event (e.g., 'step_start', 'progress')
            data: Event data payload

        Returns:
            Number of connections that received the event
        """
        count = 0

        with self._lock:
            connections = self._connections.get(scan_id, set())

            for connection in connections:
                try:
                    connection.send_event(event_type, data)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to send event to {connection.client_id}: {e}")

        if count > 0:
            logger.debug(f"Broadcasted {event_type} to {count} client(s) for scan {scan_id}")

        return count

    def get_connections(self, scan_id: str) -> List[SSEConnection]:
        """
        Get all active connections for a scan.

        Args:
            scan_id: Scan ID

        Returns:
            List of SSEConnection objects
        """
        with self._lock:
            return list(self._connections.get(scan_id, set()))

    def get_connection_count(self, scan_id: str) -> int:
        """
        Get number of active connections for a scan.

        Args:
            scan_id: Scan ID

        Returns:
            Number of active connections
        """
        with self._lock:
            return len(self._connections.get(scan_id, set()))

    def cleanup_stale_connections(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up connections older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds (default: 1 hour)

        Returns:
            Number of connections removed
        """
        removed = 0
        current_time = time.time()

        with self._lock:
            for scan_id in list(self._connections.keys()):
                stale = set()

                for conn in self._connections[scan_id]:
                    connected_timestamp = datetime.fromisoformat(conn.connected_at).timestamp()
                    age = current_time - connected_timestamp

                    if age > max_age_seconds:
                        stale.add(conn)

                if stale:
                    self._connections[scan_id] -= stale
                    removed += len(stale)

                    # Clean up empty scan entries
                    if not self._connections[scan_id]:
                        del self._connections[scan_id]

        if removed > 0:
            logger.info(f"Cleaned up {removed} stale SSE connection(s)")

        return removed


def format_sse_message(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format data as SSE message.

    Args:
        event_type: Event type
        data: Event data

    Returns:
        SSE-formatted message string
    """
    lines = []
    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")  # Empty line to end message
    return "\n".join(lines) + "\n"


def format_keepalive() -> str:
    """
    Format SSE keepalive comment.

    Returns:
        SSE keepalive comment
    """
    return f": keepalive {datetime.utcnow().isoformat()}\n\n"
