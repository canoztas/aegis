"""Event broadcasting system for pipeline execution.

Provides a lightweight event system for tracking pipeline progress,
model execution, and findings discovery. Events can be consumed by:
- SSE streams (Phase C)
- WebSocket connections (future)
- Logging systems
- Telemetry collectors
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events emitted during pipeline execution."""

    # Pipeline events
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"

    # Step events
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_SKIPPED = "step_skipped"
    STEP_FAILED = "step_failed"

    # Model events
    MODEL_STARTED = "model_started"
    MODEL_COMPLETED = "model_completed"
    MODEL_FAILED = "model_failed"

    # Finding events
    FINDING_EMITTED = "finding_emitted"
    FINDINGS_MERGED = "findings_merged"

    # Progress events
    PROGRESS_UPDATE = "progress_update"
    CHUNK_STARTED = "chunk_started"
    CHUNK_COMPLETED = "chunk_completed"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Cancellation events
    CANCELLED = "cancelled"


@dataclass
class Event:
    """Base event class."""

    type: EventType
    scan_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.value,
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return Event(
            type=EventType(data["type"]),
            scan_id=data["scan_id"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            data=data.get("data", {}),
        )


class EventBus:
    """Central event bus for publishing and subscribing to events.

    Thread-safe event broadcasting system with support for:
    - Multiple subscribers per event type
    - Wildcard subscriptions (all events)
    - Synchronous and asynchronous delivery (sync only for Phase B)
    """

    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._wildcard_subscribers: List[Callable] = []
        self._event_history: List[Event] = []
        self._max_history = 1000  # Keep last 1000 events

    def subscribe(self, event_type: Optional[EventType], callback: Callable[[Event], None]):
        """
        Subscribe to events.

        Args:
            event_type: Type of event to subscribe to, or None for all events
            callback: Function to call when event is published
        """
        if event_type is None:
            # Wildcard subscription
            self._wildcard_subscribers.append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: Optional[EventType], callback: Callable[[Event], None]):
        """
        Unsubscribe from events.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type is None:
            if callback in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(callback)
        else:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def publish(self, event: Event):
        """
        Publish event to all subscribers.

        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Log event
        logger.debug(f"Event published: {event.type.value} for scan {event.scan_id}")

        # Forward to SSE (if manager is available)
        self._forward_to_sse(event)

        # Notify wildcard subscribers
        for callback in self._wildcard_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in wildcard event callback: {e}")

        # Notify type-specific subscribers
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback for {event.type.value}: {e}")

    def _forward_to_sse(self, event: Event):
        """
        Forward event to SSE connections (if SSE manager is available).

        Args:
            event: Event to forward
        """
        try:
            # Try to import and get SSE manager
            from aegis.sse.stream import SSEManager
            from flask import current_app

            if hasattr(current_app, 'sse_manager'):
                sse_manager: SSEManager = current_app.sse_manager

                # Map event type to SSE event name
                sse_event_type = self._map_event_to_sse(event.type)

                # Broadcast to SSE connections for this scan
                sse_manager.broadcast(
                    scan_id=event.scan_id,
                    event_type=sse_event_type,
                    data=event.data
                )
        except Exception as e:
            # SSE forwarding is optional, don't fail if it's not available
            logger.debug(f"Could not forward event to SSE: {e}")

    def _map_event_to_sse(self, event_type: EventType) -> str:
        """
        Map internal EventType to SSE event names.

        Args:
            event_type: Internal event type

        Returns:
            SSE event name
        """
        # Map EventType enum to SSE event names
        mapping = {
            EventType.PIPELINE_STARTED: "pipeline_started",
            EventType.PIPELINE_COMPLETED: "completed",
            EventType.PIPELINE_FAILED: "error",
            EventType.STEP_STARTED: "step_start",
            EventType.STEP_COMPLETED: "step_completed",
            EventType.STEP_SKIPPED: "step_skipped",
            EventType.STEP_FAILED: "error",
            EventType.MODEL_STARTED: "model_start",
            EventType.MODEL_COMPLETED: "model_completed",
            EventType.MODEL_FAILED: "error",
            EventType.FINDING_EMITTED: "finding",
            EventType.FINDINGS_MERGED: "findings_merged",
            EventType.PROGRESS_UPDATE: "progress",
            EventType.CHUNK_STARTED: "chunk_start",
            EventType.CHUNK_COMPLETED: "chunk_completed",
            EventType.ERROR: "error",
            EventType.WARNING: "warning",
            EventType.CANCELLED: "cancelled",
        }
        # Handle custom events (like "cancelled") that aren't in EventType enum
        return mapping.get(event_type, event_type.value if hasattr(event_type, 'value') else event_type)

    def get_history(self, scan_id: Optional[str] = None, event_type: Optional[EventType] = None) -> List[Event]:
        """
        Get event history.

        Args:
            scan_id: Filter by scan ID (optional)
            event_type: Filter by event type (optional)

        Returns:
            List of events matching filters
        """
        events = self._event_history

        if scan_id:
            events = [e for e in events if e.scan_id == scan_id]

        if event_type:
            events = [e for e in events if e.type == event_type]

        return events

    def clear_history(self, scan_id: Optional[str] = None):
        """
        Clear event history.

        Args:
            scan_id: Clear only events for this scan (optional)
        """
        if scan_id:
            self._event_history = [e for e in self._event_history if e.scan_id != scan_id]
        else:
            self._event_history.clear()


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Get global event bus instance (singleton).

    Returns:
        Global EventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


class EventEmitter:
    """Helper class for emitting events from executors and runners."""

    def __init__(self, scan_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize event emitter.

        Args:
            scan_id: Scan ID for all events
            event_bus: EventBus to use (defaults to global)
        """
        self.scan_id = scan_id
        self.event_bus = event_bus or get_event_bus()

    def emit(self, event_type, data: Optional[Dict[str, Any]] = None):
        """
        Emit an event.

        Args:
            event_type: Type of event (EventType enum or string)
            data: Event data (optional)
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                # If not a valid EventType, log warning and skip
                logger.warning(f"Unknown event type: {event_type}")
                return
        
        event = Event(
            type=event_type,
            scan_id=self.scan_id,
            data=data or {},
        )
        self.event_bus.publish(event)

    def pipeline_started(self, pipeline_name: str, pipeline_version: str):
        """Emit pipeline started event."""
        self.emit(EventType.PIPELINE_STARTED, {
            "pipeline_name": pipeline_name,
            "pipeline_version": pipeline_version,
        })

    def pipeline_completed(self, total_findings: int, duration_ms: int):
        """Emit pipeline completed event."""
        self.emit(EventType.PIPELINE_COMPLETED, {
            "total_findings": total_findings,
            "duration_ms": duration_ms,
        })

    def pipeline_failed(self, error: str):
        """Emit pipeline failed event."""
        self.emit(EventType.PIPELINE_FAILED, {"error": error})

    def step_started(self, step_id: str, step_kind: str):
        """Emit step started event."""
        self.emit(EventType.STEP_STARTED, {
            "step_id": step_id,
            "step_kind": step_kind,
        })

    def step_completed(self, step_id: str, findings_count: int, duration_ms: int):
        """Emit step completed event."""
        self.emit(EventType.STEP_COMPLETED, {
            "step_id": step_id,
            "findings_count": findings_count,
            "duration_ms": duration_ms,
        })

    def step_skipped(self, step_id: str, reason: str):
        """Emit step skipped event."""
        self.emit(EventType.STEP_SKIPPED, {
            "step_id": step_id,
            "reason": reason,
        })

    def step_failed(self, step_id: str, error: str):
        """Emit step failed event."""
        self.emit(EventType.STEP_FAILED, {
            "step_id": step_id,
            "error": error,
        })

    def model_started(self, model_id: str, model_name: str, model_type: Optional[str] = None,
                      device: Optional[str] = None, vram_mb: int = 0, load_time_ms: int = 0,
                      quantization: Optional[str] = None, precision: Optional[str] = None):
        """Emit model started event with detailed telemetry."""
        data = {
            "model_id": model_id,
            "model_name": model_name,
        }

        # Add optional telemetry data
        if model_type:
            data["model_type"] = model_type
        if device:
            data["device"] = device
        if vram_mb > 0:
            data["vram_mb"] = vram_mb
        if load_time_ms > 0:
            data["load_time_ms"] = load_time_ms
        if quantization:
            data["quantization"] = quantization
        if precision:
            data["precision"] = precision

        self.emit(EventType.MODEL_STARTED, data)

    def model_completed(self, model_id: str, findings_count: int, latency_ms: int,
                       input_tokens: int = 0, output_tokens: int = 0, tokens_per_sec: float = 0.0):
        """Emit model completed event with token metrics."""
        data = {
            "model_id": model_id,
            "findings_count": findings_count,
            "latency_ms": latency_ms,
        }

        # Add token metrics if available
        if input_tokens > 0:
            data["input_tokens"] = input_tokens
        if output_tokens > 0:
            data["output_tokens"] = output_tokens
        if tokens_per_sec > 0:
            data["tokens_per_sec"] = tokens_per_sec

        self.emit(EventType.MODEL_COMPLETED, data)

    def model_failed(self, model_id: str, error: str):
        """Emit model failed event."""
        self.emit(EventType.MODEL_FAILED, {
            "model_id": model_id,
            "error": error,
        })

    def finding_emitted(self, finding: Dict[str, Any], model_id: str):
        """Emit finding emitted event."""
        self.emit(EventType.FINDING_EMITTED, {
            "finding": finding,
            "model_id": model_id,
        })

    def findings_merged(self, strategy: str, total_findings: int):
        """Emit findings merged event."""
        self.emit(EventType.FINDINGS_MERGED, {
            "strategy": strategy,
            "total_findings": total_findings,
        })

    def progress_update(self, progress_pct: float, current: int, total: int, message: str):
        """Emit progress update event."""
        self.emit(EventType.PROGRESS_UPDATE, {
            "progress_pct": progress_pct,
            "current": current,
            "total": total,
            "message": message,
        })

    def chunk_started(self, chunk_index: int, total_chunks: int, file_path: str):
        """Emit chunk started event."""
        self.emit(EventType.CHUNK_STARTED, {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "file_path": file_path,
        })

    def chunk_completed(self, chunk_index: int, findings_count: int):
        """Emit chunk completed event."""
        self.emit(EventType.CHUNK_COMPLETED, {
            "chunk_index": chunk_index,
            "findings_count": findings_count,
        })

    def error(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Emit error event."""
        self.emit(EventType.ERROR, {
            "error": error_message,
            "context": context or {},
        })

    def warning(self, warning_message: str, context: Optional[Dict[str, Any]] = None):
        """Emit warning event."""
        self.emit(EventType.WARNING, {
            "warning": warning_message,
            "context": context or {},
        })
