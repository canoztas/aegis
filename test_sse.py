#!/usr/bin/env python3
"""Phase C Week 1 regression tests: SSE Infrastructure."""

import pytest
import time
import threading
from aegis.sse.stream import SSEManager, SSEConnection, format_sse_message, format_keepalive


# ============================================================================
# SSEConnection Tests
# ============================================================================

def test_sse_connection_creation():
    """Test creating an SSE connection."""
    conn = SSEConnection(scan_id="scan-123", client_id="client-1")

    assert conn.scan_id == "scan-123"
    assert conn.client_id == "client-1"
    assert conn.queue is not None
    assert conn.connected_at is not None


def test_sse_connection_send_event():
    """Test sending an event to a connection."""
    conn = SSEConnection(scan_id="scan-123", client_id="client-1")

    conn.send_event("test_event", {"message": "hello"})

    # Get event from queue
    events = conn.get_events(timeout=0.1)
    assert len(events) == 1
    assert events[0]["event"] == "test_event"
    assert events[0]["data"]["message"] == "hello"
    assert "timestamp" in events[0]


def test_sse_connection_get_events_timeout():
    """Test get_events with timeout (no events)."""
    conn = SSEConnection(scan_id="scan-123", client_id="client-1")

    # No events sent, should timeout
    start = time.time()
    events = conn.get_events(timeout=0.5)
    duration = time.time() - start

    assert len(events) == 0
    assert duration >= 0.5  # Should wait at least timeout duration


def test_sse_connection_multiple_events():
    """Test sending multiple events."""
    conn = SSEConnection(scan_id="scan-123", client_id="client-1")

    conn.send_event("event1", {"index": 1})
    conn.send_event("event2", {"index": 2})
    conn.send_event("event3", {"index": 3})

    events = conn.get_events(timeout=0.1)
    assert len(events) == 3
    assert events[0]["data"]["index"] == 1
    assert events[1]["data"]["index"] == 2
    assert events[2]["data"]["index"] == 3


# ============================================================================
# SSEManager Tests
# ============================================================================

def test_sse_manager_connect():
    """Test connecting a client."""
    manager = SSEManager()

    conn = manager.connect(scan_id="scan-123", client_id="client-1")

    assert conn.scan_id == "scan-123"
    assert conn.client_id == "client-1"
    assert manager.get_connection_count("scan-123") == 1


def test_sse_manager_connect_auto_client_id():
    """Test connecting without explicit client ID."""
    manager = SSEManager()

    conn = manager.connect(scan_id="scan-123")

    assert conn.scan_id == "scan-123"
    assert conn.client_id.startswith("client-")
    assert manager.get_connection_count("scan-123") == 1


def test_sse_manager_disconnect():
    """Test disconnecting a client."""
    manager = SSEManager()

    conn = manager.connect(scan_id="scan-123", client_id="client-1")
    assert manager.get_connection_count("scan-123") == 1

    manager.disconnect(scan_id="scan-123", client_id="client-1")
    assert manager.get_connection_count("scan-123") == 0


def test_sse_manager_broadcast():
    """Test broadcasting to all connections."""
    manager = SSEManager()

    conn1 = manager.connect(scan_id="scan-123", client_id="client-1")
    conn2 = manager.connect(scan_id="scan-123", client_id="client-2")

    # Broadcast event
    count = manager.broadcast(
        scan_id="scan-123",
        event_type="progress",
        data={"percent": 50}
    )

    assert count == 2  # Should reach both connections

    # Check both received event
    events1 = conn1.get_events(timeout=0.1)
    events2 = conn2.get_events(timeout=0.1)

    # Skip initial "connected" event
    progress1 = [e for e in events1 if e["event"] == "progress"]
    progress2 = [e for e in events2 if e["event"] == "progress"]

    assert len(progress1) == 1
    assert len(progress2) == 1
    assert progress1[0]["data"]["percent"] == 50
    assert progress2[0]["data"]["percent"] == 50


def test_sse_manager_broadcast_isolation():
    """Test that broadcasts are isolated by scan_id."""
    manager = SSEManager()

    conn1 = manager.connect(scan_id="scan-123", client_id="client-1")
    conn2 = manager.connect(scan_id="scan-456", client_id="client-2")

    # Broadcast to scan-123 only
    count = manager.broadcast(
        scan_id="scan-123",
        event_type="progress",
        data={"percent": 50}
    )

    assert count == 1  # Should only reach conn1

    # Check conn1 received event
    events1 = conn1.get_events(timeout=0.1)
    progress1 = [e for e in events1 if e["event"] == "progress"]
    assert len(progress1) == 1

    # Check conn2 did NOT receive event
    events2 = conn2.get_events(timeout=0.1)
    progress2 = [e for e in events2 if e["event"] == "progress"]
    assert len(progress2) == 0


def test_sse_manager_multiple_scans():
    """Test managing multiple scans simultaneously."""
    manager = SSEManager()

    # Connect to multiple scans
    conn1 = manager.connect(scan_id="scan-123", client_id="client-1")
    conn2 = manager.connect(scan_id="scan-456", client_id="client-2")
    conn3 = manager.connect(scan_id="scan-123", client_id="client-3")

    assert manager.get_connection_count("scan-123") == 2
    assert manager.get_connection_count("scan-456") == 1

    # Broadcast to each scan
    manager.broadcast("scan-123", "test", {"scan": "123"})
    manager.broadcast("scan-456", "test", {"scan": "456"})

    # Verify isolation
    events1 = conn1.get_events(timeout=0.1)
    events2 = conn2.get_events(timeout=0.1)
    events3 = conn3.get_events(timeout=0.1)

    test1 = [e for e in events1 if e["event"] == "test"]
    test2 = [e for e in events2 if e["event"] == "test"]
    test3 = [e for e in events3 if e["event"] == "test"]

    assert test1[0]["data"]["scan"] == "123"
    assert test2[0]["data"]["scan"] == "456"
    assert test3[0]["data"]["scan"] == "123"


def test_sse_manager_cleanup_stale():
    """Test cleaning up stale connections."""
    manager = SSEManager()

    conn1 = manager.connect(scan_id="scan-123", client_id="client-1")

    # Manually set connected_at to old timestamp
    conn1.connected_at = "2020-01-01T00:00:00"

    # Cleanup connections older than 1 hour
    removed = manager.cleanup_stale_connections(max_age_seconds=3600)

    assert removed == 1
    assert manager.get_connection_count("scan-123") == 0


# ============================================================================
# SSE Formatting Tests
# ============================================================================

def test_format_sse_message():
    """Test SSE message formatting."""
    msg = format_sse_message("progress", {"percent": 50})

    assert "event: progress" in msg
    assert '"percent": 50' in msg
    assert msg.endswith("\n\n")  # SSE requires double newline


def test_format_keepalive():
    """Test SSE keepalive comment formatting."""
    msg = format_keepalive()

    assert msg.startswith(": keepalive")
    assert msg.endswith("\n\n")


# ============================================================================
# Concurrent Access Tests
# ============================================================================

def test_sse_manager_thread_safety():
    """Test SSEManager with concurrent connections and broadcasts."""
    manager = SSEManager()
    connections = []
    errors = []

    def connect_client(client_id):
        try:
            conn = manager.connect(scan_id="scan-123", client_id=client_id)
            connections.append(conn)
        except Exception as e:
            errors.append(e)

    def broadcast_event(event_num):
        try:
            manager.broadcast(
                scan_id="scan-123",
                event_type="test",
                data={"event_num": event_num}
            )
        except Exception as e:
            errors.append(e)

    # Connect 5 clients concurrently
    threads = []
    for i in range(5):
        t = threading.Thread(target=connect_client, args=(f"client-{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Broadcast 10 events concurrently
    threads = []
    for i in range(10):
        t = threading.Thread(target=broadcast_event, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(connections) == 5
    assert manager.get_connection_count("scan-123") == 5


if __name__ == "__main__":
    print("=" * 70)
    print("Phase C Week 1 Regression Tests: SSE Infrastructure")
    print("=" * 70)

    tests = [
        ("SSEConnection: creation", test_sse_connection_creation),
        ("SSEConnection: send_event", test_sse_connection_send_event),
        ("SSEConnection: get_events timeout", test_sse_connection_get_events_timeout),
        ("SSEConnection: multiple events", test_sse_connection_multiple_events),
        ("SSEManager: connect", test_sse_manager_connect),
        ("SSEManager: auto client_id", test_sse_manager_connect_auto_client_id),
        ("SSEManager: disconnect", test_sse_manager_disconnect),
        ("SSEManager: broadcast", test_sse_manager_broadcast),
        ("SSEManager: broadcast isolation", test_sse_manager_broadcast_isolation),
        ("SSEManager: multiple scans", test_sse_manager_multiple_scans),
        ("SSEManager: cleanup stale", test_sse_manager_cleanup_stale),
        ("SSE Formatting: message", test_format_sse_message),
        ("SSE Formatting: keepalive", test_format_keepalive),
        ("SSEManager: thread safety", test_sse_manager_thread_safety),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"[OK] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        exit(1)
    else:
        print("\n[OK] All tests passed!")
