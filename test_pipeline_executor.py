#!/usr/bin/env python3
"""Test pipeline executor."""

from aegis.pipeline import PipelineLoader, PipelineExecutor
from aegis.events import get_event_bus, EventType


def test_classic_pipeline():
    """Test executing the classic pipeline."""
    print("[TEST] Loading classic pipeline...")
    loader = PipelineLoader()

    try:
        classic = loader.load_preset("classic")
        print(f"  [OK] Loaded pipeline: {classic.name}")
        print(f"  [OK] Steps: {len(classic.steps)}")

        # Validate pipeline
        warnings = loader.validate_pipeline(classic)
        if warnings:
            print(f"  [WARN] Validation warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        else:
            print("  [OK] No validation warnings")

    except Exception as e:
        print(f"  [ERR] Failed to load classic pipeline: {e}")
        return False

    return True


def test_event_system():
    """Test event system."""
    print("\n[TEST] Testing event system...")

    from aegis.events import EventEmitter

    # Create emitter
    emitter = EventEmitter("test-scan-id")

    # Track events
    events_received = []

    def event_handler(event):
        events_received.append(event)

    # Subscribe to all events
    event_bus = get_event_bus()
    event_bus.subscribe(None, event_handler)

    # Emit some events
    emitter.pipeline_started("test_pipeline", "1.0")
    emitter.step_started("scan", "role")
    emitter.model_started("ollama:test", "Test Model")
    emitter.model_completed("ollama:test", 5, 1000)
    emitter.step_completed("scan", 5, 2000)
    emitter.pipeline_completed(5, 2000)

    # Check we received events
    if len(events_received) == 6:
        print(f"  [OK] Received {len(events_received)} events")
        for event in events_received:
            print(f"    - {event.type.value}: {event.data}")
        return True
    else:
        print(f"  [ERR] Expected 6 events, got {len(events_received)}")
        return False


def test_pipeline_execution_simulation():
    """Test pipeline execution with mock data."""
    print("\n[TEST] Testing pipeline executor (simulation)...")

    try:
        loader = PipelineLoader()
        executor = PipelineExecutor()

        # Load fast_scan (simpler, faster)
        pipeline = loader.load_preset("fast_scan")
        print(f"  [OK] Loaded pipeline: {pipeline.name}")

        # Mock source files
        source_files = {
            "test.py": """
# Simple test file
def get_user(user_id):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

def unsafe_eval(code):
    # Dangerous eval
    return eval(code)
"""
        }

        print("  [OK] Created mock source files")
        print(f"  [INFO] Pipeline has {len(pipeline.steps)} steps")

        # Note: Full execution requires Ollama models to be running
        # This is just a structure test
        print("  [OK] Pipeline executor initialized")
        print("  [INFO] Full execution test requires running Ollama models")

        return True

    except Exception as e:
        print(f"  [ERR] Pipeline execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_presets_loadable():
    """Test that all preset pipelines load successfully."""
    print("\n[TEST] Loading all preset pipelines...")

    loader = PipelineLoader()
    presets = loader.list_presets()

    print(f"  [INFO] Found {len(presets)} presets: {presets}")

    all_ok = True
    for preset_name in presets:
        try:
            pipeline = loader.load_preset(preset_name)
            print(f"  [OK] {preset_name}: {len(pipeline.steps)} steps")
        except Exception as e:
            print(f"  [ERR] {preset_name}: {e}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Executor Tests")
    print("=" * 60)

    results = []

    # Test 1: Load classic pipeline
    results.append(("Classic pipeline loading", test_classic_pipeline()))

    # Test 2: Event system
    results.append(("Event system", test_event_system()))

    # Test 3: All presets loadable
    results.append(("All presets loadable", test_all_presets_loadable()))

    # Test 4: Pipeline execution simulation
    results.append(("Pipeline execution simulation", test_pipeline_execution_simulation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[ERR] {failed} test(s) failed")
        exit(1)
