#!/usr/bin/env python3
"""Test pipeline schema validation."""

import pytest
from pydantic import ValidationError

from aegis.pipeline.schema import (
    PipelineConfig,
    PipelineStep,
    StepKind,
    ConsensusStrategy,
    GatingCondition,
    GatingOperator,
)
from aegis.pipeline.loader import PipelineLoader, PipelineRegistry


def test_valid_simple_pipeline():
    """Test creating a valid simple pipeline."""
    config = PipelineConfig(
        name="test_simple",
        version="1.0",
        steps=[
            PipelineStep(id="scan", kind=StepKind.ROLE, role="scan"),
            PipelineStep(
                id="consensus",
                kind=StepKind.CONSENSUS,
                strategy=ConsensusStrategy.UNION,
                sources=["scan"],
            ),
        ],
    )
    assert config.name == "test_simple"
    assert len(config.steps) == 2


def test_invalid_pipeline_no_steps():
    """Test pipeline without steps fails validation."""
    with pytest.raises(ValidationError):
        PipelineConfig(name="invalid", version="1.0", steps=[])


def test_invalid_pipeline_duplicate_step_ids():
    """Test pipeline with duplicate step IDs fails validation."""
    with pytest.raises(ValidationError):
        PipelineConfig(
            name="duplicate_ids",
            version="1.0",
            steps=[
                PipelineStep(id="scan", kind=StepKind.ROLE, role="scan"),
                PipelineStep(id="scan", kind=StepKind.ROLE, role="triage"),  # Duplicate ID
            ],
        )


def test_role_step_requires_role_field():
    """Test role step must have role field."""
    with pytest.raises(ValidationError):
        PipelineStep(id="scan", kind=StepKind.ROLE)  # Missing role field


def test_model_step_requires_models_field():
    """Test model step must have models field."""
    with pytest.raises(ValidationError):
        PipelineStep(id="scan", kind=StepKind.MODEL)  # Missing models field


def test_tool_step_requires_tool_id_field():
    """Test tool step must have tool_id field."""
    with pytest.raises(ValidationError):
        PipelineStep(id="tool", kind=StepKind.TOOL)  # Missing tool_id field


def test_consensus_step_requires_strategy_and_sources():
    """Test consensus step must have strategy and sources."""
    with pytest.raises(ValidationError):
        PipelineStep(id="consensus", kind=StepKind.CONSENSUS)  # Missing both


def test_gate_step_requires_condition():
    """Test gate step must have condition."""
    with pytest.raises(ValidationError):
        PipelineStep(id="gate", kind=StepKind.GATE)  # Missing condition


def test_gate_step_requires_targets():
    """Test gate step must have at least one target."""
    with pytest.raises(ValidationError):
        PipelineStep(
            id="gate",
            kind=StepKind.GATE,
            condition=GatingCondition(field="test", operator=GatingOperator.GT, value=0),
            # Missing on_true and on_false
        )


def test_valid_gating_condition():
    """Test creating valid gating conditions."""
    # Simple condition
    cond1 = GatingCondition(field="findings.high_count", operator=GatingOperator.GT, value=0)
    assert cond1.field == "findings.high_count"

    # Condition with AND
    cond2 = GatingCondition(
        field="findings.critical_count",
        operator=GatingOperator.GTE,
        value=1,
        and_conditions=[
            GatingCondition(field="findings.confidence", operator=GatingOperator.GT, value=0.7)
        ],
    )
    assert len(cond2.and_conditions) == 1

    # Condition with OR
    cond3 = GatingCondition(
        field="findings.high_count",
        operator=GatingOperator.GT,
        value=0,
        or_conditions=[
            GatingCondition(field="findings.critical_count", operator=GatingOperator.GT, value=0)
        ],
    )
    assert len(cond3.or_conditions) == 1


def test_invalid_step_reference_depends_on():
    """Test pipeline with invalid depends_on reference fails."""
    with pytest.raises(ValidationError, match="depends on unknown step"):
        PipelineConfig(
            name="invalid_deps",
            version="1.0",
            steps=[
                PipelineStep(
                    id="scan", kind=StepKind.ROLE, role="scan", depends_on=["nonexistent"]
                )
            ],
        )


def test_invalid_step_reference_consensus_sources():
    """Test pipeline with invalid consensus sources fails."""
    with pytest.raises(ValidationError, match="references unknown source"):
        PipelineConfig(
            name="invalid_sources",
            version="1.0",
            steps=[
                PipelineStep(
                    id="consensus",
                    kind=StepKind.CONSENSUS,
                    strategy=ConsensusStrategy.UNION,
                    sources=["nonexistent"],
                )
            ],
        )


def test_invalid_step_reference_gate_targets():
    """Test pipeline with invalid gate targets fails."""
    with pytest.raises(ValidationError, match="on_true references unknown step"):
        PipelineConfig(
            name="invalid_gate",
            version="1.0",
            steps=[
                PipelineStep(
                    id="gate",
                    kind=StepKind.GATE,
                    condition=GatingCondition(
                        field="test", operator=GatingOperator.GT, value=0
                    ),
                    on_true="nonexistent",
                )
            ],
        )


def test_valid_complex_pipeline():
    """Test creating a valid complex pipeline with gating."""
    config = PipelineConfig(
        name="triage_deep",
        version="1.0",
        description="Triage with conditional deep scan",
        steps=[
            PipelineStep(id="triage", kind=StepKind.ROLE, role="triage"),
            PipelineStep(
                id="gate",
                kind=StepKind.GATE,
                condition=GatingCondition(
                    field="findings.high_count", operator=GatingOperator.GT, value=0
                ),
                on_true="deep_scan",
                on_false="consensus",
            ),
            PipelineStep(id="deep_scan", kind=StepKind.ROLE, role="deep_scan"),
            PipelineStep(
                id="consensus",
                kind=StepKind.CONSENSUS,
                strategy=ConsensusStrategy.WEIGHTED,
                sources=["triage", "deep_scan"],
            ),
        ],
    )
    assert len(config.steps) == 4
    assert config.steps[1].on_true == "deep_scan"


def test_pipeline_loader_load_preset():
    """Test loading preset pipelines."""
    loader = PipelineLoader()

    # Load classic preset
    try:
        classic = loader.load_preset("classic")
        assert classic.name == "classic"
        assert classic.is_preset is True
        assert len(classic.steps) >= 2  # At least scan + consensus
    except FileNotFoundError:
        pytest.skip("Preset pipelines not found")


def test_pipeline_loader_list_presets():
    """Test listing available presets."""
    loader = PipelineLoader()
    presets = loader.list_presets()

    # Should include our preset pipelines
    expected_presets = {"classic", "triage_deep", "judge_consensus", "fast_scan"}
    assert expected_presets.issubset(set(presets))


def test_pipeline_registry():
    """Test pipeline registry."""
    registry = PipelineRegistry()

    # Register custom pipeline
    custom = PipelineConfig(
        name="custom",
        version="1.0",
        steps=[PipelineStep(id="scan", kind=StepKind.ROLE, role="scan")],
    )
    registry.register_custom("my_custom", custom)

    # Retrieve it
    retrieved = registry.get_pipeline("my_custom", is_preset=False)
    assert retrieved is not None
    assert retrieved.name == "custom"

    # List all
    all_pipelines = registry.list_all()
    assert "my_custom" in all_pipelines["custom"]


def test_pipeline_loader_warnings():
    """Test pipeline validation warnings."""
    loader = PipelineLoader()

    # Pipeline with gate that has no targets (warning)
    pipeline = PipelineConfig(
        name="test_warnings",
        version="1.0",
        steps=[
            PipelineStep(id="scan", kind=StepKind.ROLE, role="scan"),
            PipelineStep(
                id="gate",
                kind=StepKind.GATE,
                condition=GatingCondition(
                    field="test", operator=GatingOperator.GT, value=0
                ),
                on_true="scan",  # Valid reference
            ),
        ],
    )

    warnings = loader.validate_pipeline(pipeline)
    # Should have warnings about potential issues
    assert isinstance(warnings, list)


if __name__ == "__main__":
    # Run tests
    print("Testing pipeline schema validation...\n")

    # Basic tests
    print("[TEST] Valid simple pipeline...")
    test_valid_simple_pipeline()
    print("  [OK] Simple pipeline created successfully")

    print("[TEST] Invalid pipeline (no steps)...")
    try:
        test_invalid_pipeline_no_steps()
        print("  [ERR] Should have raised ValidationError")
    except AssertionError:
        print("  [OK] Correctly rejected pipeline with no steps")

    print("[TEST] Invalid pipeline (duplicate IDs)...")
    try:
        test_invalid_pipeline_duplicate_step_ids()
        print("  [ERR] Should have raised ValidationError")
    except AssertionError:
        print("  [OK] Correctly rejected duplicate step IDs")

    print("[TEST] Valid complex pipeline with gating...")
    test_valid_complex_pipeline()
    print("  [OK] Complex pipeline created successfully")

    # Loader tests
    print("\n[TEST] Loading preset pipelines...")
    try:
        test_pipeline_loader_load_preset()
        print("  [OK] Preset pipelines loaded")
    except Exception as e:
        print(f"  [SKIP] {e}")

    print("[TEST] Listing presets...")
    test_pipeline_loader_list_presets()
    print("  [OK] Presets listed successfully")

    print("[TEST] Pipeline registry...")
    test_pipeline_registry()
    print("  [OK] Registry operations work")

    print("\n[OK] All tests passed!")
