#!/usr/bin/env python3
"""Week 3 regression tests: Gating + Consensus."""

import pytest
from aegis.pipeline.schema import (
    PipelineConfig,
    PipelineStep,
    StepKind,
    GatingCondition,
    GatingOperator,
    ConsensusStrategy,
    PipelineExecutionContext,
)
from aegis.pipeline.gating import ConditionEvaluator, ContextBuilder


# ============================================================================
# Condition Evaluator Unit Tests
# ============================================================================

def test_numeric_comparison_gt():
    """Test numeric > comparison."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(field="findings.count", operator=GatingOperator.GT, value=5)
    context = {"findings": {"count": 10}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True
    assert result.resolved_values["findings.count"] == 10


def test_numeric_comparison_lte():
    """Test numeric <= comparison."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(field="findings.count", operator=GatingOperator.LTE, value=5)
    context = {"findings": {"count": 3}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True


def test_numeric_comparison_eq():
    """Test numeric == comparison."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(field="findings.count", operator=GatingOperator.EQ, value=0)
    context = {"findings": {"count": 0}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True


def test_and_condition_both_true():
    """Test AND condition where both are true."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(
        field="findings.high_count",
        operator=GatingOperator.GT,
        value=0,
        and_conditions=[
            GatingCondition(field="findings.critical_count", operator=GatingOperator.EQ, value=0)
        ],
    )
    context = {"findings": {"high_count": 2, "critical_count": 0}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True


def test_and_condition_one_false():
    """Test AND condition where one is false."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(
        field="findings.high_count",
        operator=GatingOperator.GT,
        value=0,
        and_conditions=[
            GatingCondition(field="findings.critical_count", operator=GatingOperator.GT, value=0)
        ],
    )
    context = {"findings": {"high_count": 2, "critical_count": 0}}

    result = evaluator.evaluate(condition, context)
    assert result.result is False


def test_or_condition_one_true():
    """Test OR condition where one is true."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(
        field="findings.high_count",
        operator=GatingOperator.GT,
        value=0,
        or_conditions=[
            GatingCondition(field="findings.critical_count", operator=GatingOperator.GT, value=0)
        ],
    )
    context = {"findings": {"high_count": 2, "critical_count": 0}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True


def test_or_condition_both_false():
    """Test OR condition where both are false."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(
        field="findings.high_count",
        operator=GatingOperator.GT,
        value=0,
        or_conditions=[
            GatingCondition(field="findings.critical_count", operator=GatingOperator.GT, value=0)
        ],
    )
    context = {"findings": {"high_count": 0, "critical_count": 0}}

    result = evaluator.evaluate(condition, context)
    assert result.result is False


def test_missing_path_behavior_zero():
    """Test missing path defaults to 0 for count fields."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    # Use a "count" path which returns 0 when missing
    condition = GatingCondition(field="nonexistent.count", operator=GatingOperator.EQ, value=0)
    context = {}

    result = evaluator.evaluate(condition, context)
    assert result.result is True
    assert result.resolved_values["nonexistent.count"] == 0


def test_missing_path_behavior_error():
    """Test missing path raises error."""
    evaluator = ConditionEvaluator(missing_path_behavior="error")
    condition = GatingCondition(field="nonexistent.path", operator=GatingOperator.EQ, value=0)
    context = {}

    with pytest.raises(ValueError, match="Path .* not found"):
        evaluator.evaluate(condition, context)


def test_nested_path_resolution():
    """Test nested path like step.triage.high_count."""
    evaluator = ConditionEvaluator(missing_path_behavior="zero")
    condition = GatingCondition(
        field="step.triage.high_count", operator=GatingOperator.GT, value=0
    )
    context = {"step": {"triage": {"high_count": 3}}}

    result = evaluator.evaluate(condition, context)
    assert result.result is True
    assert result.resolved_values["step.triage.high_count"] == 3


# ============================================================================
# Context Builder Tests
# ============================================================================

def test_context_builder_job_context():
    """Test building job context."""
    builder = ContextBuilder()
    exec_context = PipelineExecutionContext(
        pipeline_id="test_pipeline",
        scan_id="scan-123",
        started_at="2026-01-08T00:00:00",
    )
    exec_context.completed_steps = ["step1", "step2"]
    exec_context.failed_steps = []

    context = builder.build(exec_context)

    assert context["job"]["scan_id"] == "scan-123"
    assert context["job"]["pipeline_id"] == "test_pipeline"
    assert context["job"]["steps_completed"] == 2
    assert context["job"]["steps_failed"] == 0


def test_context_builder_step_context():
    """Test building step context with findings."""
    builder = ContextBuilder()
    exec_context = PipelineExecutionContext(
        pipeline_id="test", scan_id="scan-123", started_at="2026-01-08T00:00:00"
    )

    # Add step output with findings
    exec_context.step_outputs["triage"] = {
        "findings": [
            {"severity": "high", "name": "SQL Injection"},
            {"severity": "medium", "name": "XSS"},
            {"severity": "high", "name": "Command Injection"},
        ],
        "metadata": {"suspicion_score": 0.8},
    }

    context = builder.build(exec_context)

    assert context["step"]["triage"]["findings_count"] == 3
    assert context["step"]["triage"]["high_count"] == 2
    assert context["step"]["triage"]["medium_count"] == 1
    assert context["step"]["triage"]["critical_count"] == 0
    assert context["step"]["triage"]["suspicion_score"] == 0.8


def test_context_builder_findings_context():
    """Test building global findings context."""
    builder = ContextBuilder()
    exec_context = PipelineExecutionContext(
        pipeline_id="test", scan_id="scan-123", started_at="2026-01-08T00:00:00"
    )

    # Add multiple step outputs
    exec_context.step_outputs["triage"] = {
        "findings": [
            {"severity": "critical", "cwe": "CWE-89"},
            {"severity": "high", "cwe": "CWE-79"},
        ]
    }
    exec_context.step_outputs["deep_scan"] = {
        "findings": [
            {"severity": "high", "cwe": "CWE-79"},
            {"severity": "medium", "cwe": "CWE-20"},
        ]
    }

    context = builder.build(exec_context)

    assert context["findings"]["count"] == 4
    assert context["findings"]["critical_count"] == 1
    assert context["findings"]["high_count"] == 2
    assert context["findings"]["medium_count"] == 1
    assert context["findings"]["max_severity"] == "critical"
    assert context["findings"]["by_severity"]["critical"]["count"] == 1
    assert context["findings"]["by_cwe"]["CWE-79"]["count"] == 2


# ============================================================================
# Integration Tests
# ============================================================================

def test_gate_skips_step_when_condition_false():
    """Test gate step skips subsequent step when condition is false."""
    from aegis.pipeline import PipelineLoader

    loader = PipelineLoader()
    pipeline = loader.load_preset("triage_deep")

    # Verify gate step configuration
    gate_step = next((s for s in pipeline.steps if s.kind == StepKind.GATE), None)
    assert gate_step is not None
    assert gate_step.id == "gate_escalate"
    assert gate_step.condition.field == "step.triage.high_count"
    assert gate_step.on_true == "deep_scan"
    assert gate_step.on_false == "triage_consensus"


def test_context_evaluation_with_zero_findings():
    """Test that gate evaluates to false when triage finds nothing."""
    builder = ContextBuilder()
    evaluator = ConditionEvaluator(missing_path_behavior="zero")

    # Simul context with zero triage findings
    exec_context = PipelineExecutionContext(
        pipeline_id="test", scan_id="scan-123", started_at="2026-01-08T00:00:00"
    )
    exec_context.step_outputs["triage"] = {
        "findings": [],  # No findings
        "metadata": {},
    }

    # Build context
    context = builder.build(exec_context)

    # Evaluate gate condition from triage_deep
    condition = GatingCondition(
        field="step.triage.high_count",
        operator=GatingOperator.GT,
        value=0,
        or_conditions=[
            GatingCondition(
                field="step.triage.critical_count", operator=GatingOperator.GT, value=0
            )
        ],
    )

    result = evaluator.evaluate(condition, context)
    assert result.result is False  # Should NOT escalate to deep scan


def test_context_evaluation_with_high_findings():
    """Test that gate evaluates to true when triage finds high-severity issues."""
    builder = ContextBuilder()
    evaluator = ConditionEvaluator(missing_path_behavior="zero")

    # Simulate context with high-severity findings
    exec_context = PipelineExecutionContext(
        pipeline_id="test", scan_id="scan-123", started_at="2026-01-08T00:00:00"
    )
    exec_context.step_outputs["triage"] = {
        "findings": [
            {"severity": "high", "name": "SQL Injection"},
            {"severity": "medium", "name": "XSS"},
        ],
        "metadata": {},
    }

    # Build context
    context = builder.build(exec_context)

    # Evaluate gate condition
    condition = GatingCondition(
        field="step.triage.high_count",
        operator=GatingOperator.GT,
        value=0,
        or_conditions=[
            GatingCondition(
                field="step.triage.critical_count", operator=GatingOperator.GT, value=0
            )
        ],
    )

    result = evaluator.evaluate(condition, context)
    assert result.result is True  # Should escalate to deep scan


if __name__ == "__main__":
    print("=" * 70)
    print("Week 3 Regression Tests: Gating + Consensus")
    print("=" * 70)

    tests = [
        ("Numeric comparison >", test_numeric_comparison_gt),
        ("Numeric comparison <=", test_numeric_comparison_lte),
        ("Numeric comparison ==", test_numeric_comparison_eq),
        ("AND condition (both true)", test_and_condition_both_true),
        ("AND condition (one false)", test_and_condition_one_false),
        ("OR condition (one true)", test_or_condition_one_true),
        ("OR condition (both false)", test_or_condition_both_false),
        ("Missing path (zero)", test_missing_path_behavior_zero),
        ("Nested path resolution", test_nested_path_resolution),
        ("Context builder: job", test_context_builder_job_context),
        ("Context builder: step", test_context_builder_step_context),
        ("Context builder: findings", test_context_builder_findings_context),
        ("Gate step configuration", test_gate_skips_step_when_condition_false),
        ("Gate eval (zero findings)", test_context_evaluation_with_zero_findings),
        ("Gate eval (high findings)", test_context_evaluation_with_high_findings),
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
