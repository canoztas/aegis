"""Gating logic for conditional pipeline execution.

Provides:
- ConditionEvaluator: Evaluate gating conditions
- ContextBuilder: Build evaluation context from step outputs
- Field path resolution for deterministic condition evaluation
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from aegis.pipeline.schema import GatingCondition, GatingOperator, PipelineExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of condition evaluation."""
    result: bool  # True if condition met
    resolved_values: Dict[str, Any]  # Field paths resolved during evaluation
    debug_info: Optional[str] = None  # Human-readable explanation


class ContextBuilder:
    """Build evaluation context from pipeline execution state.

    Context structure:
    {
        "job": {
            "scan_id": "...",
            "pipeline_id": "...",
            "started_at": "...",
        },
        "step": {
            "<step_id>": {
                "findings_count": 5,
                "high_count": 2,
                "critical_count": 1,
                "suspicion_score": 0.8,
                ...
            }
        },
        "findings": {
            "count": 10,
            "high_count": 3,
            "critical_count": 1,
            "max_severity": "high",
            "by_severity": {
                "critical": {"count": 1},
                "high": {"count": 3},
                "medium": {"count": 4},
                "low": {"count": 2},
            },
            "by_cwe": {
                "CWE-79": {"count": 2},
                "CWE-89": {"count": 1},
                ...
            },
            "agreement_score": 0.75,
        },
        "models": {
            "<role>": {
                "id": "model_id",
                "count": 2,
            }
        }
    }
    """

    def build(self, context: PipelineExecutionContext) -> Dict[str, Any]:
        """
        Build evaluation context from execution context.

        Args:
            context: Pipeline execution context

        Returns:
            Context dict for evaluation
        """
        eval_context = {
            "job": self._build_job_context(context),
            "step": self._build_step_context(context),
            "findings": self._build_findings_context(context),
            "models": {},  # Placeholder for future
        }

        return eval_context

    def _build_job_context(self, context: PipelineExecutionContext) -> Dict[str, Any]:
        """Build job metadata context."""
        return {
            "scan_id": context.scan_id,
            "pipeline_id": context.pipeline_id,
            "started_at": context.started_at,
            "completed_at": context.completed_at,
            "steps_completed": len(context.completed_steps),
            "steps_failed": len(context.failed_steps),
        }

    def _build_step_context(self, context: PipelineExecutionContext) -> Dict[str, Any]:
        """Build per-step output context."""
        step_context = {}

        for step_id, output in context.step_outputs.items():
            findings = output.get("findings", [])
            metadata = output.get("metadata", {})

            # Aggregate findings by severity
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for finding in findings:
                severity = finding.get("severity", "medium").lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1

            step_context[step_id] = {
                "findings_count": len(findings),
                "critical_count": severity_counts["critical"],
                "high_count": severity_counts["high"],
                "medium_count": severity_counts["medium"],
                "low_count": severity_counts["low"],
                "suspicion_score": metadata.get("suspicion_score", 0.0),
                "metadata": metadata,
            }

        return step_context

    def _build_findings_context(self, context: PipelineExecutionContext) -> Dict[str, Any]:
        """Build global findings aggregate context."""
        # Collect all findings from all steps
        all_findings = []
        for output in context.step_outputs.values():
            all_findings.extend(output.get("findings", []))

        # Aggregate by severity
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_cwe = {}
        max_severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        max_severity = "info"

        for finding in all_findings:
            severity = finding.get("severity", "medium").lower()
            if severity in by_severity:
                by_severity[severity] += 1

            # Track max severity
            if max_severity_order.get(severity, 0) > max_severity_order[max_severity]:
                max_severity = severity

            # Aggregate by CWE
            cwe = finding.get("cwe", "Unknown")
            if cwe not in by_cwe:
                by_cwe[cwe] = 0
            by_cwe[cwe] += 1

        # Calculate agreement score (placeholder - requires per-model data)
        agreement_score = 1.0  # Default to full agreement

        return {
            "count": len(all_findings),
            "critical_count": by_severity["critical"],
            "high_count": by_severity["high"],
            "medium_count": by_severity["medium"],
            "low_count": by_severity["low"],
            "max_severity": max_severity,
            "by_severity": {k: {"count": v} for k, v in by_severity.items()},
            "by_cwe": {k: {"count": v} for k, v in by_cwe.items()},
            "agreement_score": agreement_score,
        }


class ConditionEvaluator:
    """Evaluate gating conditions against pipeline context.

    Supports:
    - Numeric comparisons: >, >=, <, <=, ==, !=
    - Boolean operators: and, or (via and_conditions, or_conditions)
    - Field path resolution: findings.count, step.triage.high_count, etc.
    - Missing path behavior: Treat as None/0 (configurable)
    """

    def __init__(self, missing_path_behavior: str = "zero"):
        """
        Initialize condition evaluator.

        Args:
            missing_path_behavior: How to handle missing paths
                - "zero": Treat as 0/None/False (default)
                - "error": Raise ValueError
        """
        self.missing_path_behavior = missing_path_behavior

    def evaluate(
        self,
        condition: GatingCondition,
        context: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Evaluate a gating condition.

        Args:
            condition: Condition to evaluate
            context: Evaluation context (from ContextBuilder)

        Returns:
            EvaluationResult with result and resolved values
        """
        resolved_values = {}

        # Resolve field value
        field_value = self._resolve_path(condition.field, context)
        resolved_values[condition.field] = field_value

        # Evaluate base condition
        base_result = self._compare(field_value, condition.operator, condition.value)

        # Evaluate AND conditions (all must be true, including base)
        if condition.and_conditions:
            if not base_result:
                return EvaluationResult(
                    result=False,
                    resolved_values=resolved_values,
                    debug_info=f"Base condition failed: {condition.field} {condition.operator.value} {condition.value}",
                )

            for and_cond in condition.and_conditions:
                and_result = self.evaluate(and_cond, context)
                resolved_values.update(and_result.resolved_values)
                if not and_result.result:
                    return EvaluationResult(
                        result=False,
                        resolved_values=resolved_values,
                        debug_info=f"AND condition failed: {and_cond.field} {and_cond.operator.value} {and_cond.value}",
                    )

            # All AND conditions passed
            return EvaluationResult(
                result=True,
                resolved_values=resolved_values,
                debug_info=f"All AND conditions met",
            )

        # Evaluate OR conditions (base OR any or_condition must be true)
        if condition.or_conditions:
            if base_result:
                return EvaluationResult(
                    result=True,
                    resolved_values=resolved_values,
                    debug_info=f"Base condition met (OR)",
                )

            # Check OR conditions
            for or_cond in condition.or_conditions:
                or_result = self.evaluate(or_cond, context)
                resolved_values.update(or_result.resolved_values)
                if or_result.result:
                    return EvaluationResult(
                        result=True,
                        resolved_values=resolved_values,
                        debug_info=f"OR condition met: {or_cond.field}",
                    )

            # No OR conditions met
            return EvaluationResult(
                result=False,
                resolved_values=resolved_values,
                debug_info=f"No OR conditions met",
            )

        # No AND/OR, just base condition
        return EvaluationResult(
            result=base_result,
            resolved_values=resolved_values,
            debug_info=f"{condition.field}={field_value} {condition.operator.value} {condition.value} → {base_result}",
        )

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a field path in context.

        Examples:
            "findings.count" → context["findings"]["count"]
            "step.triage.high_count" → context["step"]["triage"]["high_count"]
            "job.scan_id" → context["job"]["scan_id"]

        Args:
            path: Dot-separated field path
            context: Evaluation context

        Returns:
            Resolved value or None/0 if missing
        """
        parts = path.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Missing path
                if self.missing_path_behavior == "error":
                    raise ValueError(f"Path '{path}' not found in context (missing at '{part}')")
                else:
                    # Default to 0 for numeric contexts, None otherwise
                    logger.debug(f"Path '{path}' not found, returning 0/None")
                    return 0 if "count" in path else None

        return current

    def _compare(self, left: Any, operator: GatingOperator, right: Any) -> bool:
        """
        Compare values using operator.

        Args:
            left: Left value (resolved from context)
            right: Right value (from condition)
            operator: Comparison operator

        Returns:
            True if comparison succeeds
        """
        # Handle None values
        if left is None:
            left = 0

        try:
            if operator == GatingOperator.GT:
                return left > right
            elif operator == GatingOperator.GTE:
                return left >= right
            elif operator == GatingOperator.LT:
                return left < right
            elif operator == GatingOperator.LTE:
                return left <= right
            elif operator == GatingOperator.EQ:
                return left == right
            elif operator == GatingOperator.NEQ:
                return left != right
            elif operator == GatingOperator.IN:
                return left in right
            elif operator == GatingOperator.NOT_IN:
                return left not in right
            else:
                raise ValueError(f"Unknown operator: {operator}")
        except TypeError as e:
            logger.error(f"Type error comparing {left} {operator.value} {right}: {e}")
            return False
