"""Pipeline executor for step-by-step workflow execution.

Executes pipelines defined in PipelineConfig, emitting events for progress tracking.
Linear execution for Phase B (DAG support in future phases).
"""

import time
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime

from aegis.pipeline.schema import (
    PipelineConfig,
    PipelineStep,
    StepKind,
    ConsensusStrategy,
    PipelineExecutionContext,
)
from aegis.models.registry import ModelRegistryV2
from aegis.models.schema import ModelRole, ModelType, FindingCandidate, ParserResult
from aegis.prompt_builder import PromptBuilder
from aegis.consensus.engine import ConsensusEngine
from aegis.data_models import ModelResponse, Finding
from aegis.events import EventEmitter
from aegis.models.engine import ModelExecutionEngine
from aegis.utils import chunk_file_lines


class PipelineExecutor:
    """Execute pipelines step-by-step with event broadcasting."""

    def __init__(
        self,
        registry: Optional[ModelRegistryV2] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        consensus_engine: Optional[ConsensusEngine] = None,
        max_workers: int = 4,
    ):
        """
        Initialize pipeline executor.

        Args:
            registry: Model registry for resolving models by role
            prompt_builder: Prompt builder for creating model prompts
            consensus_engine: Consensus engine for merging findings
            max_workers: Maximum parallel workers for chunk processing
        """
        self.registry = registry or ModelRegistryV2()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.consensus_engine = consensus_engine or ConsensusEngine(self.prompt_builder)
        self.max_workers = max_workers
        self.execution_engine = ModelExecutionEngine(self.registry)

    def execute(
        self,
        pipeline: PipelineConfig,
        source_files: Dict[str, str],
        scan_id: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        chunk_size: int = 1000,
    ) -> PipelineExecutionContext:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline configuration to execute
            source_files: Dict of file_path -> content
            scan_id: Optional scan ID (generated if not provided)
            language_hints: Optional language hints for detection
            chunk_size: Lines per chunk for processing

        Returns:
            PipelineExecutionContext with results
        """
        scan_id = scan_id or str(uuid.uuid4())
        emitter = EventEmitter(scan_id)

        # Initialize execution context
        context = PipelineExecutionContext(
            pipeline_id=pipeline.name,
            scan_id=scan_id,
            started_at=datetime.utcnow().isoformat(),
        )

        # Emit pipeline started event
        emitter.pipeline_started(pipeline.name, pipeline.version)
        start_time = time.time()

        try:
            # Execute steps with gate-aware branching
            skip_until = None  # Step ID to skip until (for gate branching)

            for step in pipeline.steps:
                # Check if we should skip this step (gate branching)
                if skip_until and step.id != skip_until:
                    emitter.step_skipped(step.id, f"Skipped by gate (branching to {skip_until})")
                    continue
                elif skip_until and step.id == skip_until:
                    skip_until = None  # Resume execution

                if not step.enabled:
                    emitter.step_skipped(step.id, "Step disabled")
                    continue

                # Execute step based on kind
                step_start = time.time()
                emitter.step_started(step.id, step.kind.value)

                try:
                    step_result = self._execute_step(
                        step=step,
                        source_files=source_files,
                        context=context,
                        emitter=emitter,
                        language_hints=language_hints,
                        chunk_size=chunk_size,
                    )

                    # Store step output
                    context.step_outputs[step.id] = step_result
                    context.completed_steps.append(step.id)

                    # Handle gate branching
                    if step.kind == StepKind.GATE:
                        gate_result = step_result.get("metadata", {}).get("result")
                        if gate_result is True and step.on_true:
                            skip_until = step.on_true
                            emitter.emit("gate_branched", {
                                "gate_id": step.id,
                                "result": True,
                                "target": step.on_true,
                            })
                        elif gate_result is False and step.on_false:
                            skip_until = step.on_false
                            emitter.emit("gate_branched", {
                                "gate_id": step.id,
                                "result": False,
                                "target": step.on_false,
                            })

                    # Emit step completed
                    step_duration = int((time.time() - step_start) * 1000)
                    findings_count = len(step_result.get("findings", []))
                    emitter.step_completed(step.id, findings_count, step_duration)

                except Exception as e:
                    context.failed_steps.append(step.id)
                    emitter.step_failed(step.id, str(e))
                    emitter.error(f"Step {step.id} failed", {"error": str(e)})
                    # Continue with next step (don't fail entire pipeline)

            # Calculate total duration
            total_duration = int((time.time() - start_time) * 1000)

            # Get final findings (from last consensus step or last completed step)
            final_findings = self._get_final_findings(context)

            # Emit pipeline completed
            emitter.pipeline_completed(len(final_findings), total_duration)
            context.completed_at = datetime.utcnow().isoformat()

            return context

        except Exception as e:
            emitter.pipeline_failed(str(e))
            emitter.error(f"Pipeline failed: {e}")
            raise

    def _execute_step(
        self,
        step: PipelineStep,
        source_files: Dict[str, str],
        context: PipelineExecutionContext,
        emitter: EventEmitter,
        language_hints: Optional[List[str]] = None,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline step.

        Args:
            step: Step to execute
            source_files: Source files to scan
            context: Execution context
            emitter: Event emitter
            language_hints: Language hints
            chunk_size: Chunk size

        Returns:
            Step output dict with findings and metadata
        """
        context.current_step_id = step.id

        if step.kind == StepKind.ROLE:
            return self._execute_role_step(step, source_files, emitter, language_hints, chunk_size)

        elif step.kind == StepKind.MODEL:
            return self._execute_model_step(step, source_files, emitter, language_hints, chunk_size)

        elif step.kind == StepKind.CONSENSUS:
            return self._execute_consensus_step(step, context, emitter)

        elif step.kind == StepKind.GATE:
            return self._execute_gate_step(step, context, emitter)

        elif step.kind == StepKind.TOOL:
            return self._execute_tool_step(step, source_files, emitter)

        else:
            raise ValueError(f"Unknown step kind: {step.kind}")

    def _execute_role_step(
        self,
        step: PipelineStep,
        source_files: Dict[str, str],
        emitter: EventEmitter,
        language_hints: Optional[List[str]],
        chunk_size: int,
    ) -> Dict[str, Any]:
        """Execute a role-based step using registered models for that role."""
        role_enum = self._normalize_role(step.role)
        if role_enum is None:
            raise ValueError(f"Invalid role '{step.role}'")

        role_models = self.registry.get_models_for_role(role_enum, enabled_only=True)
        if not role_models:
            raise ValueError(f"No models found for role '{step.role}'")

        per_model_findings: Dict[str, List[Finding]] = {}
        model_responses: List[ModelResponse] = []

        for model in role_models:
            per_model_findings[model.model_id] = []
            per_model_findings[model.model_id].extend(
                self._run_model_on_sources(
                    model=model,
                    role_enum=role_enum,
                    source_files=source_files,
                    emitter=emitter,
                    step_id=step.id,
                    chunk_size=chunk_size,
                )
            )

            model_responses.append(
                ModelResponse(
                    model_id=model.model_id,
                    findings=per_model_findings[model.model_id],
                    usage={},
                )
            )

        consensus_findings = self.consensus_engine.merge(
            model_responses,
            strategy="union",
        )

        return {
            "findings": [f.to_dict() for f in consensus_findings],
            "per_model_findings": {
                model_id: [f.to_dict() for f in findings]
                for model_id, findings in per_model_findings.items()
            },
            "metadata": {
                "role": role_enum.value,
                "models_count": len(role_models),
                "models": [m.model_id for m in role_models],
            },
        }

    def _execute_model_step(
        self,
        step: PipelineStep,
        source_files: Dict[str, str],
        emitter: EventEmitter,
        language_hints: Optional[List[str]],
        chunk_size: int,
    ) -> Dict[str, Any]:
        """Execute a model-specific step."""
        per_model_findings: Dict[str, List[Finding]] = {}
        model_responses: List[ModelResponse] = []

        for model_id in step.models or []:
            model = self.registry.get_model(model_id)
            if not model:
                emitter.warning(f"Model '{model_id}' not found", {"model_id": model_id})
                continue

            per_model_findings[model_id] = []
            role_hint = model.roles[0] if model.roles else None

            per_model_findings[model_id].extend(
                self._run_model_on_sources(
                    model=model,
                    role_enum=role_hint,
                    source_files=source_files,
                    emitter=emitter,
                    step_id=step.id,
                    chunk_size=chunk_size,
                )
            )

            model_responses.append(
                ModelResponse(
                    model_id=model_id,
                    findings=per_model_findings[model_id],
                    usage={},
                )
            )

        if not model_responses:
            raise ValueError(f"No adapters available for models {step.models}")

        consensus_findings = self.consensus_engine.merge(model_responses, strategy="union")

        return {
            "findings": [f.to_dict() for f in consensus_findings],
            "per_model_findings": {
                model_id: [f.to_dict() for f in findings]
                for model_id, findings in per_model_findings.items()
            },
            "metadata": {
                "models": [model_id for model_id in per_model_findings.keys()],
            },
        }

    def _normalize_role(self, role: Optional[str]) -> Optional[ModelRole]:
        """Normalize legacy role strings to ModelRole."""
        if not role:
            return None
        mapping = {
            "scan": ModelRole.DEEP_SCAN,
            "deep_scan": ModelRole.DEEP_SCAN,
            "triage": ModelRole.TRIAGE,
            "judge": ModelRole.JUDGE,
            "explain": ModelRole.EXPLAIN,
        }
        try:
            return ModelRole(role)
        except Exception:
            return mapping.get(role.lower())

    def _chunk_batches(self, chunks: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
        if batch_size <= 1:
            for chunk in chunks:
                yield [chunk]
            return
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def _run_model_on_sources(
        self,
        model,
        role_enum,
        source_files: Dict[str, str],
        emitter: EventEmitter,
        step_id: str,
        chunk_size: int,
    ) -> List[Finding]:
        settings = model.settings or {}
        runtime_cfg = settings.get("runtime", {})
        batch_size = int(settings.get("batch_size") or 1)
        max_workers = int(settings.get("chunk_workers") or runtime_cfg.get("max_concurrency") or self.max_workers)
        max_workers = max(1, max_workers)

        collected: List[Finding] = []

        # Emit model_started event with telemetry (only once per model)
        model_start_time = time.time()
        try:
            runtime = self.execution_engine.runtime_manager.get_runtime(model)
            telemetry = runtime.telemetry or {}

            # Determine model type for display
            model_type = "unknown"
            if model.model_type.value.startswith("hf_"):
                model_type = "hf_local"
            elif "_cloud" in model.model_type.value:
                model_type = model.model_type.value.replace("_cloud", "_api")
            elif model.model_type == "tool_ml":
                model_type = "tool"

            emitter.model_started(
                model_id=model.model_id,
                model_name=model.model_name,
                model_type=model_type,
                device=telemetry.get("device"),
                vram_mb=telemetry.get("vram_mb", 0),
                load_time_ms=telemetry.get("load_time_ms", 0),
                quantization=telemetry.get("quantization"),
                precision=telemetry.get("precision"),
            )
        except Exception as e:
            # Don't fail if telemetry fails
            pass

        for file_path, content in source_files.items():
            chunks: List[Dict[str, Any]] = []
            for chunk_content, line_start, line_end in chunk_file_lines(content, chunk_size):
                chunks.append({
                    "code": chunk_content,
                    "file_path": file_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "snippet": chunk_content,
                })

            if not chunks:
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batches = list(self._chunk_batches(chunks, batch_size))
                futures = [
                    executor.submit(
                        self.execution_engine.run_model_batch_to_findings,
                        model,
                        batch,
                        role_enum,
                    )
                    for batch in batches
                ]

                for future in futures:
                    try:
                        batch_findings = future.result()
                    except Exception as e:
                        emitter.warning(f"Batch failed for model {model.model_id}: {e}", {"model_id": model.model_id})
                        continue

                    for chunk_findings in batch_findings:
                        collected.extend(chunk_findings)
                        for finding in chunk_findings:
                            emitter.finding_emitted(finding.to_dict(), step_id)

        # Emit model_completed event with metrics
        model_latency_ms = int((time.time() - model_start_time) * 1000)
        try:
            # TODO: For now, we don't have token tracking for HF models
            # This will be implemented when we add tokenizer-based tracking
            emitter.model_completed(
                model_id=model.model_id,
                findings_count=len(collected),
                latency_ms=model_latency_ms,
                input_tokens=0,  # TODO: Track input tokens
                output_tokens=0,  # TODO: Track output tokens
                tokens_per_sec=0.0,  # TODO: Calculate tokens/sec
            )
        except Exception as e:
            # Don't fail if event emission fails
            pass

        return collected

    def _execute_consensus_step(
        self,
        step: PipelineStep,
        context: PipelineExecutionContext,
        emitter: EventEmitter,
    ) -> Dict[str, Any]:
        """Execute a consensus step."""
        # Collect findings from source steps
        all_findings: List[Finding] = []
        source_metadata: Dict[str, Any] = {}

        for source_id in step.sources:
            source_output = context.step_outputs.get(source_id)
            if source_output:
                findings_dicts = source_output.get("findings", [])
                # Convert dicts back to Finding objects
                for f_dict in findings_dicts:
                    all_findings.append(Finding(**f_dict))
                source_metadata[source_id] = source_output.get("metadata", {})

        if not all_findings:
            return {"findings": [], "metadata": {"strategy": step.strategy.value, "sources": step.sources}}

        # Create mock ModelResponse objects for consensus engine
        mock_responses = [
            ModelResponse(
                model_id="consensus_source",
                findings=all_findings,
                usage={},
            )
        ]

        # Apply consensus strategy
        if step.strategy == ConsensusStrategy.UNION:
            merged = self.consensus_engine.merge(mock_responses, strategy="union")
        elif step.strategy == ConsensusStrategy.MAJORITY:
            merged = self.consensus_engine.merge(mock_responses, strategy="majority")
        elif step.strategy == ConsensusStrategy.WEIGHTED:
            merged = self.consensus_engine.merge(mock_responses, strategy="weighted")
        elif step.strategy == ConsensusStrategy.JUDGE:
            # Judge strategy deferred to Week 3
            emitter.warning(f"Judge consensus not yet implemented, using majority", {"step_id": step.id})
            merged = self.consensus_engine.merge(mock_responses, strategy="majority")
        else:
            merged = all_findings  # Default to union

        # Emit findings merged event
        emitter.findings_merged(step.strategy.value, len(merged))

        return {
            "findings": [f.__dict__ for f in merged],
            "metadata": {
                "strategy": step.strategy.value,
                "sources": step.sources,
                "source_metadata": source_metadata,
            },
        }

    def _execute_gate_step(
        self,
        step: PipelineStep,
        context: PipelineExecutionContext,
        emitter: EventEmitter,
    ) -> Dict[str, Any]:
        """Execute a gate step (conditional branching)."""
        from aegis.pipeline.gating import ConditionEvaluator, ContextBuilder

        # Build evaluation context
        context_builder = ContextBuilder()
        eval_context = context_builder.build(context)

        # Evaluate condition
        evaluator = ConditionEvaluator(missing_path_behavior="zero")
        result = evaluator.evaluate(step.condition, eval_context)

        # Emit gate evaluation event
        emitter.emit(
            "gate_evaluated",
            {
                "step_id": step.id,
                "condition": step.condition.field,
                "operator": step.condition.operator.value,
                "expected": step.condition.value,
                "result": result.result,
                "resolved_values": result.resolved_values,
                "debug_info": result.debug_info,
            },
        )

        # Note: Actual branching is handled by the execute() method
        # This method just evaluates and stores the result
        return {
            "findings": [],
            "metadata": {
                "condition": step.condition.field,
                "operator": step.condition.operator.value,
                "expected_value": step.condition.value,
                "evaluated": True,
                "result": result.result,
                "resolved_values": result.resolved_values,
                "debug_info": result.debug_info,
                "on_true": step.on_true,
                "on_false": step.on_false,
            },
        }

    def _candidate_to_finding(self, candidate: FindingCandidate) -> Finding:
        fingerprint_src = (
            f"{candidate.file_path}|{candidate.line_start}|{candidate.line_end}|"
            f"{candidate.category}|{candidate.description}"
        )
        fingerprint = hashlib.sha1(fingerprint_src.encode("utf-8")).hexdigest()

        return Finding(
            name=candidate.title or candidate.category,
            severity=str(candidate.severity).lower(),
            cwe=candidate.cwe or candidate.metadata.get("cwe", "CWE-000"),
            file=candidate.file_path,
            start_line=int(candidate.line_start or 0),
            end_line=int(candidate.line_end or candidate.line_start or 0),
            message=candidate.description,
            confidence=float(candidate.confidence or 0.0),
            fingerprint=fingerprint,
        )

    def _execute_tool_step(
        self,
        step: PipelineStep,
        source_files: Dict[str, str],
        emitter: EventEmitter,
    ) -> Dict[str, Any]:
        from aegis.tools import DEFAULT_TOOL_REGISTRY

        tool = DEFAULT_TOOL_REGISTRY.get(step.tool_id)
        if not tool:
            raise ValueError(f"Tool '{step.tool_id}' not found")

        result = tool.analyze_project(source_files, step.tool_config or {})
        if not isinstance(result, ParserResult):
            raise ValueError(f"Tool '{step.tool_id}' returned unsupported result type")

        if result.parse_errors:
            emitter.warning(
                f"Tool '{step.tool_id}' parse errors",
                {"tool_id": step.tool_id, "errors": result.parse_errors},
            )

        findings = [self._candidate_to_finding(c) for c in result.findings]
        for finding in findings:
            emitter.finding_emitted(finding.to_dict(), step.id)

        return {
            "findings": [f.to_dict() for f in findings],
            "metadata": {
                "tool_id": step.tool_id,
                "findings_count": len(findings),
                "parse_errors": result.parse_errors,
            },
        }

    def _get_final_findings(self, context: PipelineExecutionContext) -> List[Dict[str, Any]]:
        """Get final findings from execution context."""
        # Look for last consensus step, or last completed step
        for step_id in reversed(context.completed_steps):
            output = context.step_outputs.get(step_id)
            if output and "findings" in output:
                return output["findings"]

        return []
