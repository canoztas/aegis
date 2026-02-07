"""Scan orchestration service for background execution."""

from dataclasses import dataclass
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Any, Optional, Iterable, Set

from aegis.consensus.engine import ConsensusEngine
from aegis.consensus.cascade import (
    CascadeConfig,
    CascadeResult,
    PassResult,
    identify_flagged_files,
)
from aegis.data_models import ScanResult, ModelResponse, Finding
from aegis.events import EventEmitter
from aegis.models.engine import ModelExecutionEngine, _candidate_to_finding
from aegis.models.registry import ModelRegistryV2
from aegis.models.runtime_manager import DEFAULT_RUNTIME_MANAGER
from aegis.utils import chunk_file_lines, debug_scan_log


def _get_model_device(model, engine) -> str:
    """Get actual device from runtime manager, falling back to settings."""
    try:
        # Try to get from runtime telemetry (if model is already loaded)
        runtime = engine.runtime_manager.get_runtime(model)
        if hasattr(runtime, 'telemetry') and runtime.telemetry:
            device = runtime.telemetry.get('device')
            if device:
                return str(device)
        # Try provider telemetry
        if hasattr(runtime, 'provider') and hasattr(runtime.provider, 'get_telemetry'):
            telemetry = runtime.provider.get_telemetry()
            device = telemetry.get('device')
            if device:
                return str(device)
    except Exception:
        pass
    # Fallback to settings
    settings = model.settings or {}
    return settings.get("runtime", {}).get("device") or settings.get("device") or "cpu"


@dataclass
class ScanState:
    """Shared in-memory scan state containers."""
    results: Dict[str, ScanResult]
    status: Dict[str, str]
    cancel_events: Dict[str, threading.Event]
    results_ts: Dict[str, float] = None  # Timestamps for cache eviction

    def __post_init__(self):
        if self.results_ts is None:
            self.results_ts = {}

    def store_result(self, scan_id: str, result: ScanResult) -> None:
        """Store a scan result with timestamp for TTL eviction."""
        self.results[scan_id] = result
        self.results_ts[scan_id] = time.time()


class ScanService:
    """Runs scans in a background worker-friendly form."""

    def __init__(
        self,
        scan_state: ScanState,
        use_v2: bool,
        get_v2_repositories: Callable[[], Any],
    ):
        self.scan_state = scan_state
        self.use_v2 = use_v2
        self.get_v2_repositories = get_v2_repositories

    def _is_cancelled(self, scan_id: str) -> bool:
        event = self.scan_state.cancel_events.get(scan_id)
        return bool(event and event.is_set())

    def _chunk_batches(self, chunks: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
        if batch_size <= 1:
            for chunk in chunks:
                yield [chunk]
            return
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def run_background(
        self,
        scan_id: str,
        source_files: Dict[str, str],
        model_ids: List[str],
        consensus_strategy: str,
        app,
        judge_model_id: Optional[str] = None,
        chunk_size: int = 800,
    ) -> None:
        """Execute scan work inside a background thread."""
        with app.app_context():
            emitter = EventEmitter(scan_id)
            try:
                processed_files: set[str] = set()
                cancel_requested = False

                debug_scan_log(
                    f"[scan-debug] scan start: {scan_id} models={model_ids} files={len(source_files)} "
                    f"chunk_size={chunk_size}"
                )
                self.scan_state.status[scan_id] = "running"
                emitter.pipeline_started("multi_model_scan", "1.0")
                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "running")
                    except Exception as e:
                        print(f"Warning: Failed to mark scan running in database: {e}")

                registry = ModelRegistryV2()
                engine = ModelExecutionEngine(registry)
                consensus = ConsensusEngine()

                per_model_findings: Dict[str, List[Finding]] = {}
                model_responses: List[ModelResponse] = []

                total_work_items = len(model_ids) * len(source_files)
                processed_items = 0

                def finalize_scan(status: str, strategy_override: Optional[str] = None) -> None:
                    nonlocal processed_files, model_responses
                    effective_strategy = strategy_override or (consensus_strategy or "union")
                    if status == "cancelled":
                        effective_strategy = "union"

                    try:
                        consensus_findings = consensus.merge(
                            model_responses,
                            strategy=effective_strategy,
                        )
                    except Exception as e:
                        emitter.warning("Consensus failed", {"error": str(e)})
                        consensus_findings = []

                    scan_result = ScanResult(
                        scan_id=scan_id,
                        consensus_findings=consensus_findings,
                        per_model_findings=per_model_findings,
                        scan_metadata={
                            "models": model_ids,
                            "strategy": consensus_strategy,
                            "effective_strategy": effective_strategy,
                            "files_scanned": len(processed_files) or len(source_files),
                            "total_findings": len(consensus_findings),
                            "status": status,
                            "partial": status != "completed",
                        },
                        source_files=source_files,
                    )

                    self.scan_state.store_result(scan_id, scan_result)
                    self.scan_state.status[scan_id] = status

                    if self.use_v2:
                        scan_repo, finding_repo = self.get_v2_repositories()
                        try:
                            scan_repo.update_status(scan_id, status)
                            scan_repo.update_progress(
                                scan_id,
                                total_files=len(source_files),
                                processed_files=len(processed_files) or len(source_files),
                            )

                            finding_repo.create_batch(
                                scan_result.consensus_findings,
                                scan_id,
                                is_consensus=True,
                            )
                            for model_id, findings in scan_result.per_model_findings.items():
                                finding_repo.create_batch(
                                    findings,
                                    scan_id,
                                    model_id=model_id,
                                    is_consensus=False,
                                )
                        except Exception as e:
                            print(f"Warning: Failed to persist scan to database: {e}")

                if self._is_cancelled(scan_id):
                    debug_scan_log(f"[scan-debug] scan cancelled before execution: {scan_id}")
                    emitter.emit("cancelled", {"message": "Scan cancelled before execution"})
                    finalize_scan("cancelled")
                    return

                for model_id in model_ids:
                    if cancel_requested or self._is_cancelled(scan_id):
                        cancel_requested = True
                        break
                    model = registry.get_model(model_id)
                    if not model:
                        emitter.warning(f"Model '{model_id}' not found", {"model_id": model_id})
                        debug_scan_log(f"[scan-debug] model not found: {model_id}")
                        continue

                    per_model_findings[model_id] = []
                    settings = model.settings or {}
                    runtime_cfg = settings.get("runtime", {})
                    batch_size = int(settings.get("batch_size") or 1)
                    max_workers = int(settings.get("chunk_workers") or runtime_cfg.get("max_concurrency") or 1)
                    max_workers = max(1, max_workers)
                    debug_scan_log(
                        f"[scan-debug] model config: {model_id} batch_size={batch_size} "
                        f"max_workers={max_workers} roles={model.roles}"
                    )

                    # Check if HF model needs downloading and emit event
                    model_name = model.display_name or model.model_name or model_id
                    model_type = model.model_type or "unknown"
                    cache_status = engine.runtime_manager.is_model_cached(model)
                    if cache_status.get("needs_download"):
                        size_mb = cache_status.get("size_mb", 0)
                        emitter.model_downloading(
                            model_id=model_id,
                            model_name=model_name,
                            progress_pct=0,
                            total_mb=float(size_mb),
                            file_name="Preparing download..."
                        )
                        debug_scan_log(
                            f"[scan-debug] model needs download: {model_id} (~{size_mb}MB)"
                        )

                    # Emit step and model started events
                    device = _get_model_device(model, engine)

                    emitter.step_started(step_id=model_id, step_kind="model_scan")
                    emitter.model_started(
                        model_id=model_id,
                        model_name=model_name,
                        model_type=model_type,
                        device=device,
                    )
                    model_start_time = time.time()

                    for file_path, content in source_files.items():
                        if cancel_requested or self._is_cancelled(scan_id):
                            cancel_requested = True
                            break
                        processed_files.add(file_path)
                        processed_items += 1
                        progress_pct = int((processed_items / total_work_items) * 100)
                        file_name = os.path.basename(file_path)
                        model_name = model.display_name or model.model_name or model_id
                        
                        emitter.progress_update(
                            progress_pct=progress_pct, 
                            current=processed_items, 
                            total=total_work_items, 
                            message=f"Scanning {file_name} [{model_name}]"
                        )

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
                            debug_scan_log(f"[scan-debug] no chunks for {file_path} (scan={scan_id})")
                            continue
                        debug_scan_log(
                            f"[scan-debug] chunks for {file_path}: {len(chunks)} "
                            f"(scan={scan_id}, model={model_id})"
                        )

                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            batches = list(self._chunk_batches(chunks, batch_size))
                            debug_scan_log(
                                f"[scan-debug] submitting {len(batches)} batches "
                                f"(scan={scan_id}, model={model_id})"
                            )
                            futures = [
                                executor.submit(
                                    engine.run_model_batch_sync,
                                    model,
                                    batch,
                                    model.roles[0] if model.roles else None,
                                )
                                for batch in batches
                            ]

                            for future, batch in zip(futures, batches):
                                if cancel_requested or self._is_cancelled(scan_id):
                                    cancel_requested = True
                                    break
                                try:
                                    batch_results = future.result()
                                except Exception as e:
                                    emitter.warning(f"Batch failed for model {model_id}: {e}", {"model_id": model_id})
                                    debug_scan_log(f"[scan-debug] batch failed: {model_id} error={e}")
                                    continue

                                for result, chunk in zip(batch_results, batch):
                                    if result.parse_errors:
                                        raw_snippet = None
                                        if result.raw_output:
                                            raw_snippet = str(result.raw_output)
                                            if len(raw_snippet) > 800:
                                                raw_snippet = raw_snippet[:800] + "..."
                                        emitter.warning(
                                            "Model parse errors",
                                            {
                                                "model_id": model_id,
                                                "file_path": chunk.get("file_path"),
                                                "line_start": chunk.get("line_start"),
                                                "line_end": chunk.get("line_end"),
                                                "errors": result.parse_errors,
                                                "raw_snippet": raw_snippet,
                                            },
                                        )
                                        debug_scan_log(
                                            f"[scan-debug] parse errors: model={model_id} file={chunk.get('file_path')} "
                                            f"errors={result.parse_errors}"
                                        )
                                    chunk_findings = [_candidate_to_finding(c) for c in result.findings]
                                    per_model_findings[model_id].extend(chunk_findings)
                                    for finding in chunk_findings:
                                        emitter.finding_emitted(finding.to_dict(), model_id)

                        if cancel_requested:
                            break

                    # Emit step and model completed events
                    model_duration_ms = int((time.time() - model_start_time) * 1000)
                    model_findings_count = len(per_model_findings[model_id])
                    
                    emitter.model_completed(
                        model_id=model_id,
                        findings_count=model_findings_count,
                        latency_ms=model_duration_ms,
                    )
                    emitter.step_completed(
                        step_id=model_id,
                        findings_count=model_findings_count,
                        duration_ms=model_duration_ms,
                    )

                    model_responses.append(
                        ModelResponse(
                            model_id=model_id,
                            findings=per_model_findings[model_id],
                            usage={},
                        )
                    )

                    if cancel_requested:
                        break

                if cancel_requested or self._is_cancelled(scan_id):
                    debug_scan_log(f"[scan-debug] scan cancelled during execution: {scan_id}")
                    emitter.emit("cancelled", {"message": "Scan cancelled"})
                    finalize_scan("cancelled")
                    return

                if not model_responses:
                    raise ValueError("No runnable models found for scan")

                # Get judge model if needed for judge strategy
                judge_model = None
                if consensus_strategy == "judge" and judge_model_id:
                    judge_model = registry.get_model(judge_model_id)
                    if not judge_model:
                        emitter.warning(
                            f"Judge model '{judge_model_id}' not found, falling back to union",
                            {}
                        )
                        consensus_strategy = "union"

                consensus_findings = consensus.merge(
                    model_responses,
                    strategy=consensus_strategy or "union",
                    judge_model=judge_model,
                )

                scan_result = ScanResult(
                    scan_id=scan_id,
                    consensus_findings=consensus_findings,
                    per_model_findings=per_model_findings,
                    scan_metadata={
                        "models": model_ids,
                        "strategy": consensus_strategy,
                        "files_scanned": len(source_files),
                        "total_findings": len(consensus_findings),
                    },
                    source_files=source_files,
                )

                if self._is_cancelled(scan_id):
                    debug_scan_log(f"[scan-debug] scan cancelled after execution: {scan_id}")
                    emitter.emit("cancelled", {"message": "Scan cancelled"})
                    finalize_scan("cancelled")
                    return

                self.scan_state.store_result(scan_id, scan_result)
                self.scan_state.status[scan_id] = "completed"
                debug_scan_log(
                    f"[scan-debug] scan completed: {scan_id} findings={len(scan_result.consensus_findings)}"
                )

                if self.use_v2:
                    scan_repo, finding_repo = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "completed")
                        scan_repo.update_progress(
                            scan_id,
                            total_files=len(source_files),
                            processed_files=len(source_files),
                        )

                        finding_repo.create_batch(
                            scan_result.consensus_findings,
                            scan_id,
                            is_consensus=True,
                        )
                        for model_id, findings in scan_result.per_model_findings.items():
                            finding_repo.create_batch(
                                findings,
                                scan_id,
                                model_id=model_id,
                                is_consensus=False,
                            )
                    except Exception as e:
                        print(f"Warning: Failed to persist scan to database: {e}")

                emitter.pipeline_completed(len(scan_result.consensus_findings), 0)

            except Exception as e:
                debug_scan_log(f"[scan-debug] scan failed: {scan_id} error={e}")
                self.scan_state.status[scan_id] = "failed"
                emitter.error(str(e))
                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "failed", error=str(e))
                    except Exception as err:
                        print(f"Warning: Failed to mark scan failed in database: {err}")
            finally:
                self.scan_state.cancel_events.pop(scan_id, None)

    def run_cascade_background(
        self,
        scan_id: str,
        source_files: Dict[str, str],
        cascade_config: Dict[str, Any],
        app,
        chunk_size: int = 800,
    ) -> None:
        """Execute cascade consensus scan (two-pass gated flow).

        Pass 1: Scan ALL files with pass1_models, identify files with findings.
        Pass 2: Scan ONLY flagged files with pass2_models.

        Args:
            scan_id: Unique scan identifier
            source_files: Dict mapping file paths to content
            cascade_config: Cascade configuration dict
            app: Flask application context
            chunk_size: Lines per chunk for splitting large files
        """
        with app.app_context():
            emitter = EventEmitter(scan_id)
            config = CascadeConfig.from_dict(cascade_config)
            total_start_time = time.time()

            try:
                debug_scan_log(
                    f"[scan-debug] cascade scan start: {scan_id} "
                    f"pass1_models={config.pass1_models} pass2_models={config.pass2_models} "
                    f"files={len(source_files)}"
                )

                self.scan_state.status[scan_id] = "running"
                emitter.pipeline_started("cascade_consensus", "1.0")

                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "running")
                    except Exception as e:
                        print(f"Warning: Failed to mark scan running in database: {e}")

                registry = ModelRegistryV2()
                engine = ModelExecutionEngine(registry)
                consensus = ConsensusEngine()

                # ============================================================
                # PASS 1: Scan ALL files with pass1_models
                # ============================================================
                emitter.emit("cascade_pass_started", {
                    "pass_number": 1,
                    "models": config.pass1_models,
                    "files_count": len(source_files),
                    "description": "Triage scan - analyzing all files",
                })

                pass1_start_time = time.time()
                pass1_responses, pass1_per_model = self._run_pass(
                    scan_id=scan_id,
                    source_files=source_files,
                    model_ids=config.pass1_models,
                    registry=registry,
                    engine=engine,
                    emitter=emitter,
                    chunk_size=chunk_size,
                    pass_number=1,
                )

                if self._is_cancelled(scan_id):
                    self._finalize_cascade_cancelled(scan_id, emitter)
                    return

                # Apply Pass 1 consensus
                pass1_judge_model = None
                if config.pass1_strategy == "judge" and config.pass1_judge_model_id:
                    pass1_judge_model = registry.get_model(config.pass1_judge_model_id)

                pass1_consensus = consensus.merge(
                    pass1_responses,
                    strategy=config.pass1_strategy,
                    judge_model=pass1_judge_model,
                )

                # Identify flagged files
                flagged_files = identify_flagged_files(pass1_consensus, config)
                pass1_duration_ms = int((time.time() - pass1_start_time) * 1000)

                pass1_result = PassResult(
                    pass_number=1,
                    models_used=config.pass1_models,
                    strategy_used=config.pass1_strategy,
                    files_scanned=len(source_files),
                    files_flagged=len(flagged_files),
                    findings_count=len(pass1_consensus),
                    duration_ms=pass1_duration_ms,
                    consensus_findings=pass1_consensus,
                    per_model_findings=pass1_per_model,
                    flagged_files=flagged_files,
                )

                emitter.emit("cascade_pass_completed", {
                    "pass_number": 1,
                    "files_scanned": len(source_files),
                    "files_flagged": len(flagged_files),
                    "findings_count": len(pass1_consensus),
                    "duration_ms": pass1_duration_ms,
                })

                debug_scan_log(
                    f"[scan-debug] cascade pass 1 complete: {scan_id} "
                    f"flagged={len(flagged_files)}/{len(source_files)} findings={len(pass1_consensus)}"
                )

                # ============================================================
                # PASS 2: Scan only flagged files (if any)
                # ============================================================
                pass2_result: Optional[PassResult] = None
                pass2_skipped = False
                final_findings: List[Finding] = []

                if not flagged_files:
                    # No findings in Pass 1, skip Pass 2
                    pass2_skipped = True
                    final_findings = pass1_consensus
                    emitter.emit("cascade_pass_skipped", {
                        "pass_number": 2,
                        "reason": "No files flagged in Pass 1",
                    })
                    debug_scan_log(f"[scan-debug] cascade pass 2 skipped: {scan_id} (no flagged files)")
                else:
                    # Filter source files to only flagged ones
                    pass2_source_files = {
                        fp: content for fp, content in source_files.items()
                        if fp in flagged_files
                    }

                    emitter.emit("cascade_pass_started", {
                        "pass_number": 2,
                        "models": config.pass2_models,
                        "files_count": len(pass2_source_files),
                        "description": "Deep scan - analyzing flagged files only",
                    })

                    pass2_start_time = time.time()
                    pass2_responses, pass2_per_model = self._run_pass(
                        scan_id=scan_id,
                        source_files=pass2_source_files,
                        model_ids=config.pass2_models,
                        registry=registry,
                        engine=engine,
                        emitter=emitter,
                        chunk_size=chunk_size,
                        pass_number=2,
                    )

                    if self._is_cancelled(scan_id):
                        self._finalize_cascade_cancelled(scan_id, emitter)
                        return

                    # Apply Pass 2 consensus
                    pass2_judge_model = None
                    if config.pass2_strategy == "judge" and config.pass2_judge_model_id:
                        pass2_judge_model = registry.get_model(config.pass2_judge_model_id)

                    pass2_consensus = consensus.merge(
                        pass2_responses,
                        strategy=config.pass2_strategy,
                        judge_model=pass2_judge_model,
                    )

                    pass2_duration_ms = int((time.time() - pass2_start_time) * 1000)

                    pass2_result = PassResult(
                        pass_number=2,
                        models_used=config.pass2_models,
                        strategy_used=config.pass2_strategy,
                        files_scanned=len(pass2_source_files),
                        findings_count=len(pass2_consensus),
                        duration_ms=pass2_duration_ms,
                        consensus_findings=pass2_consensus,
                        per_model_findings=pass2_per_model,
                    )

                    final_findings = pass2_consensus

                    emitter.emit("cascade_pass_completed", {
                        "pass_number": 2,
                        "files_scanned": len(pass2_source_files),
                        "findings_count": len(pass2_consensus),
                        "duration_ms": pass2_duration_ms,
                    })

                    debug_scan_log(
                        f"[scan-debug] cascade pass 2 complete: {scan_id} "
                        f"files={len(pass2_source_files)} findings={len(pass2_consensus)}"
                    )

                # ============================================================
                # FINALIZE: Build cascade result and persist
                # ============================================================
                total_duration_ms = int((time.time() - total_start_time) * 1000)

                cascade_result = CascadeResult(
                    scan_id=scan_id,
                    config=config,
                    pass1_result=pass1_result,
                    pass2_result=pass2_result,
                    pass2_skipped=pass2_skipped,
                    final_findings=final_findings,
                    total_duration_ms=total_duration_ms,
                )

                # Convert to standard ScanResult for storage
                scan_result = cascade_result.to_scan_result()
                scan_result.source_files = source_files

                self.scan_state.store_result(scan_id, scan_result)
                self.scan_state.status[scan_id] = "completed"

                if self.use_v2:
                    scan_repo, finding_repo = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "completed")
                        scan_repo.update_progress(
                            scan_id,
                            total_files=len(source_files),
                            processed_files=len(source_files),
                        )

                        finding_repo.create_batch(
                            scan_result.consensus_findings,
                            scan_id,
                            is_consensus=True,
                        )
                        for model_id, findings in scan_result.per_model_findings.items():
                            finding_repo.create_batch(
                                findings,
                                scan_id,
                                model_id=model_id,
                                is_consensus=False,
                            )
                    except Exception as e:
                        print(f"Warning: Failed to persist cascade scan to database: {e}")

                emitter.pipeline_completed(len(final_findings), 0)
                debug_scan_log(
                    f"[scan-debug] cascade scan completed: {scan_id} "
                    f"total_findings={len(final_findings)} duration={total_duration_ms}ms"
                )

            except Exception as e:
                debug_scan_log(f"[scan-debug] cascade scan failed: {scan_id} error={e}")
                self.scan_state.status[scan_id] = "failed"
                emitter.error(str(e))
                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "failed", error=str(e))
                    except Exception as err:
                        print(f"Warning: Failed to mark cascade scan failed in database: {err}")
            finally:
                self.scan_state.cancel_events.pop(scan_id, None)

    def _run_pass(
        self,
        scan_id: str,
        source_files: Dict[str, str],
        model_ids: List[str],
        registry: ModelRegistryV2,
        engine: ModelExecutionEngine,
        emitter: EventEmitter,
        chunk_size: int,
        pass_number: int,
    ) -> tuple[List[ModelResponse], Dict[str, List[Finding]]]:
        """Run a single pass of model execution on source files.

        Args:
            scan_id: Unique scan identifier
            source_files: Files to scan
            model_ids: Model IDs to use
            registry: Model registry
            engine: Model execution engine
            emitter: Event emitter
            chunk_size: Lines per chunk
            pass_number: 1 or 2 for logging/events

        Returns:
            Tuple of (model_responses, per_model_findings)
        """
        model_responses: List[ModelResponse] = []
        per_model_findings: Dict[str, List[Finding]] = {}

        total_work_items = len(model_ids) * len(source_files)
        processed_items = 0

        for model_id in model_ids:
            if self._is_cancelled(scan_id):
                break

            model = registry.get_model(model_id)
            if not model:
                emitter.warning(f"Model '{model_id}' not found", {"model_id": model_id, "pass": pass_number})
                debug_scan_log(f"[scan-debug] cascade pass {pass_number}: model not found: {model_id}")
                continue

            per_model_findings[model_id] = []
            settings = model.settings or {}
            runtime_cfg = settings.get("runtime", {})
            batch_size = int(settings.get("batch_size") or 1)
            max_workers = int(settings.get("chunk_workers") or runtime_cfg.get("max_concurrency") or 1)
            max_workers = max(1, max_workers)

            model_name = model.display_name or model.model_name or model_id
            model_type = model.model_type or "unknown"

            # Check if HF model needs downloading and emit event
            cache_status = engine.runtime_manager.is_model_cached(model)
            if cache_status.get("needs_download"):
                size_mb = cache_status.get("size_mb", 0)
                emitter.model_downloading(
                    model_id=model_id,
                    model_name=model_name,
                    progress_pct=0,
                    total_mb=float(size_mb),
                    file_name="Preparing download..."
                )
                debug_scan_log(
                    f"[scan-debug] cascade pass {pass_number}: model needs download: {model_id} (~{size_mb}MB)"
                )

            device = _get_model_device(model, engine)

            emitter.step_started(step_id=f"pass{pass_number}:{model_id}", step_kind="model_scan")
            emitter.model_started(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                device=device,
            )
            model_start_time = time.time()

            for file_path, content in source_files.items():
                if self._is_cancelled(scan_id):
                    break

                processed_items += 1
                progress_pct = int((processed_items / total_work_items) * 100)
                file_name = os.path.basename(file_path)

                emitter.progress_update(
                    progress_pct=progress_pct,
                    current=processed_items,
                    total=total_work_items,
                    message=f"Pass {pass_number}: {file_name} [{model_name}]",
                )

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
                            engine.run_model_batch_sync,
                            model,
                            batch,
                            model.roles[0] if model.roles else None,
                        )
                        for batch in batches
                    ]

                    for future, batch in zip(futures, batches):
                        if self._is_cancelled(scan_id):
                            break
                        try:
                            batch_results = future.result()
                        except Exception as e:
                            emitter.warning(
                                f"Batch failed for model {model_id}: {e}",
                                {"model_id": model_id, "pass": pass_number}
                            )
                            continue

                        for result, chunk in zip(batch_results, batch):
                            if result.parse_errors:
                                emitter.warning(
                                    "Model parse errors",
                                    {
                                        "model_id": model_id,
                                        "file_path": chunk.get("file_path"),
                                        "pass": pass_number,
                                        "errors": result.parse_errors,
                                    },
                                )
                            chunk_findings = [_candidate_to_finding(c) for c in result.findings]
                            per_model_findings[model_id].extend(chunk_findings)
                            for finding in chunk_findings:
                                emitter.finding_emitted(finding.to_dict(), model_id)

            model_duration_ms = int((time.time() - model_start_time) * 1000)
            model_findings_count = len(per_model_findings[model_id])

            emitter.model_completed(
                model_id=model_id,
                findings_count=model_findings_count,
                latency_ms=model_duration_ms,
            )
            emitter.step_completed(
                step_id=f"pass{pass_number}:{model_id}",
                findings_count=model_findings_count,
                duration_ms=model_duration_ms,
            )

            model_responses.append(
                ModelResponse(
                    model_id=model_id,
                    findings=per_model_findings[model_id],
                    usage={},
                )
            )

        return model_responses, per_model_findings

    def _finalize_cascade_cancelled(self, scan_id: str, emitter: EventEmitter) -> None:
        """Handle cancellation of cascade scan."""
        debug_scan_log(f"[scan-debug] cascade scan cancelled: {scan_id}")
        self.scan_state.status[scan_id] = "cancelled"
        emitter.emit("cancelled", {"message": "Cascade scan cancelled"})

        if self.use_v2:
            scan_repo, _ = self.get_v2_repositories()
            try:
                scan_repo.update_status(scan_id, "cancelled")
            except Exception as e:
                print(f"Warning: Failed to update cancelled status: {e}")
