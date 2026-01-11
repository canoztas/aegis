"""Scan orchestration service for background execution."""

from dataclasses import dataclass
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Any, Optional, Iterable

from aegis.consensus.engine import ConsensusEngine
from aegis.data_models import ScanResult, ModelResponse, Finding
from aegis.events import EventEmitter
from aegis.models.engine import ModelExecutionEngine, _candidate_to_finding
from aegis.models.registry import ModelRegistryV2
from aegis.utils import chunk_file_lines, debug_scan_log


@dataclass
class ScanState:
    """Shared in-memory scan state containers."""
    results: Dict[str, ScanResult]
    status: Dict[str, str]
    cancel_events: Dict[str, threading.Event]


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

                if self._is_cancelled(scan_id):
                    debug_scan_log(f"[scan-debug] scan cancelled before execution: {scan_id}")
                    self.scan_state.status[scan_id] = "cancelled"
                    emitter.emit("cancelled", {"message": "Scan cancelled before execution"})
                    if self.use_v2:
                        scan_repo, _ = self.get_v2_repositories()
                        try:
                            scan_repo.update_status(scan_id, "cancelled")
                        except Exception as e:
                            print(f"Warning: Failed to mark scan cancelled in database: {e}")
                    return

                registry = ModelRegistryV2()
                engine = ModelExecutionEngine(registry)
                consensus = ConsensusEngine()

                per_model_findings: Dict[str, List[Finding]] = {}
                model_responses: List[ModelResponse] = []

                total_work_items = len(model_ids) * len(source_files)
                processed_items = 0

                for model_id in model_ids:
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

                    # Emit step and model started events
                    model_name = model.display_name or model.model_name or model_id
                    model_type = model.model_type or "unknown"
                    device = (model.settings or {}).get("runtime", {}).get("device", "cpu")
                    
                    emitter.step_started(step_id=model_id, step_kind="model_scan")
                    emitter.model_started(
                        model_id=model_id,
                        model_name=model_name,
                        model_type=model_type,
                        device=device,
                    )
                    model_start_time = time.time()

                    for file_path, content in source_files.items():
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
                    self.scan_state.status[scan_id] = "cancelled"
                    emitter.emit("cancelled", {"message": "Scan cancelled"})
                    if self.use_v2:
                        scan_repo, _ = self.get_v2_repositories()
                        try:
                            scan_repo.update_status(scan_id, "cancelled")
                        except Exception as e:
                            print(f"Warning: Failed to mark scan cancelled in database: {e}")
                    return

                self.scan_state.results[scan_id] = scan_result
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
