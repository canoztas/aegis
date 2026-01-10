"""Scan orchestration service for background execution."""

from dataclasses import dataclass
import threading
from typing import Dict, List, Callable, Any, Optional

from aegis.consensus.engine import ConsensusEngine
from aegis.data_models import ScanResult, ModelResponse, Finding
from aegis.events import EventEmitter
from aegis.models.engine import ModelExecutionEngine
from aegis.models.registry import ModelRegistryV2
from aegis.utils import chunk_file_lines


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

    def run_background(
        self,
        scan_id: str,
        source_files: Dict[str, str],
        model_ids: List[str],
        consensus_strategy: str,
        app,
        chunk_size: int = 800,
    ) -> None:
        """Execute scan work inside a background thread."""
        with app.app_context():
            emitter = EventEmitter(scan_id)
            try:
                self.scan_state.status[scan_id] = "running"
                emitter.pipeline_started("multi_model_scan", "1.0")
                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(scan_id, "running")
                    except Exception as e:
                        print(f"Warning: Failed to mark scan running in database: {e}")

                if self._is_cancelled(scan_id):
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

                for model_id in model_ids:
                    model = registry.get_model(model_id)
                    if not model:
                        emitter.warning(f"Model '{model_id}' not found", {"model_id": model_id})
                        continue

                    per_model_findings[model_id] = []
                    for file_path, content in source_files.items():
                        for chunk_content, line_start, line_end in chunk_file_lines(content, chunk_size):
                            findings = engine.run_model_to_findings(
                                model=model,
                                code=chunk_content,
                                file_path=file_path,
                                role=model.roles[0] if model.roles else None,
                                line_start=line_start,
                                line_end=line_end,
                            )
                            per_model_findings[model_id].extend(findings)
                            for finding in findings:
                                emitter.finding_emitted(finding.to_dict(), model_id)

                    model_responses.append(
                        ModelResponse(
                            model_id=model_id,
                            findings=per_model_findings[model_id],
                            usage={},
                        )
                    )

                if not model_responses:
                    raise ValueError("No runnable models found for scan")

                consensus_findings = consensus.merge(
                    model_responses,
                    strategy=consensus_strategy or "union",
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
