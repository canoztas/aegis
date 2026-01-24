"""Background scan worker with persistent queue support."""

from dataclasses import dataclass, field
import queue
import threading
from typing import Dict, List, Optional, Any

from aegis.models.registry import ModelRegistryV2
from aegis.utils import debug_scan_log


@dataclass
class ScanJob:
    """Job for standard scan execution."""
    scan_id: str
    source_files: Optional[Dict[str, str]]
    model_ids: List[str]
    consensus_strategy: str
    judge_model_id: Optional[str] = None
    # Cascade consensus configuration (only used when consensus_strategy == "cascade")
    cascade_config: Optional[Dict[str, Any]] = None


class ScanWorker:
    """Single-threaded background worker for scan jobs."""

    def __init__(self, scan_service: Any, use_v2: bool, get_v2_repositories: Any):
        self.scan_service = scan_service
        self.use_v2 = use_v2
        self.get_v2_repositories = get_v2_repositories
        self._queue: "queue.Queue[ScanJob]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._app = None

    def start(self, app) -> None:
        if self._thread:
            return
        self._app = app
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        debug_scan_log("[scan-debug] ScanWorker started")

    def stop(self) -> None:
        self._stop_event.set()

    def enqueue(self, job: ScanJob) -> None:
        self._ensure_scan_state(job.scan_id)
        self._queue.put(job)
        debug_scan_log(f"[scan-debug] enqueued scan {job.scan_id} (models={len(job.model_ids)})")

    def requeue_pending(self) -> None:
        """Requeue scans that were pending/running when the server restarted."""
        if not self.use_v2:
            return
        scan_repo, _ = self.get_v2_repositories()
        try:
            pending = scan_repo.list_by_statuses(["pending", "running"], limit=100)
        except Exception:
            return
        debug_scan_log(f"[scan-debug] requeue pending scans: {len(pending)}")

        for scan_data in pending:
            scan_id = scan_data.get("scan_id")
            if not scan_id:
                continue
            # Note: pipeline_config is a legacy DB column name that stores scan configuration
            scan_config = scan_data.get("pipeline_config", {}) or {}
            model_ids = scan_config.get("models", []) or []
            consensus_strategy = scan_data.get("consensus_strategy", "union")
            judge_model_id = scan_config.get("judge_model_id")
            cascade_config = scan_config.get("cascade")

            if not model_ids:
                registry = ModelRegistryV2()
                model_ids = [m.model_id for m in registry.list_models()]
            if not model_ids:
                continue

            self.enqueue(ScanJob(
                scan_id=scan_id,
                source_files=None,
                model_ids=model_ids,
                consensus_strategy=consensus_strategy,
                judge_model_id=judge_model_id,
                cascade_config=cascade_config,
            ))

    def _ensure_scan_state(self, scan_id: str) -> None:
        state = self.scan_service.scan_state
        if scan_id not in state.status:
            state.status[scan_id] = "pending"
        if scan_id not in state.cancel_events:
            state.cancel_events[scan_id] = threading.Event()

    def _load_source_files(self, scan_id: str) -> Optional[Dict[str, str]]:
        if not self.use_v2:
            return None
        scan_repo, _ = self.get_v2_repositories()
        files = scan_repo.list_files(scan_id)
        if not files:
            return None
        source_files: Dict[str, str] = {}
        for file_info in files:
            file_path = file_info.get("file_path")
            if not file_path:
                continue
            content = scan_repo.get_file(scan_id, file_path)
            if content is None:
                continue
            source_files[file_path] = content
        debug_scan_log(f"[scan-debug] loaded {len(source_files)} files for scan {scan_id}")
        return source_files

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not job:
                self._queue.task_done()
                continue

            if job.source_files is None:
                job.source_files = self._load_source_files(job.scan_id)

            if not job.source_files:
                debug_scan_log(f"[scan-debug] no source files for scan {job.scan_id}")
                if self.use_v2:
                    scan_repo, _ = self.get_v2_repositories()
                    try:
                        scan_repo.update_status(job.scan_id, "failed", error="No source files found")
                    except Exception:
                        pass
                self._queue.task_done()
                continue

            # Check if this is a cascade consensus scan
            if job.consensus_strategy == "cascade" and job.cascade_config:
                self.scan_service.run_cascade_background(
                    scan_id=job.scan_id,
                    source_files=job.source_files,
                    cascade_config=job.cascade_config,
                    app=self._app,
                )
            else:
                self.scan_service.run_background(
                    scan_id=job.scan_id,
                    source_files=job.source_files,
                    model_ids=job.model_ids,
                    consensus_strategy=job.consensus_strategy,
                    judge_model_id=job.judge_model_id,
                    app=self._app,
                )
            debug_scan_log(f"[scan-debug] scan completed: {job.scan_id}")
            self._queue.task_done()
