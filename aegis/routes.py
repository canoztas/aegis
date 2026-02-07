"""Routes for Aegis application."""
import os
import json
import time
import uuid
import threading
from typing import Any, Dict, List
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    Response,
    stream_with_context,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from aegis.utils import allowed_file, extract_source_files, debug_scan_log
from aegis.exports import export_sarif, export_csv
from aegis.data_models import ScanResult, Finding
from aegis.models.registry import ModelRegistryV2 as NewModelRegistry
from aegis.models.schema import ModelStatus
from aegis.services.scan_service import ScanService, ScanState
from aegis.services.scan_worker import ScanJob

main_bp = Blueprint("main", __name__)

# Global scan storage
_scan_results: Dict[str, ScanResult] = {}
_scan_results_ts: Dict[str, float] = {}  # Timestamps for cache eviction
_scan_status: Dict[str, str] = {}  # Track scan status: pending, running, completed, failed, cancelled
_scan_cancel_events: Dict[str, threading.Event] = {}  # Cancel events for running scans

# Cache eviction settings
_SCAN_CACHE_TTL = 3600  # 1 hour
_SCAN_CACHE_MAX_SIZE = 100

# V2 database repositories (lazy loaded)
_use_v2 = os.environ.get("AEGIS_USE_V2", "true").lower() == "true"
_scan_repo = None
_finding_repo = None

def get_v2_repositories():
    """Get V2 database repositories (lazy load)."""
    global _scan_repo, _finding_repo
    if _use_v2 and (_scan_repo is None or _finding_repo is None):
        from aegis.database.repositories import ScanRepository, FindingRepository
        _scan_repo = ScanRepository()
        _finding_repo = FindingRepository()
    return _scan_repo, _finding_repo


_scan_state = ScanState(
    results=_scan_results,
    status=_scan_status,
    cancel_events=_scan_cancel_events,
    results_ts=_scan_results_ts,
)
_scan_service = ScanService(
    scan_state=_scan_state,
    use_v2=_use_v2,
    get_v2_repositories=get_v2_repositories,
)
_scan_worker = None


def _cleanup_scan_cache() -> int:
    """Evict expired entries from the in-memory scan results cache.

    Returns the number of evicted entries.
    """
    now = time.time()
    evicted = 0

    # Evict entries older than TTL
    expired_ids = [
        sid for sid, ts in _scan_results_ts.items()
        if now - ts > _SCAN_CACHE_TTL
    ]
    for sid in expired_ids:
        _scan_results.pop(sid, None)
        _scan_results_ts.pop(sid, None)
        evicted += 1

    # If still over max size, evict oldest entries
    if len(_scan_results) > _SCAN_CACHE_MAX_SIZE:
        sorted_ids = sorted(_scan_results_ts, key=_scan_results_ts.get)
        excess = len(_scan_results) - _SCAN_CACHE_MAX_SIZE
        for sid in sorted_ids[:excess]:
            _scan_results.pop(sid, None)
            _scan_results_ts.pop(sid, None)
            evicted += 1

    return evicted


def init_scan_worker(app) -> None:
    """Initialize background scan worker and requeue pending scans."""
    global _scan_worker
    if _scan_worker is None:
        from aegis.services.scan_worker import ScanWorker
        _scan_worker = ScanWorker(
            scan_service=_scan_service,
            use_v2=_use_v2,
            get_v2_repositories=get_v2_repositories,
        )
        _scan_worker.start(app)
        _scan_worker.requeue_pending()


def get_scan_worker():
    """Return the background scan worker."""
    return _scan_worker


@main_bp.after_request
def add_cache_headers(response):
    """Add cache control headers to prevent stale JavaScript in development."""
    if current_app.debug:
        # In debug mode, disable caching for all responses
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@main_bp.route("/")
def index() -> str:
    """Main index page."""
    return render_template("index.html")


@main_bp.route("/models")
def models_page() -> str:
    """Models management page."""
    return render_template("models.html")


@main_bp.route("/history")
def history_page() -> str:
    """Scan history page."""
    return render_template("history.html")


@main_bp.route("/gpu")
def gpu_manager_page() -> str:
    """GPU Runtime Manager page."""
    return render_template("gpu_manager.html")


@main_bp.route("/scan/<scan_id>")
def scan_detail(scan_id: str) -> str:
    """Scan detail page."""
    return render_template("scan_detail.html", scan_id=scan_id)


@main_bp.route("/scan/<scan_id>/progress")
def scan_progress(scan_id: str) -> str:
    """Real-time scan progress page."""
    return render_template("scan_progress.html", scan_id=scan_id)


@main_bp.route("/settings")
def settings_page() -> str:
    """Settings and credentials management page."""
    return redirect(url_for("main.models_page"))


# API Routes
# All model management endpoints moved to aegis/api/routes_models.py
@main_bp.route("/api/scan", methods=["POST"])
def create_scan() -> Any:
    """Create a new scan (runs in background)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file: FileStorage = request.files["file"]

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    data = request.form.to_dict()
    model_ids = data.get("models", "").split(",") if data.get("models") else []
    # CWE IDs are auto-selected by language, no need for user input
    consensus_strategy = data.get("consensus_strategy", "union")
    judge_model_id = data.get("judge_model_id")

    # Cascade consensus configuration
    cascade_config = None
    if consensus_strategy == "cascade":
        # Parse cascade configuration
        pass1_models = data.get("pass1_models", "").split(",") if data.get("pass1_models") else []
        pass2_models = data.get("pass2_models", "").split(",") if data.get("pass2_models") else []
        pass1_models = [m.strip() for m in pass1_models if m.strip()]
        pass2_models = [m.strip() for m in pass2_models if m.strip()]

        cascade_config = {
            "pass1_models": pass1_models,
            "pass2_models": pass2_models,
            "pass1_strategy": data.get("pass1_strategy", "union"),
            "pass2_strategy": data.get("pass2_strategy", "union"),
            "pass1_judge_model_id": data.get("pass1_judge_model_id"),
            "pass2_judge_model_id": data.get("pass2_judge_model_id"),
            "min_severity": data.get("min_severity", "low"),
            "min_confidence": float(data.get("min_confidence", "0.0")),
            "flag_any_finding": data.get("flag_any_finding", "true").lower() == "true",
        }

    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = None
        debug_scan_log(f"[scan-debug] upload saved: {filepath} size={file_size}")

        # Extract source files
        source_files = extract_source_files(filepath)
        debug_scan_log(f"[scan-debug] source files extracted: {len(source_files)}")

        # Validate selected models against new registry
        registry = NewModelRegistry()

        if consensus_strategy == "cascade" and cascade_config:
            # Validate cascade models
            debug_scan_log(
                f"[scan-debug] cascade config received: pass1={cascade_config['pass1_models']} "
                f"pass2={cascade_config['pass2_models']}"
            )

            # Log available models for debugging
            available_models = [m.model_id for m in registry.list_models()]
            debug_scan_log(f"[scan-debug] available models in registry: {available_models}")

            valid_pass1_models = []
            for model_id in cascade_config["pass1_models"]:
                model_id_clean = model_id.strip()
                # get_model now handles trimming internally
                model = registry.get_model(model_id_clean)
                if model:
                    valid_pass1_models.append(model.model_id)  # Use the actual ID from registry
                else:
                    debug_scan_log(f"[scan-debug] pass1 model not found: '{model_id}'")

            valid_pass2_models = []
            for model_id in cascade_config["pass2_models"]:
                model_id_clean = model_id.strip()
                # get_model now handles trimming internally
                model = registry.get_model(model_id_clean)
                if model:
                    valid_pass2_models.append(model.model_id)  # Use the actual ID from registry
                else:
                    debug_scan_log(f"[scan-debug] pass2 model not found: '{model_id}'")

            if not valid_pass1_models:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "error": "No valid Pass 1 models selected",
                    "received": cascade_config["pass1_models"],
                    "available": available_models
                }), 400

            if not valid_pass2_models:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "error": "No valid Pass 2 models selected",
                    "received": cascade_config["pass2_models"],
                    "available": available_models
                }), 400

            cascade_config["pass1_models"] = valid_pass1_models
            cascade_config["pass2_models"] = valid_pass2_models
            # For cascade, use combined model list for display
            valid_model_ids = valid_pass1_models + valid_pass2_models
            debug_scan_log(
                f"[scan-debug] cascade models: pass1={valid_pass1_models} pass2={valid_pass2_models}"
            )
        else:
            valid_model_ids = []
            for model_id in model_ids:
                model_id_clean = model_id.strip()
                model = registry.get_model(model_id_clean)
                if model:
                    valid_model_ids.append(model.model_id)  # Use the actual ID from registry

            if not valid_model_ids:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({"error": "No valid models selected"}), 400
            debug_scan_log(f"[scan-debug] valid models: {valid_model_ids}")

        # Generate scan ID
        scan_id = str(uuid.uuid4())
        _scan_status[scan_id] = "pending"

        # Create cancel event for this scan
        _scan_cancel_events[scan_id] = threading.Event()

        # V2: Create scan record
        if _use_v2:
            scan_repo, _ = get_v2_repositories()
            try:
                # Save source files to database
                from aegis.utils import detect_language
                pipeline_config = {
                    "consensus_strategy": consensus_strategy,
                    "models": valid_model_ids,
                }
                if judge_model_id:
                    pipeline_config["judge_model_id"] = judge_model_id
                if cascade_config:
                    pipeline_config["cascade"] = cascade_config

                scan_repo.create(
                    scan_id=scan_id,
                    pipeline_config=pipeline_config,
                    consensus_strategy=consensus_strategy,
                    upload_filename=filename
                )

                for file_path, content in source_files.items():
                    language = detect_language(file_path)
                    scan_repo.add_file(scan_id, file_path, content, language)

                scan_repo.update_progress(
                    scan_id,
                    total_files=len(source_files),
                    processed_files=0
                )
            except Exception as e:
                print(f"Warning: Failed to create scan record: {e}")

        # Start background scan via worker (reconnectable)
        worker = get_scan_worker()
        if worker is None:
            init_scan_worker(current_app._get_current_object())
            worker = get_scan_worker()

        if worker is None:
            raise RuntimeError("Scan worker failed to initialize")

        worker.enqueue(ScanJob(
            scan_id=scan_id,
            source_files=source_files,
            model_ids=valid_model_ids,
            consensus_strategy=consensus_strategy,
            judge_model_id=judge_model_id,
            cascade_config=cascade_config,
        ))
        debug_scan_log(f"[scan-debug] scan enqueued: {scan_id}")

        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

        # Return immediately with scan ID
        return jsonify({
            "scan_id": scan_id,
            "status": "pending",
            "message": "Scan started"
        })

    except Exception as e:
        debug_scan_log(f"[scan-debug] scan create failed: {e}")
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/scan/<scan_id>/status", methods=["GET"])
def get_scan_status(scan_id: str) -> Any:
    """Get scan status."""
    status = _scan_status.get(scan_id)
    
    started_at = None
    if status == "running" and scan_id in _scan_results:
         # Try to get start time from in-memory results if available (unlikely for running)
         pass

    if _use_v2:
        # Fallback to database
        scan_repo, _ = get_v2_repositories()
        try:
            scan_data = scan_repo.get_by_scan_id(scan_id)
            if scan_data:
                status = scan_data.get("status")
                started_at = scan_data.get("started_at")
        except Exception:
            pass

    return jsonify({
        "scan_id": scan_id, 
        "status": status or "unknown",
        "started_at": started_at
    })


@main_bp.route("/api/scan/<scan_id>/cancel", methods=["POST"])
def cancel_scan(scan_id: str) -> Any:
    """Cancel a running scan."""
    from aegis.events import EventEmitter

    # Check if scan exists
    if scan_id not in _scan_status:
        return jsonify({"error": "Scan not found"}), 404

    status = _scan_status[scan_id]

    # Can only cancel running or pending scans
    if status not in ["running", "pending"]:
        return jsonify({"error": f"Cannot cancel scan with status: {status}"}), 400

    # Set the cancel event
    if scan_id in _scan_cancel_events:
        _scan_cancel_events[scan_id].set()

    # Update status
    _scan_status[scan_id] = "cancelled"

    # Emit cancelled event
    emitter = EventEmitter(scan_id)
    emitter.emit("cancelled", {"message": "Scan cancelled by user"})

    # Update database if V2
    if _use_v2:
        scan_repo, _ = get_v2_repositories()
        try:
            scan_repo.update_status(scan_id, "cancelled")
        except Exception as e:
            print(f"Warning: Failed to update scan status in database: {e}")

    return jsonify({
        "scan_id": scan_id,
        "status": "cancelled",
        "message": "Scan cancelled successfully"
    })


@main_bp.route("/api/scan/<scan_id>/retry", methods=["POST"])
def retry_scan(scan_id: str) -> Any:
    """Retry a failed or cancelled scan."""
    # Check if scan exists
    if scan_id not in _scan_status and scan_id not in _scan_results:
        # Try to load from database
        if _use_v2:
            scan_repo, _ = get_v2_repositories()
            try:
                scan_data = scan_repo.get_by_scan_id(scan_id)
                if not scan_data:
                    return jsonify({"error": "Scan not found"}), 404
            except Exception:
                return jsonify({"error": "Scan not found"}), 404
        else:
            return jsonify({"error": "Scan not found"}), 404

    status = _scan_status.get(scan_id, "unknown")

    # Can only retry failed or cancelled scans
    if status not in ["failed", "cancelled"]:
        return jsonify({"error": f"Cannot retry scan with status: {status}"}), 400

    # Get original scan configuration from database
    if _use_v2:
        scan_repo, _ = get_v2_repositories()
        try:
            scan_data = scan_repo.get_by_scan_id(scan_id)
            if not scan_data:
                return jsonify({"error": "Original scan data not found"}), 404

            # Get source files from database
            files_data = scan_repo.list_files(scan_id)
            source_files = {}
            for file_info in files_data:
                content = scan_repo.get_file(scan_id, file_info['file_path'])
                if content:
                    source_files[file_info['file_path']] = content

            if not source_files:
                return jsonify({"error": "Source files not found"}), 404

            # Get pipeline config
            pipeline_config = scan_data.get("pipeline_config", {})
            consensus_strategy = pipeline_config.get("consensus_strategy", "union")
            judge_model_id = pipeline_config.get("judge_model_id")

            # Create new scan with same configuration
            new_scan_id = str(uuid.uuid4())
            _scan_status[new_scan_id] = "pending"
            _scan_cancel_events[new_scan_id] = threading.Event()

            # Get original upload filename
            upload_filename = scan_data.get("upload_filename")

            # Create new scan record
            scan_repo.create(
                scan_id=new_scan_id,
                pipeline_config=pipeline_config,
                consensus_strategy=consensus_strategy,
                upload_filename=upload_filename
            )

            # Copy source files to new scan
            from aegis.utils import detect_language
            for file_path, content in source_files.items():
                language = detect_language(file_path)
                scan_repo.add_file(new_scan_id, file_path, content, language)

            scan_repo.update_progress(
                new_scan_id,
                total_files=len(source_files),
                processed_files=0
            )

            # Resolve models by ID from registry (fallback to all registered)
            registry = NewModelRegistry()
            model_ids = pipeline_config.get("models", [])
            if model_ids:
                model_ids = [model_id for model_id in model_ids if registry.get_model(model_id)]
            else:
                model_ids = [model.model_id for model in registry.list_models(status=ModelStatus.REGISTERED)]

            if not model_ids:
                return jsonify({"error": "No models available for retry"}), 400

            # Start background scan via worker
            worker = get_scan_worker()
            if worker is None:
                init_scan_worker(current_app._get_current_object())
                worker = get_scan_worker()

            if worker is None:
                raise RuntimeError("Scan worker failed to initialize")

            worker.enqueue(ScanJob(
                scan_id=new_scan_id,
                source_files=source_files,
                model_ids=model_ids,
                consensus_strategy=consensus_strategy,
                judge_model_id=judge_model_id,
            ))

            return jsonify({
                "scan_id": new_scan_id,
                "original_scan_id": scan_id,
                "status": "pending",
                "message": "Scan retry started"
            })

        except Exception as e:
            return jsonify({"error": f"Failed to retry scan: {str(e)}"}), 500
    else:
        return jsonify({"error": "Retry requires database (V2) support"}), 400


@main_bp.route("/api/scan/<scan_id>", methods=["DELETE"])
def delete_scan(scan_id: str) -> Any:
    """Delete a scan and its results."""
    # Check if scan is running
    status = _scan_status.get(scan_id)
    if status in ["running", "pending"]:
        return jsonify({"error": "Cannot delete a running scan. Cancel it first."}), 400

    # Remove from in-memory storage
    _scan_results.pop(scan_id, None)
    _scan_results_ts.pop(scan_id, None)
    _scan_status.pop(scan_id, None)
    _scan_cancel_events.pop(scan_id, None)

    # Remove from database if V2
    deleted_from_db = False
    if _use_v2:
        scan_repo, finding_repo = get_v2_repositories()
        try:
            # Delete findings first (foreign key constraint)
            finding_repo.delete_by_scan_id(scan_id)

            # Delete scan files
            files = scan_repo.list_files(scan_id)
            for file_info in files:
                scan_repo.delete_file(scan_id, file_info['file_path'])

            # Delete scan record
            scan_repo.delete_scan(scan_id)

            deleted_from_db = True
        except Exception as e:
            print(f"Warning: Failed to delete scan from database: {e}")

    return jsonify({
        "scan_id": scan_id,
        "deleted": True,
        "deleted_from_database": deleted_from_db,
        "message": "Scan deleted successfully"
    })


@main_bp.route("/api/scans/clear", methods=["DELETE"])
def clear_all_scans() -> Any:
    """Clear all scan history."""
    deleted_count = 0
    deleted_from_db = False

    # Get all scan IDs
    scan_ids = list(_scan_status.keys())

    # Check if any scans are running
    running_scans = [sid for sid in scan_ids if _scan_status.get(sid) in ["running", "pending"]]
    if running_scans:
        return jsonify({"error": f"Cannot clear history while {len(running_scans)} scan(s) are running. Cancel them first."}), 400

    # Clear in-memory storage
    _scan_results.clear()
    _scan_results_ts.clear()
    _scan_status.clear()
    _scan_cancel_events.clear()
    deleted_count = len(scan_ids)

    # Clear database if V2
    if _use_v2:
        scan_repo, finding_repo = get_v2_repositories()
        try:
            # Get all scans from database
            all_scans = scan_repo.list_all(limit=10000)

            for scan in all_scans:
                scan_id = scan.get('scan_id')
                if scan_id:
                    # Delete findings first (foreign key constraint)
                    finding_repo.delete_by_scan_id(scan_id)

                    # Delete scan files
                    files = scan_repo.list_files(scan_id)
                    for file_info in files:
                        scan_repo.delete_file(scan_id, file_info['file_path'])

                    # Delete scan record
                    scan_repo.delete_scan(scan_id)

            deleted_from_db = True
        except Exception as e:
            print(f"Warning: Failed to clear database: {e}")

    return jsonify({
        "deleted_count": deleted_count,
        "deleted_from_database": deleted_from_db,
        "message": f"Cleared {deleted_count} scan(s) from history"
    })


@main_bp.route("/api/scan/<scan_id>", methods=["GET"])
def get_scan(scan_id: str) -> Any:
    """Get scan results."""
    # First try in-memory cache
    if scan_id in _scan_results:
        return jsonify(_scan_results[scan_id].to_dict())

    # V2: Try database
    if _use_v2:
        scan_repo, finding_repo = get_v2_repositories()
        try:
            scan_data = scan_repo.get_by_scan_id(scan_id)
            if scan_data:
                # Reconstruct ScanResult from database
                consensus_findings = finding_repo.get_consensus_findings(scan_id)

                # Get per-model findings
                from aegis.data_models import Finding
                all_findings = finding_repo.get_all_findings(scan_id, include_consensus=False)
                per_model_findings = {}
                for finding_dict in all_findings:
                    model_id = finding_dict.get('model_id')
                    if model_id and model_id not in per_model_findings:
                        per_model_findings[model_id] = []
                    if model_id:
                        per_model_findings[model_id].append(Finding.from_dict(finding_dict))

                scan_result = ScanResult(
                    scan_id=scan_id,
                    consensus_findings=consensus_findings,
                    per_model_findings=per_model_findings,
                    scan_metadata={
                        "status": scan_data.get("status"),
                        "consensus_strategy": scan_data.get("consensus_strategy"),
                        "total_files": scan_data.get("total_files", 0),
                        "started_at": scan_data.get("started_at"),
                        "completed_at": scan_data.get("completed_at"),
                    }
                )

                # Cache it for future requests
                _scan_results[scan_id] = scan_result
                _scan_results_ts[scan_id] = time.time()

                return jsonify(scan_result.to_dict())
        except Exception as e:
            print(f"Warning: Failed to load scan from database: {e}")

    return jsonify({"error": "Scan not found"}), 404


@main_bp.route("/api/scan/<scan_id>/stream", methods=["GET"])
def stream_scan_progress(scan_id: str) -> Any:
    """
    Stream real-time scan progress via Server-Sent Events (SSE).

    Event types:
    - connected: Initial connection established
    - step_start: Pipeline step started
    - progress: Task progress update
    - finding: New finding discovered
    - step_completed: Pipeline step completed
    - completed: Scan completed
    - error: Error occurred
    - cancelled: Scan cancelled
    """
    from aegis.sse.stream import SSEManager, format_sse_message, format_keepalive
    from aegis.events import get_event_bus  # Import EventBus

    # Get or create SSE manager instance
    if not hasattr(current_app, 'sse_manager'):
        current_app.sse_manager = SSEManager()

    sse_manager = current_app.sse_manager
    event_bus = get_event_bus()  # Get event bus instance

    # Connect client
    connection = sse_manager.connect(scan_id=scan_id)

    def event_stream():
        """Generate SSE events."""
        try:
            # 1. Replay history first
            history = event_bus.get_history(scan_id=scan_id)
            for event in history:
                # Map event type to SSE event name
                sse_event_type = event_bus._map_event_to_sse(event.type)
                msg = format_sse_message(
                    event_type=sse_event_type,
                    data=event.data
                )
                yield msg

            # 2. Stream new events
            while True:
                # Get events from queue (with timeout for keepalive)
                events = connection.get_events(timeout=15.0)

                if events:
                    # Send events
                    for event in events:
                        msg = format_sse_message(
                            event_type=event['event'],
                            data=event['data']
                        )
                        yield msg

                        # Check for terminal events
                        if event['event'] in ['completed', 'error', 'cancelled']:
                            # Send one last event and close
                            return
                else:
                    # Send keepalive comment
                    yield format_keepalive()

        except GeneratorExit:
            # Client disconnected
            sse_manager.disconnect(scan_id=scan_id, client_id=connection.client_id)
        except Exception as e:
            logger = current_app.logger
            logger.error(f"SSE stream error for scan {scan_id}: {e}")
            sse_manager.disconnect(scan_id=scan_id, client_id=connection.client_id)

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive',
        }
    )


@main_bp.route("/api/scan/<scan_id>/file/<path:file_path>", methods=["GET"])
def get_scan_file(scan_id: str, file_path: str) -> Any:
    """Get source code content for a file in a scan."""
    # First try in-memory cache
    if scan_id in _scan_results:
        scan_result = _scan_results[scan_id]
        if scan_result.source_files:
            # Normalize file path (handle URL encoding)
            if file_path not in scan_result.source_files:
                # Try to find by matching end of path
                for stored_path, content in scan_result.source_files.items():
                    if stored_path.endswith(file_path) or file_path.endswith(stored_path):
                        return jsonify({"content": content, "file_path": stored_path})
            else:
                return jsonify({"content": scan_result.source_files[file_path], "file_path": file_path})

    # V2: Try database
    if _use_v2:
        scan_repo, _ = get_v2_repositories()
        try:
            # Normalize file path for database lookup
            content = scan_repo.get_file(scan_id, file_path)
            if content:
                return jsonify({"content": content, "file_path": file_path})

            # Try to find by partial match
            files = scan_repo.list_files(scan_id)
            for file_info in files:
                stored_path = file_info['file_path']
                if stored_path.endswith(file_path) or file_path.endswith(stored_path):
                    content = scan_repo.get_file(scan_id, stored_path)
                    if content:
                        return jsonify({"content": content, "file_path": stored_path})
        except Exception as e:
            print(f"Warning: Failed to load file from database: {e}")

    return jsonify({"error": "File not found"}), 404


@main_bp.route("/api/scan/<scan_id>/sarif", methods=["GET"])
def get_scan_sarif(scan_id: str) -> Any:
    """Get scan results as SARIF."""
    if scan_id not in _scan_results:
        return jsonify({"error": "Scan not found"}), 404
    
    scan_result = _scan_results[scan_id]
    sarif = export_sarif(scan_result)
    
    return Response(
        json.dumps(sarif, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=scan_{scan_id}.sarif.json"},
    )


@main_bp.route("/api/scan/<scan_id>/csv", methods=["GET"])
def get_scan_csv(scan_id: str) -> Any:
    """Get scan results as CSV."""
    if scan_id not in _scan_results:
        return jsonify({"error": "Scan not found"}), 404
    
    scan_result = _scan_results[scan_id]
    
    # Create temporary CSV
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        export_csv(scan_result.consensus_findings, f.name)
        csv_content = open(f.name, "r").read()
        os.unlink(f.name)
    
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scan_{scan_id}.csv"},
    )


@main_bp.route("/api/scans", methods=["GET"])
def list_scans() -> Any:
    """List recent scans."""
    _cleanup_scan_cache()
    limit = request.args.get("limit", 50, type=int)

    scans = []

    # V2: Load from database
    if _use_v2:
        scan_repo, finding_repo = get_v2_repositories()
        try:
            db_scans = scan_repo.list_recent(limit=limit)
            for scan_data in db_scans:
                # Get finding counts
                consensus_findings = finding_repo.get_consensus_findings(scan_data['scan_id'])

                # Calculate severity counts
                severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                for finding in consensus_findings:
                    severity = finding.severity.lower()
                    if severity in severity_counts:
                        severity_counts[severity] += 1

                scans.append({
                    "scan_id": scan_data['scan_id'],
                    "status": scan_data['status'],
                    "consensus_strategy": scan_data.get('consensus_strategy', 'union'),
                    "total_files": scan_data.get('total_files', 0),
                    "total_findings": len(consensus_findings),
                    "severity_counts": severity_counts,
                    "started_at": scan_data.get('started_at'),
                    "completed_at": scan_data.get('completed_at'),
                    "created_at": scan_data.get('created_at')
                })
        except Exception as e:
            print(f"Warning: Failed to load scans from database: {e}")

    # Add in-memory scans (for V1 or as backup)
    for scan_id, scan_result in list(_scan_results.items())[:limit]:
        # Skip if already in database results
        if any(s['scan_id'] == scan_id for s in scans):
            continue

        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for finding in scan_result.consensus_findings:
            severity = finding.severity.lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

        scans.append({
            "scan_id": scan_id,
            "status": "completed",
            "consensus_strategy": scan_result.scan_metadata.get('consensus_strategy', 'union'),
            "total_files": len(scan_result.source_files) if scan_result.source_files else 0,
            "total_findings": len(scan_result.consensus_findings),
            "severity_counts": severity_counts,
            "started_at": None,
            "completed_at": None,
            "created_at": None
        })

    # Add pending/running scans from in-memory status (if not already listed)
    existing_ids = {s.get("scan_id") for s in scans}
    for scan_id, status in _scan_status.items():
        if scan_id in existing_ids:
            continue

        scan_data = None
        if _use_v2:
            try:
                scan_repo, _ = get_v2_repositories()
                scan_data = scan_repo.get_by_scan_id(scan_id)
            except Exception as e:
                print(f"Warning: Failed to load scan {scan_id} from database: {e}")

        scans.append({
            "scan_id": scan_id,
            "status": status,
            "consensus_strategy": (scan_data or {}).get("consensus_strategy", "unknown"),
            "total_files": (scan_data or {}).get("total_files", 0),
            "total_findings": 0,
            "severity_counts": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "started_at": (scan_data or {}).get("started_at"),
            "completed_at": (scan_data or {}).get("completed_at"),
            "created_at": (scan_data or {}).get("created_at"),
        })

    # Sort by created_at (most recent first), handling None values
    scans.sort(key=lambda x: x.get('created_at') or '', reverse=True)

    return jsonify({"scans": scans[:limit]})


@main_bp.route("/health")
def health_check() -> Any:
    """Health check endpoint."""
    return "OK", 200
