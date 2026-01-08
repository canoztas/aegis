"""Routes for Aegis application."""
import os
import json
import uuid
import threading
from typing import Any, Dict, List
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    flash,
    redirect,
    url_for,
    Response,
    stream_with_context,
)
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from aegis.utils import allowed_file, extract_source_files
from aegis.registry import ModelRegistry
from aegis.runner import MultiModelRunner
from aegis.exports import export_sarif, export_csv
from aegis.models import ScanResult

main_bp = Blueprint("main", __name__)

# Global registry and scan storage
_registry = ModelRegistry()
_scan_results: Dict[str, ScanResult] = {}
_scan_status: Dict[str, str] = {}  # Track scan status: pending, running, completed, failed, cancelled
_scan_cancel_events: Dict[str, threading.Event] = {}  # Cancel events for running scans

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


@main_bp.route("/scan/<scan_id>")
def scan_detail(scan_id: str) -> str:
    """Scan detail page."""
    return render_template("scan_detail.html", scan_id=scan_id)


@main_bp.route("/scan/<scan_id>/progress")
def scan_progress(scan_id: str) -> str:
    """Real-time scan progress page."""
    return render_template("scan_progress.html", scan_id=scan_id)


# API Routes

@main_bp.route("/api/models", methods=["GET"])
def list_models() -> Any:
    """List all registered models."""
    return jsonify({"models": _registry.list_all()})


@main_bp.route("/api/models/<provider>", methods=["GET"])
def list_models_by_provider(provider: str) -> Any:
    """List models by provider."""
    return jsonify({"models": _registry.list_by_provider(provider)})


@main_bp.route("/api/models/ollama", methods=["GET"])
def list_ollama_models() -> Any:
    """List installed Ollama models."""
    base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    models = _registry.discover_ollama_models(base_url)
    return jsonify({"models": models})


@main_bp.route("/api/models/ollama/pull", methods=["POST"])
def pull_ollama_model() -> Any:
    """Pull an Ollama model."""
    data = request.get_json()
    model_name = data.get("name")
    
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    result = _registry.pull_ollama_model(model_name, base_url)
    
    if result.get("success"):
        return jsonify(result)
    else:
        return jsonify({"error": result.get("error")}), 500


@main_bp.route("/api/models/ollama/register", methods=["POST"])
def register_ollama_model() -> Any:
    """Register an Ollama model as an adapter."""
    data = request.get_json()
    model_name = data.get("name")
    
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    adapter = _registry.create_ollama_adapter(model_name, base_url)
    
    return jsonify({"adapter": adapter.to_dict()})


@main_bp.route("/api/models/llm/register", methods=["POST"])
def register_llm_model() -> Any:
    """Register a cloud LLM model."""
    data = request.get_json()
    provider = data.get("provider")
    model_name = data.get("model_name")
    api_key = data.get("api_key")
    api_base = data.get("api_base")
    
    if not all([provider, model_name, api_key]):
        return jsonify({"error": "provider, model_name, and api_key required"}), 400
    
    adapter = _registry.create_llm_adapter(provider, model_name, api_key, api_base)
    return jsonify({"adapter": adapter.to_dict()})


@main_bp.route("/api/models/hf/register", methods=["POST"])
def register_hf_model() -> Any:
    """Register a HuggingFace model."""
    data = request.get_json()
    model_id = data.get("model_id")
    task = data.get("task", "text-generation")
    cache_dir = data.get("cache_dir")
    
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    
    adapter = _registry.create_hf_adapter(model_id, task, cache_dir)
    return jsonify({"adapter": adapter.to_dict()})


@main_bp.route("/api/models/classic/register", methods=["POST"])
def register_classic_model() -> Any:
    """Register a classic ML model."""
    data = request.get_json()
    model_path = data.get("model_path")
    model_type = data.get("model_type", "sklearn")
    
    if not model_path:
        return jsonify({"error": "model_path required"}), 400
    
    adapter = _registry.create_classic_adapter(model_path, model_type)
    return jsonify({"adapter": adapter.to_dict()})


@main_bp.route("/api/models/<adapter_id>", methods=["DELETE"])
def remove_model(adapter_id: str) -> Any:
    """Remove a model adapter."""
    success = _registry.remove(adapter_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Model not found"}), 404


def _run_scan_background(scan_id: str, source_files: Dict[str, str], models: List[Any],
                         consensus_strategy: str, judge_model: Any, app):
    """Run scan in background thread."""
    from aegis.events import EventEmitter

    with app.app_context():
        try:
            _scan_status[scan_id] = "running"
            emitter = EventEmitter(scan_id)

            # Emit pipeline started
            emitter.pipeline_started("multi_model_scan", "1.0")

            # Check for cancellation before starting
            if scan_id in _scan_cancel_events and _scan_cancel_events[scan_id].is_set():
                _scan_status[scan_id] = "cancelled"
                emitter.emit("cancelled", {"message": "Scan cancelled before execution"})
                return

            # Run scan with EventEmitter for progress tracking
            runner = MultiModelRunner(emitter=emitter)
            scan_result = runner.run_scan(
                source_files=source_files,
                models=models,
                cwe_ids=None,  # Auto-select by language
                consensus_strategy=consensus_strategy,
                judge_model=judge_model,
                emitter=emitter,
            )

            # Check for cancellation after scan
            if scan_id in _scan_cancel_events and _scan_cancel_events[scan_id].is_set():
                _scan_status[scan_id] = "cancelled"
                emitter.emit("cancelled", {"message": "Scan cancelled"})
                return

            # Store result with source files for code snippet display
            scan_result.source_files = source_files
            _scan_results[scan_id] = scan_result
            _scan_status[scan_id] = "completed"

            # V2: Persist to database
            if _use_v2:
                scan_repo, finding_repo = get_v2_repositories()
                try:
                    # Update scan status
                    scan_repo.update_status(scan_id, 'completed')
                    scan_repo.update_progress(
                        scan_id,
                        total_files=len(source_files),
                        processed_files=len(source_files)
                    )

                    # Save consensus findings
                    finding_repo.create_batch(
                        scan_result.consensus_findings,
                        scan_id,
                        is_consensus=True
                    )

                    # Save per-model findings
                    for model_id, findings in scan_result.per_model_findings.items():
                        finding_repo.create_batch(
                            findings,
                            scan_id,
                            model_id=model_id,
                            is_consensus=False
                        )
                except Exception as e:
                    print(f"Warning: Failed to persist scan to database: {e}")

            # Emit completion event
            emitter.pipeline_completed(len(scan_result.consensus_findings), 0)

        except Exception as e:
            _scan_status[scan_id] = "failed"
            emitter = EventEmitter(scan_id)
            emitter.error(str(e))
        finally:
            # Clean up cancel event
            if scan_id in _scan_cancel_events:
                del _scan_cancel_events[scan_id]


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

    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract source files
        source_files = extract_source_files(filepath)

        # Get selected models
        models = []
        for model_id in model_ids:
            adapter = _registry.get(model_id)
            if adapter:
                models.append(adapter)

        if not models:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "No valid models selected"}), 400

        # Get judge model if specified
        judge_model = None
        if judge_model_id:
            judge_model = _registry.get(judge_model_id)

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
                scan_repo.create(
                    scan_id=scan_id,
                    pipeline_config={"consensus_strategy": consensus_strategy},
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

        # Start background scan
        thread = threading.Thread(
            target=_run_scan_background,
            args=(scan_id, source_files, models, consensus_strategy, judge_model, current_app._get_current_object())
        )
        thread.daemon = True
        thread.start()

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
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/scan/<scan_id>/status", methods=["GET"])
def get_scan_status(scan_id: str) -> Any:
    """Get scan status."""
    status = _scan_status.get(scan_id, "unknown")
    return jsonify({"scan_id": scan_id, "status": status})


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

            # Get models (from registry - assumes same models still available)
            models = []
            for model_id in pipeline_config.get("models", []):
                adapter = _registry.get(model_id)
                if adapter:
                    models.append(adapter)

            if not models:
                # Fallback: use all available models
                all_models = _registry.list_all()
                for model_info in all_models:
                    adapter = _registry.get(model_info['id'])
                    if adapter:
                        models.append(adapter)

            if not models:
                return jsonify({"error": "No models available for retry"}), 400

            # Start background scan
            thread = threading.Thread(
                target=_run_scan_background,
                args=(new_scan_id, source_files, models, consensus_strategy, None, current_app._get_current_object())
            )
            thread.daemon = True
            thread.start()

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
    if scan_id in _scan_results:
        del _scan_results[scan_id]
    if scan_id in _scan_status:
        del _scan_status[scan_id]
    if scan_id in _scan_cancel_events:
        del _scan_cancel_events[scan_id]

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
                from aegis.models import Finding
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

    # Get or create SSE manager instance
    if not hasattr(current_app, 'sse_manager'):
        current_app.sse_manager = SSEManager()

    sse_manager = current_app.sse_manager

    # Connect client
    connection = sse_manager.connect(scan_id=scan_id)

    def event_stream():
        """Generate SSE events."""
        try:
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


# Pipeline Routes

@main_bp.route("/api/pipelines", methods=["GET"])
def list_pipelines() -> Any:
    """List available pipelines."""
    from aegis.pipeline import PipelineRegistry

    registry = PipelineRegistry()
    all_pipelines = registry.list_all()

    return jsonify({
        "presets": all_pipelines["presets"],
        "custom": all_pipelines["custom"],
    })


@main_bp.route("/api/pipelines/<pipeline_name>", methods=["GET"])
def get_pipeline(pipeline_name: str) -> Any:
    """Get pipeline configuration by name."""
    from aegis.pipeline import PipelineRegistry

    registry = PipelineRegistry()

    # Try presets first
    pipeline = registry.get_pipeline(pipeline_name, is_preset=True)
    if not pipeline:
        # Try custom
        pipeline = registry.get_pipeline(pipeline_name, is_preset=False)

    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404

    return jsonify({
        "name": pipeline.name,
        "version": pipeline.version,
        "description": pipeline.description,
        "is_preset": pipeline.is_preset,
        "steps": [
            {
                "id": step.id,
                "kind": step.kind.value,
                "role": step.role,
                "models": step.models,
                "enabled": step.enabled,
            }
            for step in pipeline.steps
        ],
    })


@main_bp.route("/api/scan/pipeline", methods=["POST"])
def create_pipeline_scan() -> Any:
    """Create a scan using a pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file: FileStorage = request.files["file"]

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    data = request.form.to_dict()
    pipeline_name = data.get("pipeline", "classic")  # Default to classic

    filepath = None
    try:
        # Import pipeline components
        from aegis.pipeline import PipelineLoader, PipelineExecutor

        # Load pipeline
        loader = PipelineLoader()
        try:
            pipeline = loader.load_preset(pipeline_name)
        except FileNotFoundError:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Pipeline '{pipeline_name}' not found"}), 404

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract source files
        source_files = extract_source_files(filepath)

        # Execute pipeline
        executor = PipelineExecutor()
        scan_id = str(uuid.uuid4())

        context = executor.execute(
            pipeline=pipeline,
            source_files=source_files,
            scan_id=scan_id,
        )

        # Get final findings
        final_findings = executor._get_final_findings(context)

        # Convert findings dicts to Finding objects for compatibility
        from aegis.models import Finding
        consensus_findings = [Finding(**f) for f in final_findings]

        # Create ScanResult for compatibility
        scan_result = ScanResult(
            scan_id=scan_id,
            consensus_findings=consensus_findings,
            per_model_findings={},  # Could be extracted from context if needed
            consensus_strategy=pipeline.name,
        )
        scan_result.source_files = source_files

        # Store result
        _scan_results[scan_id] = scan_result

        # V2: Persist to database
        if _use_v2:
            scan_repo, finding_repo = get_v2_repositories()
            try:
                # Create scan record with pipeline config
                scan_repo.create(
                    scan_id=scan_id,
                    pipeline_config={
                        "pipeline_name": pipeline.name,
                        "pipeline_version": pipeline.version,
                        "steps": len(pipeline.steps),
                    },
                    consensus_strategy=pipeline.name,
                )

                # Store consensus findings
                for finding in consensus_findings:
                    finding_repo.create_consensus_finding(
                        scan_id=scan_id,
                        finding=finding,
                    )

                # Mark scan as completed
                scan_repo.update_status(scan_id, "completed")

            except Exception as e:
                current_app.logger.error(f"Failed to persist pipeline scan to database: {e}")

        # Cleanup
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "scan_id": scan_id,
            "pipeline": pipeline.name,
            "total_findings": len(consensus_findings),
            "steps_completed": len(context.completed_steps),
            "steps_failed": len(context.failed_steps),
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        current_app.logger.error(f"Pipeline scan failed: {e}")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/scans", methods=["GET"])
def list_scans() -> Any:
    """List recent scans."""
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

    # Sort by created_at (most recent first), handling None values
    scans.sort(key=lambda x: x.get('created_at') or '', reverse=True)

    return jsonify({"scans": scans[:limit]})


# Legacy routes for backward compatibility

@main_bp.route("/upload", methods=["POST"])
def upload_file() -> Any:
    """Legacy upload route."""
    if "file" not in request.files:
        flash("No file selected")
        return redirect(request.url)
    
    file: FileStorage = request.files["file"]
    
    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        try:
            # Use new scan API
            source_files = extract_source_files(filepath)
            
            # Get default Ollama model
            base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = current_app.config.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
            adapter = _registry.create_ollama_adapter(model_name, base_url)
            
            runner = MultiModelRunner()
            scan_result = runner.run_scan(
                source_files=source_files,
                models=[adapter],
            )
            
            _scan_results[scan_result.scan_id] = scan_result
            
            os.remove(filepath)
            
            # Convert to legacy format for template
            legacy_results = []
            for finding in scan_result.consensus_findings:
                legacy_results.append({
                    "file_path": finding.file,
                    "line_number": finding.start_line,
                    "severity": finding.severity,
                    "score": finding.confidence * 10.0,
                    "category": finding.cwe,
                    "description": finding.message,
                    "recommendation": f"Address {finding.cwe} vulnerability",
                    "code_snippet": "",
                })
            
            return render_template("results.html", results=legacy_results)
        
        except Exception as e:
            flash(f"Error analyzing file: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for("main.index"))
    
    flash("Invalid file type. Please upload a ZIP file.")
    return redirect(url_for("main.index"))


@main_bp.route("/api/analyze", methods=["POST"])
def api_analyze() -> Any:
    """Legacy API analyze route."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file: FileStorage = request.files["file"]
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        source_files = extract_source_files(filepath)
        
        base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = current_app.config.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        adapter = _registry.create_ollama_adapter(model_name, base_url)
        
        runner = MultiModelRunner()
        scan_result = runner.run_scan(
            source_files=source_files,
            models=[adapter],
        )
        
        os.remove(filepath)
        
        # Convert to legacy format
        legacy_results = []
        for finding in scan_result.consensus_findings:
            legacy_results.append({
                "file_path": finding.file,
                "line_number": finding.start_line,
                "severity": finding.severity,
                "score": finding.confidence * 10.0,
                "category": finding.cwe,
                "description": finding.message,
                "recommendation": f"Address {finding.cwe} vulnerability",
                "code_snippet": "",
            })
        
        return jsonify({"results": legacy_results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/health")
def health_check() -> Any:
    """Health check endpoint."""
    return "OK", 200
