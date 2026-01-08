"""Multi-model runner for parallel execution."""
import concurrent.futures
import hashlib
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from aegis.models import ModelRequest, ModelResponse, ScanResult
from aegis.prompt_builder import PromptBuilder
from aegis.consensus.engine import ConsensusEngine


class MultiModelRunner:
    """Runs multiple models in parallel and merges results."""

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        consensus_engine: Optional[ConsensusEngine] = None,
        max_workers: int = 4,
        emitter: Optional[Any] = None,
    ):
        """Initialize multi-model runner."""
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.consensus_engine = consensus_engine or ConsensusEngine(self.prompt_builder)
        self.max_workers = max_workers
        self.emitter = emitter

    def run_scan(
        self,
        source_files: Dict[str, str],
        models: List[Any],  # List of adapters
        cwe_ids: Optional[List[str]] = None,
        consensus_strategy: str = "union",
        consensus_weights: Optional[Dict[str, float]] = None,
        judge_model: Optional[Any] = None,
        language_hints: Optional[List[str]] = None,
        chunk_size: int = 1000,  # Lines per chunk
        emitter: Optional[Any] = None,  # Optional EventEmitter for progress tracking
    ) -> ScanResult:
        """Run scan across multiple models."""
        import uuid
        from datetime import datetime

        scan_id = str(uuid.uuid4())

        # Use provided emitter or instance emitter
        if emitter:
            self.emitter = emitter

        # Process files
        all_model_responses: Dict[str, List[ModelResponse]] = {}  # file -> responses
        total_files = len(source_files)
        processed_files = 0

        for file_idx, (file_path, content) in enumerate(source_files.items()):
            # Detect language
            language = self._detect_language(file_path, language_hints)

            # Split into chunks if needed
            chunks = self._chunk_file(content, chunk_size)
            total_chunks = len(chunks)

            file_responses: List[ModelResponse] = []

            for chunk_idx, (chunk_content, line_start, line_end) in enumerate(chunks):
                # Emit chunk started event
                if self.emitter:
                    self.emitter.chunk_started(chunk_idx + 1, total_chunks, file_path)

                # Build prompt (CWE IDs auto-selected by language if None)
                prompt = self.prompt_builder.build_prompt(
                    ModelRequest(
                        code_context=chunk_content,
                        file_path=file_path,
                        language=language,
                        cwe_ids=cwe_ids,  # None = auto-select by language
                        line_start=line_start,
                        line_end=line_end,
                    )
                )

                # Run all models in parallel for this chunk
                chunk_responses = self._run_models_parallel(
                    models,
                    ModelRequest(
                        code_context=prompt,
                        file_path=file_path,
                        language=language,
                        cwe_ids=cwe_ids,
                        line_start=line_start,
                        line_end=line_end,
                    ),
                )

                file_responses.extend(chunk_responses)

                # Emit chunk completed event
                if self.emitter:
                    chunk_findings = sum(len(r.findings) for r in chunk_responses)
                    self.emitter.chunk_completed(chunk_idx + 1, chunk_findings)

            all_model_responses[file_path] = file_responses
            processed_files += 1

            # Emit progress update
            if self.emitter:
                progress_pct = (processed_files / total_files) * 100
                self.emitter.progress_update(
                    progress_pct=progress_pct,
                    current=processed_files,
                    total=total_files,
                    message=f"Processing files... ({processed_files}/{total_files})"
                )

        # Group responses by model
        per_model_findings: Dict[str, List] = {}
        for model in models:
            per_model_findings[model.id] = []

        for file_path, responses in all_model_responses.items():
            for response in responses:
                if response.model_id in per_model_findings:
                    per_model_findings[response.model_id].extend(response.findings)

        # Run consensus
        all_responses: List[ModelResponse] = []
        for responses in all_model_responses.values():
            all_responses.extend(responses)

        consensus_findings = self.consensus_engine.merge(
            all_responses,
            strategy=consensus_strategy,
            weights=consensus_weights,
            judge_model=judge_model,
            judge_request_params={
                "language": language_hints[0] if language_hints else "unknown",
                "repo_name": "unknown",
            },
        )

        # Build scan result
        scan_metadata = {
            "models": [m.id for m in models],
            "strategy": consensus_strategy,
            "files_scanned": len(source_files),
            "total_findings": len(consensus_findings),
        }

        return ScanResult(
            scan_id=scan_id,
            consensus_findings=consensus_findings,
            per_model_findings=per_model_findings,
            scan_metadata=scan_metadata,
        )

    def _run_models_parallel(
        self, models: List[Any], request: ModelRequest
    ) -> List[ModelResponse]:
        """Run multiple models in parallel."""
        responses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_model = {
                executor.submit(model.predict, request): model for model in models
            }

            # Emit model started events
            for model in models:
                if self.emitter:
                    self.emitter.model_started(model.id, model.display_name)

            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                start_time = time.time()
                try:
                    response = future.result()
                    responses.append(response)

                    # Emit model completed event
                    if self.emitter:
                        latency_ms = int((time.time() - start_time) * 1000)
                        self.emitter.model_completed(
                            model_id=model.id,
                            findings_count=len(response.findings),
                            latency_ms=latency_ms
                        )

                    # Emit finding events
                    if self.emitter:
                        for finding in response.findings:
                            # Convert Finding object to dict
                            finding_dict = finding.to_dict() if hasattr(finding, 'to_dict') else finding.__dict__
                            self.emitter.finding_emitted(finding_dict, model.id)

                except Exception as e:
                    # Create error response
                    from aegis.models import ModelResponse

                    responses.append(
                        ModelResponse(
                            model_id=model.id,
                            findings=[],
                            error=str(e),
                        )
                    )

                    # Emit model failed event
                    if self.emitter:
                        self.emitter.model_failed(model.id, str(e))

        return responses

    def _detect_language(self, file_path: str, hints: Optional[List[str]]) -> str:
        """Detect programming language from file path."""
        if hints:
            return hints[0]

        ext = Path(file_path).suffix.lower().lstrip(".")
        language_map = {
            "py": "Python",
            "js": "JavaScript",
            "jsx": "JavaScript",
            "ts": "TypeScript",
            "tsx": "TypeScript",
            "java": "Java",
            "cpp": "C++",
            "c": "C",
            "cs": "C#",
            "php": "PHP",
            "rb": "Ruby",
            "go": "Go",
            "rs": "Rust",
            "sql": "SQL",
            "sh": "Shell",
            "bash": "Bash",
            "ps1": "PowerShell",
        }
        return language_map.get(ext, "Unknown")

    def _chunk_file(
        self, content: str, chunk_size: int
    ) -> List[tuple[str, int, int]]:
        """Split file into chunks."""
        lines = content.split("\n")
        chunks = []

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            chunk_content = "\n".join(chunk_lines)
            line_start = i + 1
            line_end = min(i + chunk_size, len(lines))
            chunks.append((chunk_content, line_start, line_end))

        return chunks

