"""Data models for Aegis SAST tool."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime


@dataclass
class Finding:
    """A security finding detected by a model."""
    name: str
    severity: Literal["low", "medium", "high", "critical"]
    cwe: str  # e.g., "CWE-79"
    file: str
    start_line: int
    end_line: int
    message: str
    confidence: float  # 0.0 to 1.0
    fingerprint: str  # stable identifier across runs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity,
            "cwe": self.cwe,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "message": self.message,
            "confidence": self.confidence,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Security Issue"),
            severity=data.get("severity", "medium"),
            cwe=data.get("cwe", "CWE-20"),
            file=data.get("file", ""),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            message=data.get("message", ""),
            confidence=data.get("confidence", 0.5),
            fingerprint=data.get("fingerprint", ""),
        )


@dataclass
class ModelRequest:
    """Request to run a model on code."""
    code_context: str
    file_path: str
    language: str
    prompt_template_id: str = "sast_cwe_minimal"
    cwe_ids: Optional[List[str]] = None
    timeout: int = 300
    system_params: Optional[Dict[str, Any]] = None
    user_params: Optional[Dict[str, Any]] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class ModelResponse:
    """Response from a model execution."""
    model_id: str
    findings: List[Finding]
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Any] = None
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "findings": [f.to_dict() for f in self.findings],
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class HealthResponse:
    """Health check response from an adapter."""
    healthy: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ScanResult:
    """Complete scan result with consensus and per-model findings."""
    scan_id: str
    consensus_findings: List[Finding]
    per_model_findings: Dict[str, List[Finding]]  # model_id -> findings
    scan_metadata: Dict[str, Any]
    source_files: Optional[Dict[str, str]] = None  # file_path -> content
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_id": self.scan_id,
            "consensus_findings": [f.to_dict() for f in self.consensus_findings],
            "per_model_findings": {
                model_id: [f.to_dict() for f in findings]
                for model_id, findings in self.per_model_findings.items()
            },
            "scan_metadata": self.scan_metadata,
            "created_at": self.created_at.isoformat(),
            # Don't include source_files in dict to avoid large payloads
            # Use separate endpoint to fetch file content
        }

