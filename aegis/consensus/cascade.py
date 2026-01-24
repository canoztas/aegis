"""Cascade Consensus Strategy - Two-pass gated vulnerability scanning.

This module implements a two-pass consensus strategy:
- Pass 1 (Triage): Scan ALL files with N models, identify files with findings
- Pass 2 (Deep Scan): Scan ONLY flagged files with M models for detailed analysis

The final result is the Pass 2 consensus, with Pass 1 results preserved as metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any, Set
from datetime import datetime
import time

from aegis.data_models import Finding, ModelResponse, ScanResult


# Severity ordering for filtering
SEVERITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass
class CascadeConfig:
    """Configuration for cascade consensus strategy.

    Attributes:
        pass1_models: List of model IDs for Pass 1 (triage)
        pass2_models: List of model IDs for Pass 2 (deep scan)
        pass1_strategy: Consensus strategy for Pass 1 (default: union)
        pass2_strategy: Consensus strategy for Pass 2 (default: union)
        pass1_judge_model_id: Judge model for Pass 1 if strategy is "judge"
        pass2_judge_model_id: Judge model for Pass 2 if strategy is "judge"
        min_severity: Minimum severity to flag a file (default: low)
        min_confidence: Minimum confidence to flag a file (default: 0.0)
        flag_any_finding: If True, any finding flags the file (ignores severity/confidence)
        inject_pass1_context: If True, inject Pass 1 findings into Pass 2 prompts
    """
    pass1_models: List[str] = field(default_factory=list)
    pass2_models: List[str] = field(default_factory=list)
    pass1_strategy: Literal["union", "majority_vote", "weighted_vote", "judge"] = "union"
    pass2_strategy: Literal["union", "majority_vote", "weighted_vote", "judge"] = "union"
    pass1_judge_model_id: Optional[str] = None
    pass2_judge_model_id: Optional[str] = None
    min_severity: Literal["low", "medium", "high", "critical"] = "low"
    min_confidence: float = 0.0
    flag_any_finding: bool = True
    inject_pass1_context: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pass1_models": self.pass1_models,
            "pass2_models": self.pass2_models,
            "pass1_strategy": self.pass1_strategy,
            "pass2_strategy": self.pass2_strategy,
            "pass1_judge_model_id": self.pass1_judge_model_id,
            "pass2_judge_model_id": self.pass2_judge_model_id,
            "min_severity": self.min_severity,
            "min_confidence": self.min_confidence,
            "flag_any_finding": self.flag_any_finding,
            "inject_pass1_context": self.inject_pass1_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CascadeConfig":
        """Create from dictionary."""
        return cls(
            pass1_models=data.get("pass1_models", []),
            pass2_models=data.get("pass2_models", []),
            pass1_strategy=data.get("pass1_strategy", "union"),
            pass2_strategy=data.get("pass2_strategy", "union"),
            pass1_judge_model_id=data.get("pass1_judge_model_id"),
            pass2_judge_model_id=data.get("pass2_judge_model_id"),
            min_severity=data.get("min_severity", "low"),
            min_confidence=data.get("min_confidence", 0.0),
            flag_any_finding=data.get("flag_any_finding", True),
            inject_pass1_context=data.get("inject_pass1_context", False),
        )


@dataclass
class PassResult:
    """Result from a single pass in cascade consensus.

    Attributes:
        pass_number: 1 or 2
        models_used: List of model IDs used in this pass
        strategy_used: Consensus strategy applied
        files_scanned: Number of files scanned
        files_flagged: Number of files with findings (Pass 1 only)
        findings_count: Total number of consensus findings
        duration_ms: Execution time in milliseconds
        consensus_findings: The consensus findings from this pass
        per_model_findings: Findings from each model
        flagged_files: Set of file paths flagged (Pass 1 only)
    """
    pass_number: int
    models_used: List[str]
    strategy_used: str
    files_scanned: int
    files_flagged: int = 0
    findings_count: int = 0
    duration_ms: int = 0
    consensus_findings: List[Finding] = field(default_factory=list)
    per_model_findings: Dict[str, List[Finding]] = field(default_factory=dict)
    flagged_files: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            "pass_number": self.pass_number,
            "models_used": self.models_used,
            "strategy_used": self.strategy_used,
            "files_scanned": self.files_scanned,
            "files_flagged": self.files_flagged,
            "findings_count": self.findings_count,
            "duration_ms": self.duration_ms,
            "consensus_findings": [f.to_dict() for f in self.consensus_findings],
            "per_model_findings": {
                model_id: [f.to_dict() for f in findings]
                for model_id, findings in self.per_model_findings.items()
            },
            "flagged_files": list(self.flagged_files),
        }


@dataclass
class CascadeResult:
    """Complete result from cascade consensus execution.

    Attributes:
        scan_id: Unique scan identifier
        config: The cascade configuration used
        pass1_result: Results from Pass 1 (triage)
        pass2_result: Results from Pass 2 (deep scan), None if skipped
        pass2_skipped: True if Pass 2 was skipped (no findings in Pass 1)
        final_findings: The final consensus findings (Pass 2 or Pass 1 if skipped)
        total_duration_ms: Total execution time
        created_at: When the scan was created
    """
    scan_id: str
    config: CascadeConfig
    pass1_result: PassResult
    pass2_result: Optional[PassResult] = None
    pass2_skipped: bool = False
    final_findings: List[Finding] = field(default_factory=list)
    total_duration_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            "scan_id": self.scan_id,
            "strategy": "cascade",
            "config": self.config.to_dict(),
            "pass1_result": self.pass1_result.to_dict(),
            "pass2_result": self.pass2_result.to_dict() if self.pass2_result else None,
            "pass2_skipped": self.pass2_skipped,
            "final_findings": [f.to_dict() for f in self.final_findings],
            "total_duration_ms": self.total_duration_ms,
            "created_at": self.created_at.isoformat(),
        }

    def to_scan_result(self) -> ScanResult:
        """Convert to standard ScanResult for compatibility."""
        # Combine per-model findings from both passes
        all_per_model: Dict[str, List[Finding]] = {}
        for model_id, findings in self.pass1_result.per_model_findings.items():
            all_per_model[f"pass1:{model_id}"] = findings
        if self.pass2_result:
            for model_id, findings in self.pass2_result.per_model_findings.items():
                all_per_model[f"pass2:{model_id}"] = findings

        return ScanResult(
            scan_id=self.scan_id,
            consensus_findings=self.final_findings,
            per_model_findings=all_per_model,
            scan_metadata={
                "strategy": "cascade",
                "cascade_result": self.to_dict(),
                "pass1_summary": {
                    "files_scanned": self.pass1_result.files_scanned,
                    "files_flagged": self.pass1_result.files_flagged,
                    "findings_count": self.pass1_result.findings_count,
                    "models": self.pass1_result.models_used,
                    "duration_ms": self.pass1_result.duration_ms,
                },
                "pass2_summary": {
                    "files_scanned": self.pass2_result.files_scanned if self.pass2_result else 0,
                    "findings_count": self.pass2_result.findings_count if self.pass2_result else 0,
                    "models": self.pass2_result.models_used if self.pass2_result else [],
                    "duration_ms": self.pass2_result.duration_ms if self.pass2_result else 0,
                    "skipped": self.pass2_skipped,
                } if not self.pass2_skipped else {"skipped": True},
                "total_findings": len(self.final_findings),
            },
            created_at=self.created_at,
        )


def identify_flagged_files(
    findings: List[Finding],
    config: CascadeConfig,
) -> Set[str]:
    """Identify files that should be flagged for Pass 2 based on findings.

    Args:
        findings: Consensus findings from Pass 1
        config: Cascade configuration with filtering rules

    Returns:
        Set of file paths that have findings meeting the criteria
    """
    if not findings:
        return set()

    flagged: Set[str] = set()
    min_severity_level = SEVERITY_ORDER.get(config.min_severity, 1)

    for finding in findings:
        # Check if finding meets criteria
        if config.flag_any_finding:
            flagged.add(finding.file)
        else:
            severity_level = SEVERITY_ORDER.get(finding.severity, 1)
            meets_severity = severity_level >= min_severity_level
            meets_confidence = finding.confidence >= config.min_confidence

            if meets_severity and meets_confidence:
                flagged.add(finding.file)

    return flagged


def get_pass1_context_for_file(
    file_path: str,
    pass1_findings: List[Finding],
) -> str:
    """Generate context string from Pass 1 findings for a specific file.

    This is used to inject Pass 1 context into Pass 2 prompts when
    inject_pass1_context is enabled.

    Args:
        file_path: The file being analyzed
        pass1_findings: All findings from Pass 1

    Returns:
        Context string describing Pass 1 findings for this file
    """
    file_findings = [f for f in pass1_findings if f.file == file_path]
    if not file_findings:
        return ""

    lines = ["Previous triage findings for this file:"]
    for finding in file_findings:
        lines.append(
            f"- {finding.cwe} ({finding.severity}): {finding.name} "
            f"at lines {finding.start_line}-{finding.end_line}"
        )
    return "\n".join(lines)
