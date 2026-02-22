"""Core schema definitions for model management."""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Canonical model types supported by Aegis."""
    OLLAMA_LOCAL = "ollama_local"
    HF_LOCAL = "hf_local"
    OPENAI_COMPATIBLE = "openai_compatible"  # Custom OpenAI-compatible endpoints
    OPENAI_CLOUD = "openai_cloud"  # Official OpenAI API (GPT-4, GPT-3.5-Turbo)
    ANTHROPIC_CLOUD = "anthropic_cloud"  # Anthropic API (Claude 3)
    GOOGLE_CLOUD = "google_cloud"  # Google Generative AI (Gemini)
    TOOL_ML = "tool_ml"  # Classic/legacy ML tools (SySeVR, etc.)
    CLAUDE_CODE = "claude_code"  # Claude Code CLI subprocess-based scanning


class ModelRole(str, Enum):
    """Roles that models can fulfill in scan pipelines."""
    TRIAGE = "triage"           # Initial classification/filtering
    DEEP_SCAN = "deep_scan"     # Detailed vulnerability analysis
    JUDGE = "judge"             # Final verdict/severity assessment
    EXPLAIN = "explain"         # Human-readable explanations
    CUSTOM = "custom"           # User-defined roles


# Legacy role mapping for backward compatibility
# Maps old role names to current ModelRole enums
LEGACY_ROLE_MAPPING = {
    "scan": ModelRole.DEEP_SCAN,  # Old 'scan' maps to deep_scan
    "triage": ModelRole.TRIAGE,
    "deep_scan": ModelRole.DEEP_SCAN,
    "judge": ModelRole.JUDGE,
    "explain": ModelRole.EXPLAIN,
    "custom": ModelRole.CUSTOM,
}


def parse_role(role_str: str) -> ModelRole:
    """
    Parse role string with legacy mapping support.

    Args:
        role_str: Role string to parse

    Returns:
        ModelRole enum value

    Raises:
        ValueError: If role string is not recognized
    """
    try:
        return ModelRole(role_str)
    except ValueError:
        mapped = LEGACY_ROLE_MAPPING.get(role_str.lower())
        if mapped:
            return mapped
        raise ValueError(f"Invalid role: {role_str}")


class ModelStatus(str, Enum):
    """Registration status of a model."""
    REGISTERED = "registered"   # Active and usable
    DISABLED = "disabled"        # Registered but not active
    UNAVAILABLE = "unavailable"  # Registered but model not found


class ModelAvailability(str, Enum):
    """Runtime availability of a model at its provider."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class DiscoveredModel(BaseModel):
    """A model discovered from a provider (not yet registered)."""
    name: str = Field(..., description="Model name at provider")
    model_type: ModelType
    provider: str = Field(..., description="Provider identifier (e.g., 'ollama', 'huggingface')")
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")

    class Config:
        use_enum_values = True


class ModelRecord(BaseModel):
    """A registered model in the database."""
    id: Optional[int] = None
    model_id: str = Field(..., description="Unique identifier in Aegis (e.g., 'ollama:qwen2.5-coder')")
    model_type: ModelType
    provider_id: str = Field(..., description="Provider identifier")
    provider_config: Dict[str, Any] = Field(default_factory=dict, description="Provider-level configuration")
    model_name: str = Field(..., description="Actual model name at provider")
    display_name: str = Field(..., description="Human-readable name")
    roles: List[ModelRole] = Field(default_factory=list, description="Roles this model can fulfill")
    parser_id: Optional[str] = Field(None, description="Parser to use for output (e.g., 'json_schema', 'hf_classification')")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Model-specific settings (temp, max_tokens, etc.)")
    status: ModelStatus = ModelStatus.REGISTERED
    availability: ModelAvailability = ModelAvailability.UNKNOWN
    last_checked: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class FindingCandidate(BaseModel):
    """A potential security finding from a model (pre-validation)."""
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    snippet: str
    title: Optional[str] = None
    category: str = Field(..., description="Vulnerability category (e.g., 'sql_injection', 'xss')")
    cwe: Optional[str] = Field(None, description="CWE identifier if provided by the model")
    severity: str = Field(..., description="Severity level (e.g., 'high', 'medium', 'low', 'info')")
    description: str
    recommendation: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Model confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    class Config:
        use_enum_values = True


class TriageSignal(BaseModel):
    """Output from triage models (lightweight classification)."""
    is_suspicious: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    labels: List[str] = Field(default_factory=list)
    suspicious_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Chunks needing deep scan")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParserResult(BaseModel):
    """Result from parsing model output."""
    findings: List[FindingCandidate] = Field(default_factory=list)
    triage_signal: Optional[TriageSignal] = None
    parse_errors: List[str] = Field(default_factory=list)
    raw_output: Optional[str] = Field(None, description="Original model output (for debugging)")
