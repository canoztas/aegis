"""Consensus engine for merging findings from multiple models."""

from aegis.consensus.engine import ConsensusEngine, CONSENSUS_STRATEGIES
from aegis.consensus.cascade import (
    CascadeConfig,
    CascadeResult,
    PassResult,
    identify_flagged_files,
    get_pass1_context_for_file,
    SEVERITY_ORDER,
)

__all__ = [
    "ConsensusEngine",
    "CONSENSUS_STRATEGIES",
    "CascadeConfig",
    "CascadeResult",
    "PassResult",
    "identify_flagged_files",
    "get_pass1_context_for_file",
    "SEVERITY_ORDER",
]
