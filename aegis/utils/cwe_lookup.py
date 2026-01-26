"""CWE lookup utilities for enriching findings with CWE names and metadata.

This module provides functions to look up CWE information from a local database,
enabling richer vulnerability reports with human-readable CWE names.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy-loaded CWE database
_CWE_DB: Dict[str, Any] = {}
_AEGIS_DATA = Path(__file__).parent.parent / "data"


def _load_cwe_database() -> Dict[str, Any]:
    """
    Load CWE database on first use (lazy loading).

    Returns:
        Dictionary mapping CWE IDs to their metadata.
    """
    global _CWE_DB
    if not _CWE_DB:
        db_path = _AEGIS_DATA / "cwe_database.json"
        if db_path.exists():
            try:
                _CWE_DB = json.loads(db_path.read_text(encoding="utf-8"))
                logger.debug(f"Loaded CWE database with {len(_CWE_DB)} entries")
            except Exception as e:
                logger.warning(f"Failed to load CWE database: {e}")
                _CWE_DB = {}
        else:
            logger.debug(f"CWE database not found at {db_path}")
    return _CWE_DB


def normalize_cwe_id(cwe_id: str) -> str:
    """
    Normalize CWE ID to standard format (CWE-XXX).

    Args:
        cwe_id: CWE identifier in various formats (e.g., "119", "CWE-119", "cwe-119")

    Returns:
        Normalized CWE ID (e.g., "CWE-119")
    """
    if not cwe_id:
        return cwe_id

    cwe_id = str(cwe_id).strip().upper()

    # Already in correct format
    if cwe_id.startswith("CWE-"):
        return cwe_id

    # Just the number
    if cwe_id.isdigit():
        return f"CWE-{cwe_id}"

    # Handle "CWE119" without hyphen
    if cwe_id.startswith("CWE") and cwe_id[3:].isdigit():
        return f"CWE-{cwe_id[3:]}"

    return cwe_id


def get_cwe_info(cwe_id: str) -> Optional[Dict[str, Any]]:
    """
    Get CWE metadata by ID.

    Args:
        cwe_id: CWE identifier (e.g., "CWE-119", "119")

    Returns:
        Dictionary with CWE metadata (name, full_name, severity, category) or None
    """
    db = _load_cwe_database()
    normalized = normalize_cwe_id(cwe_id)
    return db.get(normalized)


def get_cwe_name(cwe_id: str, short: bool = True) -> Optional[str]:
    """
    Get CWE name by ID.

    Args:
        cwe_id: CWE identifier
        short: If True, return short name; if False, return full name

    Returns:
        CWE name string or None if not found
    """
    info = get_cwe_info(cwe_id)
    if info:
        return info.get("name") if short else info.get("full_name", info.get("name"))
    return None


def get_cwe_severity(cwe_id: str) -> Optional[str]:
    """
    Get CWE severity level.

    Args:
        cwe_id: CWE identifier

    Returns:
        Severity string (critical, high, medium, low) or None
    """
    info = get_cwe_info(cwe_id)
    if info:
        return info.get("severity")
    return None


def get_cwe_category(cwe_id: str) -> Optional[str]:
    """
    Get CWE category.

    Args:
        cwe_id: CWE identifier

    Returns:
        Category string (e.g., "memory", "injection") or None
    """
    info = get_cwe_info(cwe_id)
    if info:
        return info.get("category")
    return None


def format_cwe(cwe_id: str, include_name: bool = True) -> str:
    """
    Format CWE for display.

    Args:
        cwe_id: CWE identifier
        include_name: If True, include the CWE name

    Returns:
        Formatted string like "CWE-119: Buffer Overflow" or just "CWE-119"
    """
    normalized = normalize_cwe_id(cwe_id)

    if not include_name:
        return normalized

    name = get_cwe_name(cwe_id)
    if name:
        return f"{normalized}: {name}"

    return normalized


def format_cwe_full(cwe_id: str) -> str:
    """
    Format CWE with full description.

    Args:
        cwe_id: CWE identifier

    Returns:
        Formatted string with full name, or just the ID if not found
    """
    normalized = normalize_cwe_id(cwe_id)
    full_name = get_cwe_name(cwe_id, short=False)

    if full_name:
        return f"{normalized}: {full_name}"

    return normalized
