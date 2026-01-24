import os
import logging
import zipfile
import tempfile
from typing import Dict, Optional, Iterable, Tuple, List
from flask import current_app

logger = logging.getLogger(__name__)


def _debug_scan_enabled() -> bool:
    value = os.environ.get("AEGIS_DEBUG_SCAN", "")
    return value.lower() not in {"0", "false", "no"}


def debug_scan_log(message: str) -> None:
    if not _debug_scan_enabled():
        return
    try:
        current_app.logger.info(message)
    except RuntimeError:
        print(message)


def allowed_file(filename: Optional[str]) -> bool:
    if filename is None:
        return False

    allowed_extensions = current_app.config.get("ALLOWED_EXTENSIONS", {"zip"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def extract_source_files(zip_path: str) -> Dict[str, str]:
    source_files = {}
    source_extensions = {
        "py",
        "js",
        "ts",
        "java",
        "cpp",
        "c",
        "cs",
        "php",
        "rb",
        "go",
        "rs",
        "sql",
        "sh",
        "bash",
        "ps1",
        "html",
        "css",
        "jsx",
        "tsx",
        "vue",
        "swift",
        "kt",
        "scala",
    }

    try:
        debug_scan_log(f"[scan-debug] extracting zip: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue

                file_path = file_info.filename

                if any(part.startswith(".") for part in file_path.split("/")):
                    continue

                if "." in file_path:
                    extension = file_path.rsplit(".", 1)[1].lower()
                    if extension in source_extensions:
                        try:
                            with zip_ref.open(file_info) as source_file:
                                content = source_file.read().decode(
                                    "utf-8", errors="ignore"
                                )
                                source_files[file_path] = content
                        except Exception as e:
                            debug_scan_log(f"[scan-debug] failed reading {file_path}: {e}")
                            continue

    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {e}")

    if not source_files:
        debug_scan_log(f"[scan-debug] no source files found in {zip_path}")
        raise ValueError("No source code files found in the ZIP archive")

    debug_scan_log(f"[scan-debug] extracted {len(source_files)} source files from {zip_path}")
    if _debug_scan_enabled():
        sample = list(source_files.keys())[:10]
        debug_scan_log(f"[scan-debug] sample files: {', '.join(sample)}")

    return source_files


def get_severity_color(severity: str) -> str:
    color_map = {
        "low": "success",
        "medium": "warning",
        "high": "danger",
        "critical": "dark",
    }
    return color_map.get(severity.lower(), "secondary")


def format_score(score: float) -> str:
    return f"{score:.1f}/10.0"


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    extension_map = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
        "cs": "csharp",
        "php": "php",
        "rb": "ruby",
        "go": "go",
        "rs": "rust",
        "sql": "sql",
        "sh": "bash",
        "bash": "bash",
        "ps1": "powershell",
        "html": "html",
        "css": "css",
        "vue": "vue",
        "swift": "swift",
        "kt": "kotlin",
        "scala": "scala",
    }

    if "." in file_path:
        extension = file_path.rsplit(".", 1)[1].lower()
        return extension_map.get(extension, "unknown")

    return "unknown"


def chunk_file_lines(content: str, chunk_size: int) -> Iterable[Tuple[str, int, int]]:
    """Yield (chunk_text, start_line, end_line) tuples for a file."""
    lines = content.split("\n")
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunk_content = "\n".join(chunk_lines)
        line_start = i + 1
        line_end = min(i + chunk_size, len(lines))
        yield chunk_content, line_start, line_end


# ============================================================================
# Device Selection Utilities
# ============================================================================

def _detect_cuda_available() -> bool:
    """Detect if CUDA is available on this system."""
    try:
        import torch
        return bool(hasattr(torch, "cuda") and torch.cuda.is_available())
    except Exception:
        return False


def _detect_mps_available() -> bool:
    """Detect if Apple Metal (MPS) is available on this system."""
    try:
        import torch
        return bool(
            hasattr(torch.backends, "mps")
            and hasattr(torch.backends.mps, "is_available")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def resolve_device(
    device_preference: Optional[List[str]] = None,
    env_override: Optional[str] = None,
    require_device: Optional[str] = None,
    allow_fallback: bool = True,
) -> str:
    """
    Resolve device string for model runtime.

    Priority:
    1. env_override or AEGIS_DEVICE environment variable
    2. require_device (if specified, must be available or error)
    3. device_preference list (try in order)
    4. Default fallback: ["cuda", "mps", "cpu"]

    Args:
        device_preference: Ordered list of preferred devices (e.g., ["cuda", "cpu"])
        env_override: Explicit device override (takes precedence)
        require_device: Required device (raises error if unavailable)
        allow_fallback: If True, fallback to CPU if preferred device unavailable

    Returns:
        Device string: "cuda", "cuda:0", "mps", or "cpu"

    Raises:
        RuntimeError: If required device is not available
    """
    # Check environment variable first
    if env_override is None:
        env_override = os.environ.get("AEGIS_DEVICE", "").strip()

    if env_override:
        env_override = env_override.lower()
        if env_override == "auto":
            env_override = None  # Fall through to auto-detection
        else:
            # Validate and return env override
            if env_override in ("gpu", "cuda"):
                if _detect_cuda_available():
                    return "cuda"
                elif allow_fallback:
                    logger.warning("AEGIS_DEVICE=cuda but CUDA unavailable, falling back to CPU")
                    return "cpu"
                else:
                    raise RuntimeError("AEGIS_DEVICE=cuda but CUDA is not available")
            elif env_override.startswith("cuda:"):
                if _detect_cuda_available():
                    return env_override
                elif allow_fallback:
                    logger.warning(f"AEGIS_DEVICE={env_override} but CUDA unavailable, falling back to CPU")
                    return "cpu"
                else:
                    raise RuntimeError(f"AEGIS_DEVICE={env_override} but CUDA is not available")
            elif env_override == "mps":
                if _detect_mps_available():
                    return "mps"
                elif allow_fallback:
                    logger.warning("AEGIS_DEVICE=mps but MPS unavailable, falling back to CPU")
                    return "cpu"
                else:
                    raise RuntimeError("AEGIS_DEVICE=mps but MPS is not available")
            elif env_override == "cpu":
                return "cpu"
            else:
                logger.warning(f"Unknown AEGIS_DEVICE value: {env_override}, using auto-detection")

    # Check required device
    if require_device:
        require_device = require_device.lower()
        if require_device in ("cuda", "gpu"):
            if not _detect_cuda_available():
                raise RuntimeError("CUDA required but not available on this system")
            return "cuda"
        elif require_device.startswith("cuda:"):
            if not _detect_cuda_available():
                raise RuntimeError(f"{require_device} required but CUDA not available")
            return require_device
        elif require_device == "mps":
            if not _detect_mps_available():
                raise RuntimeError("MPS required but not available on this system")
            return "mps"

    # Try device preference list
    if device_preference is None:
        device_preference = ["cuda", "mps", "cpu"]

    for pref in device_preference:
        pref_lower = str(pref).lower()

        if pref_lower in ("cuda", "gpu"):
            if _detect_cuda_available():
                return "cuda"
        elif pref_lower.startswith("cuda:"):
            if _detect_cuda_available():
                return pref_lower
        elif pref_lower == "mps":
            if _detect_mps_available():
                return "mps"
        elif pref_lower == "cpu":
            return "cpu"

    # Final fallback
    return "cpu"


def resolve_dtype(device: str, requested_dtype: Optional[str] = None, allow_half_cpu: bool = False) -> Optional[str]:
    """
    Resolve dtype for model runtime.

    Args:
        device: Target device ("cuda", "mps", "cpu")
        requested_dtype: Requested dtype ("bf16", "fp16", "fp32", etc.)
        allow_half_cpu: Allow half-precision on CPU (generally not recommended)

    Returns:
        Dtype string or None (use model default)
    """
    if requested_dtype is None:
        return None

    dtype_lower = requested_dtype.lower()

    # Normalize dtype names
    dtype_map = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }

    normalized_dtype = dtype_map.get(dtype_lower, dtype_lower)

    # Check if half-precision on CPU
    if device.startswith("cpu") and normalized_dtype in ("bfloat16", "float16"):
        if allow_half_cpu:
            logger.warning(f"Half-precision {normalized_dtype} on CPU may be slow")
            return normalized_dtype
        else:
            logger.warning(f"Half-precision {normalized_dtype} requested on CPU, falling back to float32")
            return "float32"

    return normalized_dtype
