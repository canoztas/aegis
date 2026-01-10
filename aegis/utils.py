import os
import zipfile
import tempfile
from typing import Dict, Optional, Iterable, Tuple
from flask import current_app


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
                            print(f"Error reading {file_path}: {e}")
                            continue

    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {e}")

    if not source_files:
        raise ValueError("No source code files found in the ZIP archive")

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
