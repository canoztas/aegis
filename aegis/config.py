import os
from pathlib import Path
from typing import Any


# Determine project root (aegis package parent directory)
_PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    SECRET_KEY: str = (
        os.environ.get("SECRET_KEY") or "aegis-secret-key-change-in-production"
    )
    UPLOAD_FOLDER: str = os.path.join(os.getcwd(), "uploads")
    MAX_CONTENT_LENGTH_IN_BYTES: int = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {"zip"}

    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL") or "gpt-oss:120b-cloud"

    # HuggingFace models cache directory (under project root)
    MODELS_CACHE_DIR: str = os.environ.get("AEGIS_MODELS_CACHE") or str(
        _PROJECT_ROOT / "models_cache"
    )

    @staticmethod
    def init_app(app: Any) -> None:
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        # Ensure models cache directory exists
        os.makedirs(Config.MODELS_CACHE_DIR, exist_ok=True)
