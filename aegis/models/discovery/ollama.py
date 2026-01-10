"""Ollama model discovery client."""

import logging
from typing import List, Optional
import httpx

from aegis.models.schema import DiscoveredModel, ModelType

logger = logging.getLogger(__name__)


class OllamaDiscoveryClient:
    """
    Client for discovering models from local Ollama installation.

    Uses Ollama HTTP API to query available models.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize discovery client.

        Args:
            base_url: Ollama API base URL
        """
        self.base_url = base_url.rstrip("/")
        self._cache: Optional[List[DiscoveredModel]] = None
        self._cache_timeout = 60  # seconds

    async def discover_models(self, force_refresh: bool = False) -> List[DiscoveredModel]:
        """
        Discover available Ollama models.

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of discovered models

        Raises:
            httpx.HTTPError: If Ollama API is unreachable
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

            models = []
            for model_data in data.get("models", []):
                model = DiscoveredModel(
                    name=model_data["name"],
                    model_type=ModelType.OLLAMA_LOCAL,
                    provider="ollama",
                    size_bytes=model_data.get("size", 0),
                    metadata={
                        "digest": model_data.get("digest"),
                        "modified_at": model_data.get("modified_at"),
                        "details": model_data.get("details", {}),
                    },
                )
                models.append(model)

            self._cache = models
            logger.info(f"Discovered {len(models)} Ollama models")
            return models

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover Ollama models: {e}")
            raise

    def discover_models_sync(self, force_refresh: bool = False) -> List[DiscoveredModel]:
        """
        Synchronous version of discover_models.

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of discovered models
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

            models = []
            for model_data in data.get("models", []):
                model = DiscoveredModel(
                    name=model_data["name"],
                    model_type=ModelType.OLLAMA_LOCAL,
                    provider="ollama",
                    size_bytes=model_data.get("size", 0),
                    metadata={
                        "digest": model_data.get("digest"),
                        "modified_at": model_data.get("modified_at"),
                        "details": model_data.get("details", {}),
                    },
                )
                models.append(model)

            self._cache = models
            logger.info(f"Discovered {len(models)} Ollama models")
            return models

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover Ollama models: {e}")
            raise

    def clear_cache(self):
        """Clear the discovery cache."""
        self._cache = None
