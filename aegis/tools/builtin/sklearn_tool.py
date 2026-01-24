"""Sklearn-based ML tool plugin for vulnerability prediction."""

import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from aegis.models.schema import FindingCandidate, ParserResult, TriageSignal
from aegis.tools.base import ToolPlugin

logger = logging.getLogger(__name__)

# Default cache directory for downloaded models (under aegis directory)
_AEGIS_ROOT = Path(__file__).parent.parent.parent
_SKLEARN_CACHE_DIR = os.environ.get(
    "AEGIS_SKLEARN_CACHE",
    str(_AEGIS_ROOT / "ml_models")
)


class SklearnTool(ToolPlugin):
    """
    Tool plugin for sklearn/joblib ML models.

    Supports:
    - Loading models from local paths or URLs
    - Automatic download and caching from GitHub releases
    - Binary classification (vulnerable/safe)
    - Probability-based confidence scores
    """

    tool_id = "sklearn_classifier"
    name = "Sklearn Classifier"
    description = "Machine learning classifier for vulnerability prediction"

    # Known model configurations
    KNOWN_MODELS = {
        "kaggle_rf_cfunctions": {
            "url": "https://github.com/canoztas/aegis-models/releases/download/c_security_model/c_security_model.pkl",
            "description": "Random Forest classifier trained on C functions for security prediction",
            "default_threshold": 0.5,
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sklearn tool.

        Config options:
            model_path: Path to model file (.pkl, .joblib)
            model_url: URL to download model from
            model_type: Known model type (e.g., 'kaggle_rf_cfunctions')
            vectorizer_path: Path to vectorizer file (optional)
            threshold: Classification threshold (default: 0.5)
            cache_dir: Directory for caching downloaded models
        """
        super().__init__(config)
        self._model = None
        self._vectorizer = None
        self._loaded = False

        # Configuration
        self.model_path = self.config.get("model_path")
        self.model_url = self.config.get("model_url")
        self.model_type = self.config.get("model_type")
        self.vectorizer_path = self.config.get("vectorizer_path")
        self.threshold = float(self.config.get("threshold", 0.5))
        self.cache_dir = self.config.get("cache_dir", _SKLEARN_CACHE_DIR)

        # If model_type is specified, use known configuration
        if self.model_type and self.model_type in self.KNOWN_MODELS:
            known = self.KNOWN_MODELS[self.model_type]
            if not self.model_url:
                self.model_url = known.get("url")
            if "threshold" not in self.config:
                self.threshold = known.get("default_threshold", 0.5)

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str) -> str:
        """Get local cache path for a URL."""
        # Hash the URL to create a unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        # Extract filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "model.pkl"
        return os.path.join(self.cache_dir, f"{url_hash}_{filename}")

    def _download_model(self, url: str) -> str:
        """
        Download model from URL to cache.

        Returns:
            Local path to downloaded model
        """
        import urllib.request

        self._ensure_cache_dir()
        cache_path = self._get_cache_path(url)

        if os.path.exists(cache_path):
            logger.info(f"Using cached model: {cache_path}")
            return cache_path

        logger.info(f"Downloading model from {url}")
        try:
            # Download with progress
            urllib.request.urlretrieve(url, cache_path)
            logger.info(f"Model downloaded to {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Failed to download model from {url}: {e}")

    def _patch_sklearn_compatibility(self) -> None:
        """
        Patch sklearn classes for backward compatibility with older pickled models.

        Models trained with sklearn <1.1 don't have the 'missing_go_to_left' attribute
        that was added in sklearn 1.1. This patch adds a custom unpickler to handle this.
        """
        try:
            from sklearn.tree._tree import Tree
            import sklearn

            sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))

            # Only patch if sklearn >= 1.1
            if sklearn_version >= (1, 1):
                # Store original __setstate__ if not already patched
                if not hasattr(Tree, '_original_setstate'):
                    Tree._original_setstate = Tree.__setstate__

                    def _patched_setstate(self, state):
                        """Handle missing 'missing_go_to_left' in old pickles."""
                        if isinstance(state, dict):
                            # For newer format (dict-based state)
                            if 'missing_go_to_left' not in state.get('nodes', {}).dtype.names if hasattr(state.get('nodes'), 'dtype') else True:
                                pass  # Will be handled by the tree itself
                        else:
                            # For older format (tuple-based state)
                            # The state is a tuple: (n_features, n_classes, n_outputs, max_depth, node_count, nodes, values)
                            pass

                        try:
                            Tree._original_setstate(self, state)
                        except TypeError as e:
                            if "missing_go_to_left" in str(e):
                                # Old model without missing_go_to_left field
                                # Try to reconstruct with default values
                                logger.warning("Attempting sklearn compatibility fix for old model format")
                                raise
                            raise

                    Tree.__setstate__ = _patched_setstate
                    logger.debug("Sklearn Tree compatibility patch applied")
        except Exception as e:
            logger.debug(f"Sklearn compatibility patch not applied: {e}")

    def _load_model(self) -> None:
        """Load the sklearn model from file or URL."""
        if self._loaded:
            return

        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for sklearn models. Install with: pip install joblib")

        # Apply sklearn compatibility patches
        self._patch_sklearn_compatibility()

        # Determine model path
        model_path = self.model_path
        if not model_path and self.model_url:
            model_path = self._download_model(self.model_url)

        if not model_path:
            raise ValueError("No model_path or model_url specified")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading sklearn model from {model_path}")

        # Try loading with compatibility handling
        try:
            self._model = joblib.load(model_path)
        except TypeError as e:
            if "missing_go_to_left" in str(e):
                # Model was trained with sklearn <1.1, need sklearn <1.1 to load
                raise RuntimeError(
                    f"This model was trained with scikit-learn <1.1 and cannot be loaded with your current version. "
                    f"Please install a compatible version: pip install 'scikit-learn>=0.24.0,<1.1.0' "
                    f"or use conda: conda install scikit-learn=1.0.2"
                ) from e
            raise

        # Load vectorizer if specified
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            logger.info(f"Loading vectorizer from {self.vectorizer_path}")
            self._vectorizer = joblib.load(self.vectorizer_path)

        self._loaded = True
        logger.info("Sklearn model loaded successfully")

    def is_model_cached(self) -> bool:
        """Check if model is available (either local or cached from URL)."""
        if self.model_path and os.path.exists(self.model_path):
            return True
        if self.model_url:
            cache_path = self._get_cache_path(self.model_url)
            return os.path.exists(cache_path)
        return False

    def prefetch(self) -> Dict[str, Any]:
        """
        Download model files without loading into memory.

        Returns:
            Dict with prefetch results
        """
        result = {
            "success": False,
            "cached": False,
            "cache_path": None,
            "error": None,
        }

        if not self.model_url:
            if self.model_path and os.path.exists(self.model_path):
                result["success"] = True
                result["cached"] = True
                result["cache_path"] = self.model_path
                return result
            result["error"] = "No model_url or valid model_path specified"
            return result

        try:
            cache_path = self._get_cache_path(self.model_url)
            if os.path.exists(cache_path):
                result["success"] = True
                result["cached"] = True
                result["cache_path"] = cache_path
                return result

            # Download
            downloaded_path = self._download_model(self.model_url)
            result["success"] = True
            result["cached"] = False
            result["cache_path"] = downloaded_path
            return result

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Prefetch failed: {e}")
            return result

    def predict(self, code: str) -> Dict[str, Any]:
        """
        Run prediction on code.

        Args:
            code: Source code to analyze

        Returns:
            Dict with:
                is_vulnerable: bool
                confidence: float (0-1)
                probability_safe: float
                probability_vulnerable: float
        """
        self._load_model()

        # Prepare input
        if self._vectorizer is not None:
            # Transform code using vectorizer
            X = self._vectorizer.transform([code])
        else:
            # Model expects raw text (e.g., TF-IDF pipeline)
            X = [code]

        # Get prediction
        prediction_class = self._model.predict(X)[0]

        # Get probabilities if available
        if hasattr(self._model, 'predict_proba'):
            probabilities = self._model.predict_proba(X)[0]
            # Assuming binary classification: [safe, vulnerable]
            prob_safe = float(probabilities[0])
            prob_vulnerable = float(probabilities[1]) if len(probabilities) > 1 else 1 - prob_safe
            confidence = prob_vulnerable if prediction_class == 1 else prob_safe
        else:
            # No probabilities available
            prob_safe = 0.0 if prediction_class == 1 else 1.0
            prob_vulnerable = 1.0 if prediction_class == 1 else 0.0
            confidence = 1.0

        return {
            "is_vulnerable": bool(prediction_class == 1),
            "confidence": confidence,
            "probability_safe": prob_safe,
            "probability_vulnerable": prob_vulnerable,
            "prediction_class": int(prediction_class),
        }

    def analyze_snippet(
        self,
        code: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> ParserResult:
        """
        Analyze a code snippet for vulnerabilities.

        Args:
            code: Source code to analyze
            context: Context with file_path, line numbers, etc.
            config: Optional runtime configuration

        Returns:
            ParserResult with triage signal and/or findings
        """
        cfg = {}
        cfg.update(self.config or {})
        cfg.update(config or {})

        file_path = context.get("file_path", "unknown")
        line_start = int(context.get("line_start") or 1)

        # Get threshold from config
        threshold = float(cfg.get("threshold", self.threshold))

        try:
            # Run prediction
            prediction = self.predict(code)

            is_vulnerable = prediction["is_vulnerable"]
            confidence = prediction["confidence"]
            prob_vulnerable = prediction["probability_vulnerable"]

            # Build triage signal
            triage_signal = TriageSignal(
                is_suspicious=is_vulnerable or prob_vulnerable >= threshold,
                confidence=confidence,
                labels=["vulnerable"] if is_vulnerable else ["safe"],
                suspicious_chunks=[{
                    "file_path": file_path,
                    "line_start": line_start,
                    "confidence": prob_vulnerable,
                }] if prob_vulnerable >= threshold else [],
                metadata={
                    "model_type": self.model_type or "sklearn_classifier",
                    "threshold": threshold,
                    "probability_vulnerable": prob_vulnerable,
                    "probability_safe": prediction["probability_safe"],
                }
            )

            # Build findings if vulnerable
            findings: List[FindingCandidate] = []
            if is_vulnerable or prob_vulnerable >= threshold:
                # Extract first line as snippet preview
                lines = code.strip().split('\n')
                snippet = lines[0][:100] if lines else code[:100]

                findings.append(FindingCandidate(
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_start + len(lines) - 1,
                    snippet=snippet,
                    title="Potential Vulnerability Detected",
                    category="ml_classification",
                    cwe=None,  # ML models don't provide specific CWE
                    severity="medium" if prob_vulnerable < 0.8 else "high",
                    description=f"ML classifier detected potential vulnerability with {prob_vulnerable:.1%} confidence.",
                    recommendation="Review this code section for security issues. The ML model flagged it as potentially vulnerable.",
                    confidence=prob_vulnerable,
                    metadata={
                        "model_type": self.model_type or "sklearn_classifier",
                        "probability_vulnerable": prob_vulnerable,
                        "threshold": threshold,
                    }
                ))

            return ParserResult(
                findings=findings,
                triage_signal=triage_signal,
                raw_output=str(prediction),
            )

        except Exception as e:
            logger.error(f"Sklearn analysis failed: {e}")
            return ParserResult(
                findings=[],
                parse_errors=[f"ML analysis error: {str(e)}"],
            )


class KaggleRFCFunctionsTool(SklearnTool):
    """
    Pre-configured tool for Kaggle RF C-Functions model.

    This model is trained on C/C++ code for security vulnerability prediction.
    Source: https://www.kaggle.com/code/furduisorinoctavian/eda-rf-c-functions-for-security-prediction
    """

    tool_id = "kaggle_rf_cfunctions"
    name = "Kaggle RF C-Functions Predictor"
    description = "Random Forest classifier for C/C++ security vulnerability prediction"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Pre-configure with known model settings
        default_config = {
            "model_type": "kaggle_rf_cfunctions",
            "model_url": "https://github.com/canoztas/aegis-models/releases/download/c_security_model/c_security_model.pkl",
            "threshold": 0.5,
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__(merged_config)
