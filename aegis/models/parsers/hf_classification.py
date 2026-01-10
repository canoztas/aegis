"""Parser for HuggingFace text-classification outputs."""

import logging
from typing import Any, Dict, Optional, List

from aegis.models.schema import FindingCandidate, TriageSignal, ParserResult
from aegis.models.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class HFTextClassificationParser(BaseParser):
    """
    Parser for HuggingFace text-classification pipeline outputs.

    Expected input format (from transformers pipeline):
    [
      {"label": "LABEL_1", "score": 0.95},
      {"label": "LABEL_0", "score": 0.05}
    ]

    Configuration:
    {
      "positive_labels": ["LABEL_1", "VULNERABLE", "INSECURE"],
      "negative_labels": ["LABEL_0", "SAFE", "SECURE"],
      "threshold": 0.5,
      "severity_high_threshold": 0.85,
      "severity_medium_threshold": 0.65
    }
    """

    def parse(self, raw_output: Any, context: Optional[Dict[str, Any]] = None) -> ParserResult:
        """
        Parse classification output into triage signal or findings.

        Args:
            raw_output: Output from HF text-classification pipeline
            context: Context with file_path, snippet, etc.

        Returns:
            ParserResult with triage signal and/or findings
        """
        context = context or {}
        errors = []

        # Get configuration
        positive_labels = self.config.get("positive_labels", ["LABEL_1", "VULNERABLE", "INSECURE"])
        negative_labels = self.config.get("negative_labels", ["LABEL_0", "SAFE", "SECURE"])
        threshold = self.config.get("threshold", 0.5)
        severity_high_threshold = self.config.get("severity_high_threshold", 0.85)
        severity_medium_threshold = self.config.get("severity_medium_threshold", 0.65)

        # Validate input
        if not isinstance(raw_output, list):
            errors.append(f"Expected list output, got {type(raw_output)}")
            return ParserResult(parse_errors=errors)

        if not raw_output:
            errors.append("Empty classification output")
            return ParserResult(parse_errors=errors)

        # Find highest confidence prediction
        predictions = sorted(raw_output, key=lambda x: x.get("score", 0), reverse=True)
        top_prediction = predictions[0]

        label = top_prediction.get("label", "").upper()
        score = top_prediction.get("score", 0.0)

        # Determine if suspicious
        is_suspicious = label in [l.upper() for l in positive_labels] and score >= threshold

        # Create triage signal
        triage_signal = TriageSignal(
            is_suspicious=is_suspicious,
            confidence=score,
            labels=[label],
            suspicious_chunks=[context] if is_suspicious else [],
            metadata={
                "all_predictions": predictions,
                "threshold": threshold,
            },
        )

        # Optionally create finding if suspicious
        findings = []
        if is_suspicious and context.get("file_path"):
            # Determine severity based on confidence
            if score >= severity_high_threshold:
                severity = "high"
            elif score >= severity_medium_threshold:
                severity = "medium"
            else:
                severity = "low"

            finding = FindingCandidate(
                file_path=context.get("file_path", "unknown"),
                line_start=context.get("line_start", 1),
                line_end=context.get("line_end"),
                snippet=context.get("snippet", ""),
                category="potential_vulnerability",  # Generic category
                severity=severity,
                description=f"Code classified as {label} with confidence {score:.2%}",
                confidence=score,
                metadata={
                    "label": label,
                    "all_predictions": predictions,
                    "source": "hf_classification",
                },
            )
            findings.append(finding)

        return ParserResult(
            findings=findings,
            triage_signal=triage_signal,
            parse_errors=errors,
        )
