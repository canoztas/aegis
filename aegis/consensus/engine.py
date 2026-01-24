"""Consensus engine for merging findings from multiple models."""
import hashlib
from typing import Dict, List, Literal, Optional, Any, Set
from aegis.data_models import Finding, ModelResponse
from aegis.prompt_builder import PromptBuilder

# Valid consensus strategies including the new cascade strategy
CONSENSUS_STRATEGIES = ["union", "majority_vote", "weighted_vote", "judge", "cascade"]


class ConsensusEngine:
    """Engine for merging findings from multiple models."""

    def __init__(self, prompt_builder: Optional[PromptBuilder] = None):
        """Initialize consensus engine."""
        self.prompt_builder = prompt_builder or PromptBuilder()

    def merge(
        self,
        model_responses: List[ModelResponse],
        strategy: Literal["union", "majority_vote", "weighted_vote", "judge", "cascade"] = "union",
        weights: Optional[Dict[str, float]] = None,
        judge_model: Optional[Any] = None,
        judge_request_params: Optional[Dict[str, Any]] = None,
    ) -> List[Finding]:
        """Merge findings from multiple models using specified strategy."""
        if not model_responses:
            return []

        # Filter out failed responses
        valid_responses = [r for r in model_responses if not r.error]

        if not valid_responses:
            return []

        if strategy == "union":
            return self._union_strategy(valid_responses)
        elif strategy == "majority_vote":
            return self._majority_vote_strategy(valid_responses)
        elif strategy == "weighted_vote":
            return self._weighted_vote_strategy(valid_responses, weights or {})
        elif strategy == "judge":
            if judge_model is None:
                raise ValueError("judge_model required for judge strategy")
            return self._judge_strategy(
                valid_responses, judge_model, judge_request_params or {}
            )
        elif strategy == "cascade":
            # Cascade strategy is handled at the scan service level, not here.
            # If called directly, fall back to union for the current pass.
            return self._union_strategy(valid_responses)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _union_strategy(self, responses: List[ModelResponse]) -> List[Finding]:
        """Union strategy: merge all findings and deduplicate."""
        all_findings = []
        for response in responses:
            all_findings.extend(response.findings)

        return self._deduplicate_findings(all_findings)

    def _majority_vote_strategy(self, responses: List[ModelResponse]) -> List[Finding]:
        """Majority vote: require >50% model agreement."""
        if len(responses) == 1:
            return responses[0].findings

        # Group findings by normalized key
        finding_groups: Dict[str, List[Finding]] = {}

        for response in responses:
            for finding in response.findings:
                key = self._normalize_finding_key(finding)
                if key not in finding_groups:
                    finding_groups[key] = []
                finding_groups[key].append(finding)

        # Keep only findings with majority agreement
        threshold = len(responses) / 2
        consensus_findings = []

        for key, findings in finding_groups.items():
            if len(findings) > threshold:
                # Merge findings: take max confidence, most informative message
                merged = self._merge_finding_group(findings)
                consensus_findings.append(merged)

        return consensus_findings

    def _weighted_vote_strategy(
        self, responses: List[ModelResponse], weights: Dict[str, float]
    ) -> List[Finding]:
        """Weighted vote: same as majority but with model weights."""
        if len(responses) == 1:
            return responses[0].findings

        # Group findings by normalized key
        finding_groups: Dict[str, List[Finding]] = {}

        for response in responses:
            weight = weights.get(response.model_id, 1.0)
            for finding in response.findings:
                key = self._normalize_finding_key(finding)
                if key not in finding_groups:
                    finding_groups[key] = []
                finding_groups[key].append((finding, weight))

        # Calculate weighted scores
        consensus_findings = []
        total_weight = sum(weights.get(r.model_id, 1.0) for r in responses)

        for key, weighted_findings in finding_groups.items():
            weighted_sum = sum(weight for _, weight in weighted_findings)
            if weighted_sum > (total_weight / 2):  # Majority by weight
                findings = [f for f, _ in weighted_findings]
                merged = self._merge_finding_group(findings)
                consensus_findings.append(merged)

        return consensus_findings

    def _judge_strategy(
        self,
        responses: List[ModelResponse],
        judge_model: Any,
        request_params: Dict[str, Any],
    ) -> List[Finding]:
        """Judge strategy: use a judge model to merge findings."""
        # Collect all findings
        all_findings = []
        for response in responses:
            all_findings.extend(response.findings)

        if not all_findings:
            return []

        # Group by file for judge processing
        findings_by_file: Dict[str, List[Finding]] = {}
        for finding in all_findings:
            if finding.file not in findings_by_file:
                findings_by_file[finding.file] = []
            findings_by_file[finding.file].append(finding)

        # Process each file with judge
        consensus_findings = []

        # Support legacy judge adapters with .predict(), or ModelRecord (new registry).
        def _run_judge_predict(prompt: str, file_path: str, candidate_data: List[Dict[str, Any]]):
            from aegis.data_models import ModelRequest

            judge_request = ModelRequest(
                code_context=prompt,
                file_path=file_path,
                language=request_params.get("language", "unknown"),
                prompt_template_id="judge_consensus",
            )
            return judge_model.predict(judge_request)

        def _run_judge_runtime(file_path: str, candidate_data: List[Dict[str, Any]]):
            import asyncio
            import json
            from aegis.models.runtime_manager import DEFAULT_RUNTIME_MANAGER
            from aegis.models.schema import ModelRole
            from aegis.models.engine import _candidate_to_finding

            runtime = DEFAULT_RUNTIME_MANAGER.get_runtime(judge_model)
            context = {
                "file_path": file_path,
                "findings_json": json.dumps(candidate_data, indent=2),
            }
            result = asyncio.run(runtime.run("", context, role=ModelRole.JUDGE))
            return [_candidate_to_finding(c) for c in result.findings]

        has_predict = hasattr(judge_model, "predict")

        for file_path, findings in findings_by_file.items():
            # Convert findings to dict for judge
            candidate_data = [f.to_dict() for f in findings]

            # Build judge prompt
            language = request_params.get("language", "unknown")
            repo_name = request_params.get("repo_name")

            prompt = self.prompt_builder.build_judge_prompt(
                candidate_findings=candidate_data,
                file_path=file_path,
                language=language,
                repo_name=repo_name,
            )
            try:
                if has_predict:
                    judge_response = _run_judge_predict(prompt, file_path, candidate_data)
                    if not judge_response.error:
                        consensus_findings.extend(judge_response.findings)
                    else:
                        consensus_findings.extend(self._deduplicate_findings(findings))
                else:
                    consensus_findings.extend(_run_judge_runtime(file_path, candidate_data))
            except Exception:
                # Fallback to union if judge fails
                consensus_findings.extend(self._deduplicate_findings(findings))

        return consensus_findings

    def _normalize_finding_key(self, finding: Finding) -> str:
        """Create normalized key for finding deduplication."""
        # Normalize line ranges to buckets (±2 lines)
        line_bucket_start = (finding.start_line // 5) * 5
        line_bucket_end = ((finding.end_line + 4) // 5) * 5

        # Normalize message (lowercase, remove extra whitespace)
        normalized_message = " ".join(finding.message.lower().split())

        # Create key
        key_parts = [
            finding.cwe,
            finding.file,
            str(line_bucket_start),
            str(line_bucket_end),
            normalized_message[:50],  # First 50 chars
        ]

        key_string = "|".join(key_parts)
        return hashlib.sha1(key_string.encode()).hexdigest()

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Deduplicate findings based on normalized keys."""
        seen_keys: Dict[str, Finding] = {}

        for finding in findings:
            key = self._normalize_finding_key(finding)

            if key not in seen_keys:
                seen_keys[key] = finding
            else:
                # Merge: keep the one with higher confidence or more informative message
                existing = seen_keys[key]
                if (
                    finding.confidence > existing.confidence
                    or len(finding.message) > len(existing.message)
                ):
                    seen_keys[key] = finding

        return list(seen_keys.values())

    def _merge_finding_group(self, findings: List[Finding]) -> Finding:
        """Merge a group of similar findings into one."""
        if not findings:
            raise ValueError("Cannot merge empty finding group")

        if len(findings) == 1:
            return findings[0]

        # Take the first finding as base
        base = findings[0]

        # Merge attributes
        max_confidence = max(f.confidence for f in findings)
        longest_message = max(findings, key=lambda f: len(f.message))

        # Use most precise line range
        min_start = min(f.start_line for f in findings)
        max_end = max(f.end_line for f in findings)

        return Finding(
            name=base.name,
            severity=base.severity,  # Keep original
            cwe=base.cwe,
            file=base.file,
            start_line=min_start,
            end_line=max_end,
            message=longest_message.message,
            confidence=max_confidence,
            fingerprint=base.fingerprint,  # Keep original fingerprint
        )

