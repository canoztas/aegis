"""Consensus engine for merging findings from multiple models."""
import hashlib
from typing import Dict, List, Literal, Optional, Any, Set, Tuple
from aegis.data_models import Finding, ModelResponse
from aegis.prompt_builder import PromptBuilder

# Lines of slack used when deciding whether two findings refer to the same
# location. Different models often report slightly different line ranges for
# the same weakness, so we allow a small gap before treating two findings as
# unrelated.
_LINE_OVERLAP_SLACK = 5

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
        tagged = [
            (finding, response.model_id)
            for response in responses
            for finding in response.findings
        ]
        return [self._merge_cluster(cluster) for cluster in self._cluster_findings(tagged)]

    def _majority_vote_strategy(self, responses: List[ModelResponse]) -> List[Finding]:
        """Keep findings that at least two distinct models report in overlapping
        line ranges of the same file and CWE family.

        Two models agreeing on the same weakness at the same location is a
        strong signal, so we require >=2 rather than the historic strict
        >50% threshold (which, with 5+ heterogeneous models and divergent
        CWE labelling, almost never cleared).
        """
        if len(responses) == 1:
            return responses[0].findings

        tagged = [
            (finding, response.model_id)
            for response in responses
            for finding in response.findings
        ]
        consensus_findings = []
        for cluster in self._cluster_findings(tagged):
            distinct_models = {mid for _, mid in cluster}
            if len(distinct_models) >= 2:
                consensus_findings.append(self._merge_cluster(cluster))
        return consensus_findings

    def _weighted_vote_strategy(
        self, responses: List[ModelResponse], weights: Dict[str, float]
    ) -> List[Finding]:
        """Weighted vote: sum per-model weights across overlapping clusters."""
        if len(responses) == 1:
            return responses[0].findings

        tagged = [
            (finding, response.model_id)
            for response in responses
            for finding in response.findings
        ]

        total_weight = sum(weights.get(r.model_id, 1.0) for r in responses)
        consensus_findings = []
        for cluster in self._cluster_findings(tagged):
            distinct_models = {mid for _, mid in cluster}
            cluster_weight = sum(weights.get(mid, 1.0) for mid in distinct_models)
            if cluster_weight > total_weight / 2:
                consensus_findings.append(self._merge_cluster(cluster))
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

    @staticmethod
    def _normalize_cwe(cwe: str) -> str:
        """Reduce a CWE string to a stable clustering token.

        If the input contains a non-zero numeric component (e.g. 'CWE-079',
        'XSS (79)') we return that integer as a string — this lets equivalent
        CWEs expressed with different prefixes/padding collide into one bucket.

        Otherwise we fall back to a slug of the raw string (lowercased,
        non-alphanumerics stripped) so that unknown codes like 'CWE-000',
        'CSRF' or 'potential_vulnerability' remain distinct buckets rather
        than all collapsing into one catch-all "unknown" cluster.
        """
        if not cwe:
            return ""
        digits = "".join(ch for ch in cwe if ch.isdigit())
        if digits:
            try:
                num = int(digits)
            except ValueError:
                num = 0
            if num != 0:
                return str(num)
        slug = "".join(ch for ch in cwe.lower() if ch.isalnum())
        return slug

    def _normalize_finding_key(self, finding: Finding) -> str:
        """Coarse key: file + CWE family.

        Kept for callers that need a hashable grouping key. Fine-grained
        line-overlap clustering happens in :meth:`_cluster_findings`.
        """
        key_parts = [finding.file or "", self._normalize_cwe(finding.cwe)]
        return hashlib.sha1("|".join(key_parts).encode()).hexdigest()

    def _cluster_findings(
        self, tagged: List[Tuple[Finding, str]]
    ) -> List[List[Tuple[Finding, str]]]:
        """Cluster findings so that the same underlying weakness reported by
        multiple models ends up in one group.

        Two findings belong to the same cluster when they share a file and
        CWE family and their line ranges overlap (with a small slack). This
        replaces the earlier exact-bucket key, which fragmented clusters when
        one model pointed at a single line and another reported the
        surrounding block.
        """
        # First bucket by the coarse file+cwe key so we only run the quadratic
        # overlap check within small groups.
        buckets: Dict[str, List[Tuple[Finding, str]]] = {}
        for item in tagged:
            buckets.setdefault(self._normalize_finding_key(item[0]), []).append(item)

        clusters: List[List[Tuple[Finding, str]]] = []
        for items in buckets.values():
            n = len(items)
            if n == 1:
                clusters.append(items)
                continue

            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n):
                fi = items[i][0]
                for j in range(i + 1, n):
                    fj = items[j][0]
                    if self._line_ranges_overlap(
                        fi.start_line, fi.end_line, fj.start_line, fj.end_line
                    ):
                        union(i, j)

            groups: Dict[int, List[Tuple[Finding, str]]] = {}
            for idx, item in enumerate(items):
                groups.setdefault(find(idx), []).append(item)
            clusters.extend(groups.values())

        return clusters

    @staticmethod
    def _line_ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        """Return True if [a_start, a_end] and [b_start, b_end] overlap within
        ``_LINE_OVERLAP_SLACK`` lines of slack."""
        if a_end < a_start:
            a_end = a_start
        if b_end < b_start:
            b_end = b_start
        return (
            a_start - _LINE_OVERLAP_SLACK <= b_end
            and b_start - _LINE_OVERLAP_SLACK <= a_end
        )

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Deduplicate findings via overlap clustering, one finding per cluster."""
        tagged = [(f, "") for f in findings]
        return [self._merge_cluster(cluster) for cluster in self._cluster_findings(tagged)]

    def _merge_cluster(self, cluster: List[Tuple[Finding, str]]) -> Finding:
        """Merge a clustered (finding, model_id) tuple list into one Finding,
        preserving the set of distinct model ids on ``contributing_models``.

        Any ids already present on the input findings (e.g. from judge-model
        outputs or earlier merges) are carried through so that nested merges
        don't lose attribution.
        """
        findings = [f for f, _ in cluster]
        merged = self._merge_finding_group(findings)
        contributors: List[str] = []
        seen: Set[str] = set()
        for finding, model_id in cluster:
            for mid in finding.contributing_models or []:
                if mid and mid not in seen:
                    seen.add(mid)
                    contributors.append(mid)
            if model_id and model_id not in seen:
                seen.add(model_id)
                contributors.append(model_id)
        merged.contributing_models = contributors
        return merged

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

