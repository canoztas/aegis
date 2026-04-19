"""Recompute consensus findings for existing scans using the current engine.

Use this after changing the consensus strategy / normalization logic so that
scans already in the database reflect the new rules without having to re-run
every model.

Usage:
    python scripts/recompute_consensus.py              # all completed scans
    python scripts/recompute_consensus.py <scan_id>    # just one
    python scripts/recompute_consensus.py --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aegis.consensus.engine import ConsensusEngine  # noqa: E402
from aegis.data_models import Finding, ModelResponse  # noqa: E402
from aegis.database import get_db  # noqa: E402


def _load_per_model_findings(conn, scan_id: str) -> Dict[str, List[Finding]]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT model_id, name, severity, cwe, file, start_line, end_line,
               message, confidence, fingerprint
        FROM findings
        WHERE scan_id = ? AND is_consensus = 0
        """,
        (scan_id,),
    )
    per_model: Dict[str, List[Finding]] = defaultdict(list)
    for row in cursor.fetchall():
        model_id = row[0] or "unknown"
        per_model[model_id].append(
            Finding(
                name=row[1] or "Security Issue",
                severity=row[2] or "medium",
                cwe=row[3] or "",
                file=row[4] or "",
                start_line=row[5] or 0,
                end_line=row[6] or 0,
                message=row[7] or "",
                confidence=row[8] if row[8] is not None else 0.5,
                fingerprint=row[9] or "",
            )
        )
    return per_model


def _load_scan_strategy(conn, scan_id: str) -> Optional[str]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT consensus_strategy FROM scans WHERE scan_id = ?", (scan_id,)
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _list_completed_scans(conn) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT scan_id FROM scans WHERE status = 'completed' ORDER BY created_at DESC"
    )
    return [row[0] for row in cursor.fetchall()]


def recompute_scan(scan_id: str, dry_run: bool = False) -> Dict[str, int]:
    db = get_db()
    engine = ConsensusEngine()
    with db.get_connection() as conn:
        strategy = _load_scan_strategy(conn, scan_id) or "union"
        per_model = _load_per_model_findings(conn, scan_id)
        if not per_model:
            return {"strategy": strategy, "per_model": 0, "old_consensus": 0, "new_consensus": 0}

        responses = [
            ModelResponse(model_id=mid, findings=findings)
            for mid, findings in per_model.items()
        ]

        # Judge strategy requires a live model; fall back to union for backfill.
        effective_strategy = strategy
        if effective_strategy == "judge":
            effective_strategy = "union"

        new_consensus = engine.merge(responses, strategy=effective_strategy)

        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ? AND is_consensus = 1",
            (scan_id,),
        )
        old_consensus_count = cursor.fetchone()[0]

        summary = {
            "strategy": strategy,
            "effective_strategy": effective_strategy,
            "per_model": sum(len(f) for f in per_model.values()),
            "old_consensus": old_consensus_count,
            "new_consensus": len(new_consensus),
        }

        if dry_run:
            return summary

        cursor.execute(
            "DELETE FROM findings WHERE scan_id = ? AND is_consensus = 1",
            (scan_id,),
        )
        if new_consensus:
            cursor.executemany(
                """
                INSERT INTO findings (scan_id, model_id, is_consensus, fingerprint,
                                      name, severity, cwe, file, start_line,
                                      end_line, message, confidence,
                                      contributing_models)
                VALUES (?, NULL, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_id,
                        f.fingerprint,
                        f.name,
                        f.severity,
                        f.cwe,
                        f.file,
                        f.start_line,
                        f.end_line,
                        f.message,
                        f.confidence,
                        json.dumps(f.contributing_models) if f.contributing_models else None,
                    )
                    for f in new_consensus
                ],
            )
        conn.commit()
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_id", nargs="?", help="Scan id (omit to process all completed scans)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    db = get_db()
    with db.get_connection() as conn:
        scan_ids = [args.scan_id] if args.scan_id else _list_completed_scans(conn)

    if not scan_ids:
        print("No scans to process.")
        return 0

    for sid in scan_ids:
        try:
            summary = recompute_scan(sid, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[{sid}] ERROR: {exc}")
            continue
        print(f"[{sid}] {json.dumps(summary)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
