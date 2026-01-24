"""Export functionality for SARIF, CSV, etc."""
import json
import csv
from typing import List, Dict, Any
from datetime import datetime
from aegis.data_models import Finding, ScanResult


def export_sarif(scan_result: ScanResult, base_uri: str = "file:///") -> Dict[str, Any]:
    """Export scan results to SARIF format."""
    # SARIF version 2.1.0
    sarif = {
        "version": "2.1.0",
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Aegis",
                        "version": "1.1.0",
                        "informationUri": "https://github.com/canoztas/aegis",
                        "rules": _build_sarif_rules(scan_result.consensus_findings),
                    }
                },
                "results": _build_sarif_results(scan_result.consensus_findings, base_uri),
                "artifacts": _build_sarif_artifacts(scan_result.consensus_findings, base_uri),
            }
        ],
    }
    return sarif


def _build_sarif_rules(findings: List[Finding]) -> List[Dict[str, Any]]:
    """Build SARIF rules from findings."""
    rules_dict: Dict[str, Dict[str, Any]] = {}
    
    for finding in findings:
        cwe = finding.cwe
        if cwe not in rules_dict:
            rules_dict[cwe] = {
                "id": cwe,
                "name": finding.name,
                "shortDescription": {
                    "text": finding.message[:200]
                },
                "helpUri": f"https://cwe.mitre.org/data/definitions/{cwe.replace('CWE-', '')}.html",
                "properties": {
                    "cwe": cwe,
                    "severity": finding.severity,
                }
            }
    
    return list(rules_dict.values())


def _build_sarif_results(findings: List[Finding], base_uri: str) -> List[Dict[str, Any]]:
    """Build SARIF results from findings."""
    results = []
    
    for finding in findings:
        result = {
            "ruleId": finding.cwe,
            "level": _severity_to_sarif_level(finding.severity),
            "message": {
                "text": finding.message
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": finding.file.replace(base_uri, "") if base_uri in finding.file else finding.file,
                            "uriBaseId": "ROOT"
                        },
                        "region": {
                            "startLine": finding.start_line,
                            "endLine": finding.end_line,
                        }
                    }
                }
            ],
            "properties": {
                "confidence": finding.confidence,
                "fingerprint": finding.fingerprint,
            }
        }
        results.append(result)
    
    return results


def _build_sarif_artifacts(findings: List[Finding], base_uri: str) -> List[Dict[str, Any]]:
    """Build SARIF artifacts from findings."""
    files = set(finding.file for finding in findings)
    artifacts = []
    
    for file_path in files:
        artifacts.append({
            "location": {
                "uri": file_path.replace(base_uri, "") if base_uri in file_path else file_path,
                "uriBaseId": "ROOT"
            }
        })
    
    return artifacts


def _severity_to_sarif_level(severity: str) -> str:
    """Convert severity to SARIF level."""
    mapping = {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "note",
    }
    return mapping.get(severity.lower(), "warning")


def export_csv(findings: List[Finding], output_path: str):
    """Export findings to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Name",
            "Severity",
            "CWE",
            "File",
            "Start Line",
            "End Line",
            "Message",
            "Confidence",
            "Fingerprint",
        ])
        
        for finding in findings:
            writer.writerow([
                finding.name,
                finding.severity,
                finding.cwe,
                finding.file,
                finding.start_line,
                finding.end_line,
                finding.message,
                finding.confidence,
                finding.fingerprint,
            ])

