// Scan detail page JavaScript
const scanId = document.getElementById("scanId").textContent.trim();

document.addEventListener("DOMContentLoaded", function () {
  loadScanResults();
});

async function loadScanResults() {
  try {
    const response = await fetch(`/api/scan/${scanId}`);
    const data = await response.json();

    if (data.error) {
      const status = await getScanStatus(scanId);
      if (status === "running" || status === "pending") {
        window.location.href = `/scan/${scanId}/progress`;
        return;
      }
      showError("Error: " + data.error);
      return;
    }

    await renderScanResults(data);
  } catch (error) {
    console.error("Error loading scan results:", error);
    showError("Error loading scan results: " + error.message);
  }
}

async function getScanStatus(id) {
  try {
    const response = await fetch(`/api/scan/${id}/status`);
    if (!response.ok) return null;
    const data = await response.json();
    return data.status || null;
  } catch (e) {
    return null;
  }
}

async function renderScanResults(data) {
  document.getElementById("loading").style.display = "none";
  document.getElementById("scanContent").style.display = "block";

  // Render threat overview dashboard
  renderThreatOverview(data);

  // Render scan status
  renderScanStatus(data);

  // Render consensus findings
  await renderConsensusFindings(data.consensus_findings || []);

  // Render per-model findings
  await renderPerModelFindings(data.per_model_findings || {});
}

function renderScanStatus(data) {
  const statusContainer = document.getElementById("scanStatus");
  const timestamp = data.created_at || "UNKNOWN";
  const modelsUsed = Object.keys(data.per_model_findings || {}).length;

  statusContainer.innerHTML = `
    <span class="text-muted">TIMESTAMP:</span>
    <span class="text-primary">${escapeHtml(timestamp)}</span>
    <span class="text-muted ms-3">MODELS:</span>
    <span class="text-primary">${modelsUsed}</span>
  `;
}

function renderThreatOverview(data) {
  const container = document.getElementById("threatOverview");
  const findings = data.consensus_findings || [];

  const critical = findings.filter(f => f.severity === "critical").length;
  const high = findings.filter(f => f.severity === "high").length;
  const medium = findings.filter(f => f.severity === "medium").length;
  const low = findings.filter(f => f.severity === "low").length;
  const info = findings.filter(f => f.severity === "info").length;

  const total = findings.length;
  const highCritCount = critical + high;

  container.innerHTML = `
    <div class="col-md-3">
      <div class="threat-card" style="--threat-color: var(--brand-primary);">
        <i class="bi bi-bullseye threat-icon"></i>
        <div class="threat-value">${total}</div>
        <div class="threat-label">Total Threats</div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="threat-card" style="--threat-color: ${critical > 0 ? '#7c3aed' : 'var(--status-danger)'};">
        <i class="bi bi-exclamation-triangle-fill threat-icon"></i>
        <div class="threat-value">${highCritCount}</div>
        <div class="threat-label">Critical / High</div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="threat-card" style="--threat-color: var(--status-warning);">
        <i class="bi bi-exclamation-circle threat-icon"></i>
        <div class="threat-value">${medium}</div>
        <div class="threat-label">Medium Severity</div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="threat-card" style="--threat-color: var(--status-success);">
        <i class="bi bi-info-circle threat-icon"></i>
        <div class="threat-value">${low + info}</div>
        <div class="threat-label">Low / Info</div>
      </div>
    </div>
  `;
}

async function renderConsensusFindings(findings) {
  const container = document.getElementById("consensusFindings");

  if (findings.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i class="bi bi-shield-check"></i>
        <h3 class="text-success mt-3 mb-2 text-uppercase" style="letter-spacing: 0.1em;">ALL CLEAR</h3>
        <p class="lead text-muted">No security vulnerabilities detected in consensus analysis.</p>
      </div>
    `;
    return;
  }

  // Sort findings by severity
  const severityOrder = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
  findings.sort((a, b) => {
    return (severityOrder[a.severity] || 5) - (severityOrder[b.severity] || 5);
  });

  let html = "";
  for (const finding of findings) {
    html += await renderFindingCard(finding);
  }

  container.innerHTML = html;
}

async function renderFindingCard(finding) {
  const severityColor = getSeverityColor(finding.severity);
  const codeSnippet = await getCodeSnippet(
    finding.file,
    finding.start_line,
    finding.end_line,
    finding.severity
  );

  return `
    <div class="vuln-card severity-${finding.severity}" style="--severity-color: ${severityColor};">
      <div class="p-3 border-bottom border-subtle">
        <div class="d-flex justify-content-between align-items-start">
          <div class="flex-grow-1">
            <div class="d-flex align-items-center gap-2 mb-2">
              <span class="severity-badge ${finding.severity}">${finding.severity}</span>
              <span class="badge bg-secondary font-monospace">${escapeHtml(finding.cwe || "CWE-UNKNOWN")}</span>
              ${finding.confidence ? `<span class="badge bg-info font-monospace">${(finding.confidence * 100).toFixed(0)}% CONFIDENCE</span>` : ""}
            </div>
            <h5 class="mb-2 text-uppercase" style="letter-spacing: 0.05em; color: var(--text-main);">
              <i class="bi bi-bug-fill me-2" style="color: ${severityColor};"></i>
              ${escapeHtml(finding.name || "Unnamed Vulnerability")}
            </h5>
          </div>
        </div>
      </div>
      <div class="p-3">
        <div class="mb-3">
          <div class="d-flex align-items-center gap-2 mb-2">
            <i class="bi bi-file-code text-primary"></i>
            <span class="font-monospace small text-secondary">${escapeHtml(finding.file)}</span>
            <span class="badge bg-dark font-monospace">L${finding.start_line}-${finding.end_line}</span>
          </div>
          <p class="mb-0 text-secondary">${escapeHtml(finding.message || "No description available")}</p>
        </div>
        ${codeSnippet}
        ${finding.fingerprint ? `<div class="mt-3 pt-3 border-top border-subtle">
          <small class="text-secondary font-monospace">FINGERPRINT: <code class="text-primary">${escapeHtml(finding.fingerprint)}</code></small>
        </div>` : ""}
      </div>
    </div>
  `;
}

async function getCodeSnippet(filePath, startLine, endLine, severity) {
  try {
    const encodedPath = encodeURIComponent(filePath);
    const response = await fetch(`/api/scan/${scanId}/file/${encodedPath}`);

    if (!response.ok) {
      return `<div class="code-box">
        <div class="code-box-header">
          <i class="bi bi-code-square"></i>
          <span>Source Code Unavailable</span>
        </div>
        <div class="p-3 text-muted text-center">
          <i class="bi bi-exclamation-circle me-2"></i>
          Unable to load source code for this file
        </div>
      </div>`;
    }

    const data = await response.json();
    const content = data.content;
    const lines = content.split("\n");

    // Context lines
    const contextBefore = 5;
    const contextAfter = 5;
    const displayStart = Math.max(1, startLine - contextBefore);
    const displayEnd = Math.min(lines.length, endLine + contextAfter);

    let codeHtml = `
      <div class="code-box">
        <div class="code-box-header">
          <i class="bi bi-code-square"></i>
          <span>Code Context</span>
          <span class="ms-auto text-primary">Lines ${displayStart}-${displayEnd}</span>
        </div>
        <div class="code-box-content">
    `;

    for (let i = displayStart; i <= displayEnd; i++) {
      const lineContent = lines[i - 1] || "";
      const isVulnerable = i >= startLine && i <= endLine;
      const lineClass = isVulnerable ? "vuln" : "context";

      codeHtml += `
        <div class="code-line ${lineClass}">
          <div class="code-line-number">${i}</div>
          <div class="code-line-content">${escapeHtml(lineContent)}</div>
        </div>
      `;
    }

    codeHtml += `
        </div>
      </div>
    `;

    return codeHtml;
  } catch (error) {
    console.error("Error fetching code snippet:", error);
    return `<div class="code-box">
      <div class="code-box-header">
        <i class="bi bi-code-square"></i>
        <span>Error Loading Code</span>
      </div>
      <div class="p-3 text-muted text-center">
        <i class="bi bi-x-circle me-2"></i>
        ${escapeHtml(error.message)}
      </div>
    </div>`;
  }
}

async function renderPerModelFindings(perModelFindings) {
  const container = document.getElementById("perModelFindings");

  if (Object.keys(perModelFindings).length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i class="bi bi-cpu"></i>
        <h3 class="mt-3 mb-2 text-uppercase" style="letter-spacing: 0.1em;">No Model Data</h3>
        <p class="lead text-muted">No per-model findings available for this scan.</p>
      </div>
    `;
    return;
  }

  let html = "";
  for (const [modelId, findings] of Object.entries(perModelFindings)) {
    // Sort findings by severity
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
    findings.sort((a, b) => {
      return (severityOrder[a.severity] || 5) - (severityOrder[b.severity] || 5);
    });

    // Count by severity
    const critical = findings.filter(f => f.severity === "critical").length;
    const high = findings.filter(f => f.severity === "high").length;
    const medium = findings.filter(f => f.severity === "medium").length;
    const low = findings.filter(f => f.severity === "low").length;

    html += `
      <div class="model-intel-card">
        <div class="model-intel-header">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <h5 class="mb-1 text-uppercase font-monospace" style="letter-spacing: 0.1em;">
                <i class="bi bi-cpu-fill me-2 text-primary"></i>
                ${escapeHtml(modelId)}
              </h5>
              <div class="d-flex gap-2 mt-2">
                ${critical > 0 ? `<span class="badge severity-badge critical">${critical} Critical</span>` : ""}
                ${high > 0 ? `<span class="badge severity-badge high">${high} High</span>` : ""}
                ${medium > 0 ? `<span class="badge severity-badge medium">${medium} Medium</span>` : ""}
                ${low > 0 ? `<span class="badge severity-badge low">${low} Low</span>` : ""}
              </div>
            </div>
            <div class="text-end">
              <div class="threat-value" style="font-size: 1.5rem;">${findings.length}</div>
              <div class="threat-label">Findings</div>
            </div>
          </div>
        </div>
        <div class="model-intel-body">
    `;

    for (const finding of findings) {
      html += await renderFindingCard(finding);
    }

    html += `
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
}

function getSeverityColor(severity) {
  const colors = {
    critical: "#7c3aed",
    high: "#ef4444",
    medium: "#f59e0b",
    low: "#10b981",
    info: "#06b6d4",
  };
  return colors[severity] || "#6b7280";
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function showError(message) {
  document.getElementById("loading").style.display = "none";
  document.getElementById("scanContent").innerHTML = `
    <div class="empty-state">
      <i class="bi bi-x-circle" style="color: var(--status-danger);"></i>
      <h3 class="mt-3 mb-2 text-uppercase" style="letter-spacing: 0.1em; color: var(--status-danger);">Error</h3>
      <p class="lead text-muted">${escapeHtml(message)}</p>
    </div>
  `;
  document.getElementById("scanContent").style.display = "block";
}

// Export functions
function exportSARIF() {
  window.location.href = `/api/scan/${scanId}/sarif`;
}

function exportCSV() {
  window.location.href = `/api/scan/${scanId}/csv`;
}

async function exportJSON() {
  try {
    const response = await fetch(`/api/scan/${scanId}`);
    const data = await response.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aegis_scan_${scanId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert("Error exporting JSON: " + error.message);
  }
}
