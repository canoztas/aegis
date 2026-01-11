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

  try {
    localStorage.setItem("aegis_last_scan_id", scanId);
    localStorage.setItem("aegis_last_scan_status", "completed");
  } catch (e) {
    // Ignore storage errors
  }

  window.currentFindings = [];

  // Render threat overview dashboard
  renderThreatOverview(data);

  // Render scan status
  renderScanStatus(data);

  // Render consensus findings
  await renderConsensusFindings(data.consensus_findings || []);

  // Render War Room
  renderWarRoom(data);

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
  const findings = data.consensus_findings || [];

  // --- 1. Calculate Stats ---
  const critical = findings.filter(f => f.severity === "critical").length;
  const high = findings.filter(f => f.severity === "high").length;

  document.getElementById("statCritical").textContent = critical;
  document.getElementById("statHigh").textContent = high;

  const affectedFiles = new Set(findings.map(f => f.file)).size;
  const totalFiles = data.scan_metadata?.total_files || affectedFiles;
  document.getElementById("statClean").textContent = Math.max(0, totalFiles - affectedFiles);

  // --- 3. Render Threat Dial (Risk Score) ---
  let riskScore = 0;
  if (findings.length > 0) {
    // Create a more dramatic score: High/Critical weight heavily
    const weightedSum = (critical * 10) + (high * 5);
    // Normalized against a "bad" scenario
    const worstCase = findings.length * 5;
    // If mostly crit/high, score approaches 100. If info/low, score is low.
    // Let's use a simpler heuristic for visual impact:
    // Score = (Crit*3 + High*2 + Low*1) / Total * 33 (roughly)
    // Actually, let's just stick to the previous simple formula but ensure it's robust
    const totalWeight = (critical * 10) + (high * 5) + (findings.length - critical - high);
    riskScore = Math.min(100, Math.round((totalWeight / (findings.length * 10)) * 100));
    if (findings.length > 0 && riskScore < 5) riskScore = 5; // Min score if findings exist
  }

  document.getElementById("riskScoreValue").textContent = riskScore + "%";

  const maxStroke = 251;
  const strokeValue = (riskScore / 100) * maxStroke;
  const dial = document.getElementById("riskDialArc");
  if (dial) dial.style.strokeDasharray = `${strokeValue}, ${maxStroke}`;

  // --- 4. Render Heatmap ---
  const heatmapContainer = document.getElementById("heatmapContainer");
  if (!heatmapContainer) return;

  heatmapContainer.innerHTML = "";

  const fileMap = {};
  findings.forEach(f => {
    if (!fileMap[f.file]) fileMap[f.file] = { score: 0, highestSev: 'info' };
    const weight = { critical: 4, high: 3, medium: 2, low: 1, info: 0 };
    fileMap[f.file].score += weight[f.severity] || 0;
    if ((weight[f.severity] || 0) > (weight[fileMap[f.file].highestSev] || 0)) {
      fileMap[f.file].highestSev = f.severity;
    }
  });

  const files = Object.keys(fileMap);
  if (files.length === 0) {
    heatmapContainer.innerHTML = '<div class="d-flex w-100 h-100 justify-content-center align-items-center text-muted small font-monospace">NO_THREAT_DATA</div>';
  } else {
    files.forEach(file => {
      const data = fileMap[file];
      const cell = document.createElement("div");
      let cssClass = "cell-success";
      let icon = "bi-file-earmark-check";

      if (data.highestSev === 'critical' || data.highestSev === 'high') {
        cssClass = "cell-danger";
        icon = "bi-radioactive";
      } else if (data.highestSev === 'medium') {
        cssClass = "cell-warning";
        icon = "bi-exclamation-triangle";
      }
      cell.className = `heatmap-cell ${cssClass}`;
      cell.title = `${file}\nSeverity: ${data.highestSev.toUpperCase()}`;
      cell.innerHTML = `<i class="bi ${icon}"></i><span class="fname">${file.split('/').pop()}</span>`;
      heatmapContainer.appendChild(cell);
    });
  }
}

async function renderConsensusFindings(findings) {
  const container = document.getElementById("consensusFindings");
  if (!container) return;

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

function renderWarRoom(data) {
  const container = document.getElementById("warRoomContent");
  if (!container) return; // Guard for older templates

  const perModel = data.per_model_findings || {};
  const models = Object.keys(perModel);

  if (models.length < 2) {
    container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-hdd-network"></i>
                <h3 class="mt-3 mb-2 text-uppercase">NOT_ENOUGH_DATA</h3>
                <p class="lead text-muted">War Room requires at least 2 models to analyze consensus.</p>
            </div>`;
    return;
  }

  // 1. Calculate Overlap
  // Finding unique ID = file + line + type (approx)
  const allFindingsMap = new Map();
  const modelContribution = {};

  models.forEach(model => {
    modelContribution[model] = { unique: 0, total: 0 };
    (perModel[model] || []).forEach(f => {
      const key = `${f.file}:${f.start_line}:${f.type}`;
      if (!allFindingsMap.has(key)) {
        allFindingsMap.set(key, { models: [], finding: f });
      }
      allFindingsMap.get(key).models.push(model);
      modelContribution[model].total++;
    });
  });

  let consensusCount = 0;
  let contentiousCount = 0;

  allFindingsMap.forEach((val, key) => {
    if (val.models.length === models.length) consensusCount++;
    if (val.models.length === 1) {
      modelContribution[val.models[0]].unique++;
      contentiousCount++;
    }
  });

  // 2. Render UI
  let modelStatsHtml = models.map(m => `
        <div class="col-md-4">
            <div class="p-3 bg-surface border border-subtle rounded-1 text-center">
                <h6 class="text-primary font-monospace">${m}</h6>
                <div class="d-flex justify-content-center gap-3 mt-2">
                    <div>
                        <div class="fs-4 fw-bold text-light">${modelContribution[m].total}</div>
                        <div class="extra-small text-muted">TOTAL</div>
                    </div>
                    <div class="border-start border-secondary mx-1"></div>
                    <div>
                        <div class="fs-4 fw-bold text-warning">${modelContribution[m].unique}</div>
                        <div class="extra-small text-muted">UNIQUE</div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');

  container.innerHTML = `
        <div class="row g-4">
            <!-- Central Intelligence -->
            <div class="col-12 text-center mb-2">
                <div class="d-inline-flex align-items-center gap-4 p-3 bg-panel border border-primary border-opacity-25 rounded-pill shadow-sm">
                    <div class="text-end">
                        <div class="text-aegis-gold fs-5 font-monospace fw-bold">${consensusCount}</div>
                        <div class="extra-small text-muted text-uppercase">High Certainty (100% Agree)</div>
                    </div>
                    <div class="vr bg-secondary opacity-50" style="height: 30px;"></div>
                    <div class="text-start">
                        <div class="text-info fs-5 font-monospace fw-bold">${contentiousCount}</div>
                        <div class="extra-small text-muted text-uppercase">Contentious (Single Source)</div>
                    </div>
                </div>
            </div>

            <!-- Model Breakdown -->
            ${modelStatsHtml}

            <!-- Insight Card -->
            <div class="col-12">
                 <div class="card bg-surface border-subtle">
                    <div class="card-body">
                        <h6 class="text-uppercase text-muted font-monospace mb-3"><i class="bi bi-lightbulb me-2"></i> Tactical Analysis</h6>
                        <p class="text-secondary mb-0">
                            ${consensusCount > 0
      ? "Multiple models confirmed critical vectors. High confidence in consensus findings."
      : "Models show high divergence. Manual verification recommended for 'Unique' findings."}
                        </p>
                    </div>
                 </div>
            </div>
        </div>
    `;
}

async function renderFindingCard(finding) {
  // Store finding for reference
  if (!Array.isArray(window.currentFindings)) {
    window.currentFindings = [];
  }
  const findingIndex = window.currentFindings.push(finding) - 1;

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
            <div class="d-flex justify-content-between align-items-center">
                <h5 class="mb-2 text-uppercase" style="letter-spacing: 0.05em; color: var(--text-main);">
                  <i class="bi bi-bug-fill me-2" style="color: ${severityColor};"></i>
                  ${escapeHtml(finding.name || "Unnamed Vulnerability")}
                </h5>
                <button class="btn btn-sm btn-outline-primary rounded-0 font-monospace" 
                        onclick="openCinematicInspector(window.currentFindings[${findingIndex}])">
                    <i class="bi bi-eye me-2"></i>INSPECT_VECTOR
                </button>
            </div>
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
  if (!container) return;

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

// --- Cinematic Inspector Logic ---
let editorInstance = null;

function openCinematicInspector(finding) {
  const modalEl = document.getElementById('cinematicInspectorModal');
  const modal = new bootstrap.Modal(modalEl);

  // 1. Populate Metadata
  document.getElementById('inspectorTitle').textContent = `VECTOR_ANALYSIS: ${finding.file}`;
  document.getElementById('inspectorMeta').textContent = `${finding.severity.toUpperCase()}::CONFIDENCE_${(finding.confidence * 100).toFixed(0)}%`;
  document.getElementById('inspectorVulnName').textContent = finding.name;
  document.getElementById('inspectorLine').textContent = `Line ${finding.start_line} - ${finding.end_line}`;
  document.getElementById('inspectorReasoning').textContent = finding.message;
  document.getElementById('inspectorFix').textContent = "AI Remediation Analysis Pending..."; // Placeholder

  modal.show();

  // 2. Load Content & Init Editor (Wait for modal transition)
  modalEl.addEventListener('shown.bs.modal', async () => {
    // Fetch full file content if possible, simplistic approach here:
    // We will just use the snippet logic or fetch file again.
    // For a full editor, we ideally want the full file. 
    // Let's try to fetch full file content.
    try {
      const encodedPath = encodeURIComponent(finding.file);
      const res = await fetch(`/api/scan/${scanId}/file/${encodedPath}`);
      const data = await res.json();

      if (!editorInstance) {
        // Determine language
        let lang = 'plaintext';
        if (finding.file.endsWith('.py')) lang = 'python';
        if (finding.file.endsWith('.js')) lang = 'javascript';
        if (finding.file.endsWith('.go')) lang = 'go';
        if (finding.file.endsWith('.html')) lang = 'html';

        // Init Monaco
        editorInstance = monaco.editor.create(document.getElementById('monaco-container'), {
          value: data.content,
          language: lang,
          theme: 'vs-dark', // we can customize this later to match aegis theme
          readOnly: true,
          minimap: { enabled: true },
          fontSize: 14,
          fontFamily: 'JetBrains Mono',
          lineNumbers: 'on',
          renderLineHighlight: 'all',
          scrollBeyondLastLine: false,
          automaticLayout: true
        });
      } else {
        editorInstance.setValue(data.content);
      }

      // 3. Apply Decorations (Red Squiggle)
      const decorations = [{
        range: new monaco.Range(finding.start_line, 1, finding.end_line, 1000),
        options: {
          isWholeLine: true,
          className: 'myContentClass',
          glyphMarginClassName: 'myGlyphMarginClass',
          inlineClassName: 'myInlineDecoration'
        }
      }];

      // We need to keep track of decorations if we want to clear them, 
      // but for this simple viewer we just overwrite.
      // Note: Monaco 0.44 uses createDecorationsCollection or deltaDecorations
      // Simplest for older/compat versions:
      editorInstance.revealLineInCenter(finding.start_line);

      // Custom styling injection for the highlight
      // We can't easily inject dynamic CSS classes without defining them globally.
      // We'll trust Monaco's selection or add a simple delta.
      const collection = editorInstance.createDecorationsCollection([
        {
          range: new monaco.Range(finding.start_line, 1, finding.end_line, 1),
          options: {
            isWholeLine: true,
            linesDecorationsClassName: 'myLineDecoration',
            className: 'bg-danger bg-opacity-25' // Bootstrap class might not work inside shadow DOM easily
          }
        }
      ]);

    } catch (e) {
      console.error("Failed to load file for inspector", e);
      document.getElementById('monaco-container').innerHTML = `<div class="p-5 text-danger">FAILED_TO_LOAD_SOURCE: ${e.message}</div>`;
    }
  }, { once: true });
}

// Global expose for onclick
window.openCinematicInspector = openCinematicInspector;


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
