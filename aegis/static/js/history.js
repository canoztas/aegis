// history.js - Refactored for Tactical UI
document.addEventListener("DOMContentLoaded", () => {
    loadScanHistory();

    // Search listener
    const searchInput = document.getElementById("searchHistory");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            filterHistory(e.target.value);
        });
    }
});

let allScans = [];

async function loadScanHistory() {
    const listContainer = document.getElementById("scanHistoryList");
    const loader = document.getElementById("historyLoader");

    try {
        const response = await fetch("/api/scans?limit=50");
        const data = await response.json();
        const scans = data.scans || [];

        allScans = scans; // Store for filtering
        renderScans(scans);

        if (loader) loader.classList.add("d-none");
    } catch (error) {
        console.error("Failed to load history:", error);
        if (listContainer) {
            listContainer.innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="text-danger font-monospace small">CONNECTION_ERROR: ${error.message}</div>
                </div>
            `;
        }
    }
}

function renderScans(scans) {
    const listContainer = document.getElementById("scanHistoryList");
    if (!listContainer) return;

    listContainer.innerHTML = ""; // Clear loader/previous
    document.getElementById("historyLoader")?.classList.add("d-none");

    if (scans.length === 0) {
        listContainer.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="bi bi-inbox fs-1 text-secondary opacity-25"></i>
                <div class="text-secondary font-monospace small mt-2">NO_ARCHIVES_FOUND</div>
            </div>
        `;
        return;
    }

    scans.forEach(scan => {
        const cardcheck = createScanCard(scan);
        listContainer.appendChild(cardcheck);
    });
}

function createScanCard(scan) {
    const col = document.createElement("div");
    col.className = "col-12";

    // Status Badge
    let statusBadge = '';
    let statusBorder = 'border-secondary';

    if (scan.status === 'completed') {
        statusBadge = '<span class="badge bg-primary bg-opacity-10 text-primary font-monospace rounded-0 border border-primary border-opacity-25">COMPLETED</span>';
        statusBorder = 'border-subtle hover-border-primary';
    } else if (scan.status === 'failed') {
        statusBadge = '<span class="badge bg-danger bg-opacity-10 text-danger font-monospace rounded-0 border border-danger border-opacity-25">FAILED</span>';
        statusBorder = 'border-subtle hover-border-danger';
    } else if (scan.status === 'running' || scan.status === 'pending') {
        statusBadge = '<span class="badge bg-warning bg-opacity-10 text-warning font-monospace rounded-0 border border-warning border-opacity-25 blink">ACTIVE</span>';
        statusBorder = 'border-warning border-opacity-50';
    }

    // Findings Badges
    const critical = scan.severity_counts?.critical || 0;
    const high = scan.severity_counts?.high || 0;
    const medium = scan.severity_counts?.medium || 0;
    const low = scan.severity_counts?.low || 0;

    const timestamp = scan.created_at ? new Date(scan.created_at).toLocaleString() : 'UNKNOWN_DATE';

    col.innerHTML = `
        <div class="card bg-panel ${statusBorder} shadow-sm rounded-0 transition-all cursor-pointer group" onclick="window.location.href='/scan/${scan.scan_id}'">
            <div class="card-body p-3 d-flex align-items-center flex-wrap gap-3">
                <!-- ID & Date -->
                <div class="flex-grow-1">
                    <div class="d-flex align-items-center gap-2 mb-1">
                        ${statusBadge}
                        <h6 class="mb-0 font-monospace text-light small text-truncate" style="max-width: 250px;">${scan.scan_id}</h6>
                    </div>
                    <div class="d-flex gap-3 text-secondary extra-small font-monospace">
                        <span><i class="bi bi-clock me-1"></i>${timestamp}</span>
                        <span><i class="bi bi-files me-1"></i>${scan.total_files} FILES</span>
                        <span><i class="bi bi-hdd-network me-1"></i>${scan.consensus_strategy}</span>
                    </div>
                </div>
                
                <!-- Stats -->
                <div class="d-flex gap-2">
                    ${critical > 0 ? `<div class="badge bg-danger text-black rounded-0 font-monospace">CRIT:${critical}</div>` : ''}
                    ${high > 0 ? `<div class="badge bg-warning text-black rounded-0 font-monospace">HIGH:${high}</div>` : ''}
                    ${medium > 0 ? `<div class="badge bg-info text-black rounded-0 font-monospace">MED:${medium}</div>` : ''}
                    ${low > 0 ? `<span class="badge bg-secondary bg-opacity-25 text-secondary rounded-0 font-monospace border border-secondary border-opacity-25">LOW:${low}</span>` : ''}
                    ${scan.total_findings === 0 ? '<span class="badge bg-success bg-opacity-10 text-success rounded-0 font-monospace border border-success border-opacity-25">CLEAN</span>' : ''}
                </div>
                
                <!-- Action -->
                <div class="border-start border-subtle ps-3 d-flex align-items-center">
                    <button class="btn btn-icon btn-sm text-secondary hover-text-primary">
                        <i class="bi bi-chevron-right"></i>
                    </button>
                </div>
            </div>
        </div>
    `;

    return col;
}

function filterHistory(query) {
    if (!query) {
        renderScans(allScans);
        return;
    }

    const lowerQuery = query.toLowerCase();
    const filtered = allScans.filter(scan =>
        scan.scan_id.toLowerCase().includes(lowerQuery) ||
        (scan.status && scan.status.toLowerCase().includes(lowerQuery))
    );
    renderScans(filtered);
}
