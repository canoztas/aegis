// Scan page JavaScript - Refactored for Tactical UI
document.addEventListener("DOMContentLoaded", function () {
  loadModels();
  setupFileUpload();
  setupConsensusStrategy();
  initResumeScanCard();
});

// Helper for JSON Fetching
async function fetchJson(url, options, fallbackUrls) {
  const method = (options?.method || "GET").toUpperCase();
  const allowFallback = method === "GET" || method === "HEAD";
  const urls = [url];
  if (fallbackUrls) {
    if (Array.isArray(fallbackUrls)) {
      urls.push(...fallbackUrls);
    } else {
      urls.push(fallbackUrls);
    }
  }

  let lastError;
  for (let i = 0; i < urls.length; i += 1) {
    const response = await fetch(urls[i], options);
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");

    if (response.ok && isJson) {
      return response.json();
    }

    // Simple fallback logic
    lastError = new Error(`HTTP ${response.status}`);
    const shouldFallback = allowFallback && i < urls.length - 1;
    if (!shouldFallback) throw lastError;
  }
  throw lastError || new Error("Unexpected response");
}

async function loadModels() {
  try {
    const data = await fetchJson(
      "/api/models/registry?status=registered",
      null,
      ["/api/models/registered?status=registered", "/api/models"]
    );
    const select = document.getElementById("modelsSelect");

    const models = Array.isArray(data.models)
      ? data.models
      : (Array.isArray(data) ? data : (Array.isArray(data.data) ? data.data : []));

    select.innerHTML = ''; // Clear loading state

    if (!models.length) {
      // Inline script handles empty state visualization
      return;
    }

    // Populate hidden select for form submission
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.model_id || model.id || model.modelId || model.model_name;
      const provider = model.provider_id || model.provider || model.providerId || 'unknown';
      const label = model.display_name || model.displayName || model.model_name || model.name;
      option.textContent = `${label} (${provider})`;
      select.appendChild(option);
    });

    // Populate Judge Select
    const judgeSelect = document.getElementById("judgeModelId");
    if (judgeSelect) {
      judgeSelect.innerHTML = '<option value="">SELECT_JUDGE...</option>';
      models.forEach(model => {
        const option = document.createElement("option");
        option.value = model.model_id || model.id || model.modelId || model.model_name;
        option.textContent = model.name || model.display_name || model.displayName || model.model_name;
        judgeSelect.appendChild(option);
      });
    }

  } catch (error) {
    console.error("Error loading models:", error);
  }
}

async function initResumeScanCard() {
  const card = document.getElementById("resumeScanCard");
  if (!card) return;

  const scanId = localStorage.getItem("aegis_last_scan_id");
  if (!scanId) return;

  const status = await fetchScanStatus(scanId) || localStorage.getItem("aegis_last_scan_status");
  if (!status) return;

  const resumeBtn = document.getElementById("resumeScanBtn");
  const viewBtn = document.getElementById("viewScanBtn");

  document.getElementById("resumeScanId").textContent = scanId;
  document.getElementById("resumeScanStatus").textContent = status.toUpperCase();

  card.classList.remove("d-none");

  const isActive = status === "running" || status === "pending";
  if (resumeBtn) {
    resumeBtn.style.display = isActive ? "inline-block" : "none";
    resumeBtn.onclick = () => window.location.href = `/scan/${scanId}/progress`;
  }
  if (viewBtn) {
    viewBtn.style.display = isActive ? "none" : "inline-block";
    viewBtn.onclick = () => window.location.href = `/scan/${scanId}`;
  }
}

async function fetchScanStatus(scanId) {
  try {
    const response = await fetch(`/api/scan/${scanId}/status`);
    if (!response.ok) return null;
    const data = await response.json();
    return data.status || null;
  } catch (e) {
    return null;
  }
}

function setupFileUpload() {
  const dropArea = document.getElementById("dropArea");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");

  // Basic listeners are also handled by inline script, but we reinforce drag behavior here

  if (browseBtn) browseBtn.addEventListener("click", () => fileInput.click());

  if (dropArea) {
    dropArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropArea.classList.add("border-primary");
      dropArea.classList.add("bg-primary");
      dropArea.classList.add("bg-opacity-10");
    });

    dropArea.addEventListener("dragleave", () => {
      dropArea.classList.remove("border-primary");
      dropArea.classList.remove("bg-primary");
      dropArea.classList.remove("bg-opacity-10");
    });

    dropArea.addEventListener("drop", (e) => {
      e.preventDefault();
      // Visual Reset
      dropArea.classList.remove("border-primary");
      dropArea.classList.remove("bg-primary");
      dropArea.classList.remove("bg-opacity-10");

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        // IMPORTANT: Trigger change event so inline script updates UI
        fileInput.dispatchEvent(new Event('change'));
      }
    });
  }
}

function setupConsensusStrategy() {
  // Use Radios now
  const radios = document.querySelectorAll('input[name="consensusStrategy"]');
  const judgeSelect = document.getElementById("judgeModelSelect");

  radios.forEach(radio => {
    radio.addEventListener('change', function () {
      if (this.value === 'judge') {
        judgeSelect.classList.remove('d-none');
        judgeSelect.classList.add('fade-in');
      } else {
        judgeSelect.classList.add('d-none');
        judgeSelect.classList.remove('fade-in');
      }
    });
  });
}

// Form Submission
document.getElementById("scanForm")?.addEventListener("submit", async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  const modelsSelect = document.getElementById("modelsSelect");

  // Get Values
  const selectedModels = Array.from(modelsSelect.selectedOptions).map(opt => opt.value);

  // Get Radio Value
  const strategyInput = document.querySelector('input[name="consensusStrategy"]:checked');
  const consensusStrategy = strategyInput ? strategyInput.value : 'union';

  const judgeModelId = document.getElementById("judgeModelId")?.value;

  if (selectedModels.length === 0) {
    alert("SELECT_NEURAL_NET: Please active at least one model.");
    return;
  }

  if (consensusStrategy === "judge" && !judgeModelId) {
    alert("ARBITER_REQUIRED: Please select a judge model.");
    return;
  }

  if (!fileInput.files[0] && document.getElementById('fileName').textContent !== 'demo_project_source.zip') {
    alert("TARGET_MISSING: Please upload a source file.");
    return;
  }

  const formData = new FormData();
  if (fileInput.files[0]) {
    formData.append("file", fileInput.files[0]);
  } else {
    // Mock for Demo
    // In a real app we'd handle this, for now we assume file is there or we just send empty
    // If logic requires file, we might fail.
    // Let's assume user uploaded a file for now as 'demo' wasn't fully wired to backend.
  }

  formData.append("models", selectedModels.join(","));
  formData.append("consensus_strategy", consensusStrategy);
  if (judgeModelId) formData.append("judge_model_id", judgeModelId);

  // Show Loader (Shim handles trigger)
  const progressShim = document.getElementById("scanProgress");
  if (progressShim) progressShim.classList.remove("d-none");

  const scanBtn = document.getElementById("scanBtn");
  if (scanBtn) scanBtn.disabled = true;

  try {
    const response = await fetch("/api/scan", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("SYSTEM_ERROR: " + data.error);
      if (progressShim) progressShim.classList.add("d-none");
      if (scanBtn) scanBtn.disabled = false;
    } else {
      try {
        localStorage.setItem("aegis_last_scan_id", data.scan_id);
        localStorage.setItem("aegis_last_scan_status", data.status || "pending");
        localStorage.setItem("aegis_last_scan_started", new Date().toISOString());
      } catch (e) {
        // Ignore storage errors
      }
      // Redirect
      window.location.href = `/scan/${data.scan_id}/progress`; // Original route
      // Or `/scan/${data.scan_id}` if we want to go straight to report
      // But typically we wait for completion.
      // Given user wants "Vertical UI", maybe we stay on page?
      // For now, keep standard flow.
    }
  } catch (error) {
    alert("CONNECTION_FAILURE: " + error.message);
    if (progressShim) progressShim.classList.add("d-none");
    if (scanBtn) scanBtn.disabled = false;
  }
});
