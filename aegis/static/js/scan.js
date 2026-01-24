// Scan page JavaScript - Refactored for Tactical UI
document.addEventListener("DOMContentLoaded", function () {
  loadModels();
  setupFileUpload();
  setupConsensusStrategy();
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
    console.log("[models-debug] Loaded models:", models.map(m => ({ id: m.model_id, name: m.display_name || m.model_name })));
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

  if (!fileInput.files[0] && document.getElementById('fileName').textContent !== 'demo_project_source.zip') {
    alert("TARGET_MISSING: Please upload a source file.");
    return;
  }

  const formData = new FormData();
  if (fileInput.files[0]) {
    formData.append("file", fileInput.files[0]);
  }

  const endpoint = "/api/scan";

  // Get consensus strategy
  const strategyInput = document.querySelector('input[name="consensusStrategy"]:checked');
  const consensusStrategy = strategyInput ? strategyInput.value : 'union';

  if (consensusStrategy === "cascade") {
    // Cascade consensus mode
    const pass1Select = document.getElementById("pass1ModelsSelect");
    const pass2Select = document.getElementById("pass2ModelsSelect");
    const pass1Models = Array.from(pass1Select?.selectedOptions || []).map(opt => opt.value);
    const pass2Models = Array.from(pass2Select?.selectedOptions || []).map(opt => opt.value);

    // Debug: log selected model IDs
    console.log("[cascade-debug] Pass 1 selected:", pass1Models);
    console.log("[cascade-debug] Pass 2 selected:", pass2Models);

    if (pass1Models.length === 0) {
      alert("CASCADE_ERROR: Please select at least one Pass 1 model.");
      return;
    }
    if (pass2Models.length === 0) {
      alert("CASCADE_ERROR: Please select at least one Pass 2 model.");
      return;
    }

    formData.append("consensus_strategy", "cascade");
    formData.append("pass1_models", pass1Models.join(","));
    formData.append("pass2_models", pass2Models.join(","));
    formData.append("pass1_strategy", document.getElementById("pass1Strategy")?.value || "union");
    formData.append("min_severity", document.getElementById("cascadeMinSeverity")?.value || "low");
    formData.append("flag_any_finding", "true");

  } else {
    // Standard consensus modes (union, majority_vote, judge)
    const modelsSelect = document.getElementById("modelsSelect");
    const selectedModels = Array.from(modelsSelect.selectedOptions).map(opt => opt.value);
    const judgeModelId = document.getElementById("judgeModelId")?.value;

    if (selectedModels.length === 0) {
      alert("SELECT_NEURAL_NET: Please activate at least one model.");
      return;
    }

    if (consensusStrategy === "judge" && !judgeModelId) {
      alert("ARBITER_REQUIRED: Please select a judge model.");
      return;
    }

    formData.append("models", selectedModels.join(","));
    formData.append("consensus_strategy", consensusStrategy);
    if (judgeModelId) formData.append("judge_model_id", judgeModelId);
  }

  // Show Loader (Shim handles trigger)
  const progressShim = document.getElementById("scanProgress");
  if (progressShim) progressShim.classList.remove("d-none");

  const scanBtn = document.getElementById("scanBtn");
  if (scanBtn) scanBtn.disabled = true;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      let errorMsg = "SYSTEM_ERROR: " + data.error;
      // Show debug info for cascade model errors
      if (data.received && data.available) {
        errorMsg += "\n\nReceived model IDs: " + JSON.stringify(data.received);
        errorMsg += "\nAvailable model IDs: " + JSON.stringify(data.available);
        console.error("Cascade model validation failed:", data);
      }
      alert(errorMsg);
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
      // Redirect to progress page
      window.location.href = `/scan/${data.scan_id}/progress`;
    }
  } catch (error) {
    alert("CONNECTION_FAILURE: " + error.message);
    if (progressShim) progressShim.classList.add("d-none");
    if (scanBtn) scanBtn.disabled = false;
  }
});
