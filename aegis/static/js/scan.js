// Scan page JavaScript
document.addEventListener("DOMContentLoaded", function () {
  loadModels();
  setupFileUpload();
  setupConsensusStrategy();
});

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

    const text = await response.text();
    const snippet = text.slice(0, 200).replace(/\s+/g, " ").trim();
    lastError = new Error(`HTTP ${response.status}: ${snippet || "Unexpected response"}`);

    const shouldFallback = allowFallback
      && i < urls.length - 1
      && (response.status === 404 || response.status === 405 || !isJson);

    if (!shouldFallback) {
      throw lastError;
    }
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

    if (!models.length) {
      select.innerHTML = '<option value="" disabled>No models available. <a href="/models">Add models</a> first.</option>';
      return;
    }
    
    // Clear loading message
    select.innerHTML = '';
    
    // Populate models selectbox
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.model_id || model.id || model.modelId || model.model_name;
      const provider = model.provider_id || model.provider || model.providerId || 'unknown';
      const label = model.display_name || model.displayName || model.model_name || model.name;
      option.textContent = `${label} (${provider})`;
      select.appendChild(option);
    });

    // Also populate judge model select
    const judgeSelect = document.getElementById("judgeModelId");
    judgeSelect.innerHTML = '<option value="">Select judge model...</option>';
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.model_id || model.id || model.modelId || model.model_name;
      option.textContent = model.name || model.display_name || model.displayName || model.model_name;
      judgeSelect.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading models:", error);
    const select = document.getElementById("modelsSelect");
    select.innerHTML = '<option value="" disabled>Error loading models</option>';
  }
}

function setupFileUpload() {
  const dropArea = document.getElementById("dropArea");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const scanBtn = document.getElementById("scanBtn");

  browseBtn.addEventListener("click", () => fileInput.click());

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
  });

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      handleFileSelect(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  });

  function handleFileSelect(file) {
    if (file.type === "application/zip" || file.name.endsWith(".zip")) {
      fileName.textContent = file.name;
      fileSize.textContent = formatFileSize(file.size);
      fileInfo.style.display = "block";
      updateScanButton();
    } else {
      alert("Please select a ZIP file.");
      fileInput.value = "";
      fileInfo.style.display = "none";
      scanBtn.disabled = true;
    }
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function updateScanButton() {
    const fileSelected = fileInput.files.length > 0;
    const modelsSelect = document.getElementById("modelsSelect");
    const modelsSelected = modelsSelect.selectedOptions.length > 0;
    scanBtn.disabled = !(fileSelected && modelsSelected);
  }

  // Update button when models are selected
  const modelsSelect = document.getElementById("modelsSelect");
  if (modelsSelect) {
    modelsSelect.addEventListener("change", updateScanButton);
  }
}

function setupConsensusStrategy() {
  const strategySelect = document.getElementById("consensusStrategy");
  const judgeSelect = document.getElementById("judgeModelSelect");
  
  strategySelect.addEventListener("change", function() {
    if (this.value === "judge") {
      judgeSelect.style.display = "block";
    } else {
      judgeSelect.style.display = "none";
    }
  });
}

document.getElementById("scanForm").addEventListener("submit", async function(e) {
  e.preventDefault();
  
  const fileInput = document.getElementById("fileInput");
  const modelsSelect = document.getElementById("modelsSelect");
  const selectedModels = Array.from(modelsSelect.selectedOptions).map(opt => opt.value);
  const consensusStrategy = document.getElementById("consensusStrategy").value;
  const judgeModelId = document.getElementById("judgeModelId").value;
  
  if (selectedModels.length === 0) {
    alert("Please select at least one model");
    return;
  }
  
  if (consensusStrategy === "judge" && !judgeModelId) {
    alert("Please select a judge model");
    return;
  }
  
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("models", selectedModels.join(","));
  formData.append("consensus_strategy", consensusStrategy);
  if (judgeModelId) {
    formData.append("judge_model_id", judgeModelId);
  }
  
  const progressDiv = document.getElementById("scanProgress");
  progressDiv.style.display = "block";
  document.getElementById("scanBtn").disabled = true;
  
  try {
    const response = await fetch("/api/scan", {
      method: "POST",
      body: formData,
    });
    
    const data = await response.json();
    
    if (data.error) {
      alert("Error: " + data.error);
      progressDiv.style.display = "none";
      document.getElementById("scanBtn").disabled = false;
    } else {
      // Redirect to real-time progress page
      window.location.href = `/scan/${data.scan_id}/progress`;
    }
  } catch (error) {
    alert("Error starting scan: " + error.message);
    progressDiv.style.display = "none";
    document.getElementById("scanBtn").disabled = false;
  }
});

