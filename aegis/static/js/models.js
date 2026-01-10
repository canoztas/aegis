// Models page JavaScript
document.addEventListener("DOMContentLoaded", function () {
  loadRegisteredModels();
  loadOllamaModels(false);
  loadHuggingFaceModels();

  // Ensure functions are available for inline handlers if scripts load as modules
  window.openRegisterOllamaModal = openRegisterOllamaModal;
  window.registerOllamaFromModal = registerOllamaFromModal;
  window.loadRegisteredModels = loadRegisteredModels;
  window.loadOllamaModels = loadOllamaModels;
  window.loadHuggingFaceModels = loadHuggingFaceModels;
  window.toggleModel = toggleModel;
  window.deleteModel = deleteModel;
  window.openTestModel = openTestModel;
  window.runModelTest = runModelTest;
  window.pullOllamaModel = pullOllamaModel;
  window.seedBuiltinModels = seedBuiltinModels;
  window.registerHFPreset = registerHFPreset;
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

// Load ALL registered models (from database) - shown in "My Models" tab
async function loadRegisteredModels() {
  try {
    const data = await fetchJson(
      "/api/models/registry",
      null,
      ["/api/models/registered", "/api/models"]
    );
    const container = document.getElementById("registeredModelsList");

    const models = Array.isArray(data.models)
      ? data.models
      : (Array.isArray(data) ? data : (Array.isArray(data.data) ? data.data : []));

    if (!models.length) {
      container.innerHTML = `
        <div class="text-center py-5">
          <i class="bi bi-inbox" style="font-size: 3rem; color: #ccc;"></i>
          <p class="text-muted mt-3">No models registered yet</p>
          <p class="small">Use the "Add Ollama", "Add Cloud LLM", or "Add HuggingFace" tabs to register models</p>
        </div>
      `;
      return;
    }

    let html = '<div class="list-group">';
    models.forEach(model => {
      const modelId = model.model_id || model.id || model.modelId || model.model_name;
      const displayName = model.display_name || model.displayName || model.name || model.model_name || modelId;
      const provider = model.provider_id || model.provider || model.providerId || 'unknown';
      const status = model.status || (model.enabled === false ? 'disabled' : 'registered');
      const isEnabled = status === 'registered';
      const availability = model.availability
        || (model.available === true ? 'available' : model.available === false ? 'unavailable' : 'unknown');
      const availabilityBadge = availability === 'available'
        ? '<span class="badge bg-success">Available</span>'
        : availability === 'unavailable'
          ? '<span class="badge bg-danger">Unavailable</span>'
          : '<span class="badge bg-secondary">Unknown</span>';
      const enabledBadge = isEnabled ?
        '<span class="badge bg-success">Enabled</span>' :
        '<span class="badge bg-secondary">Disabled</span>';

      // Provider badge color
      let providerBadgeClass = 'bg-secondary';
      if (provider === 'ollama') providerBadgeClass = 'bg-primary';
      else if (provider === 'openai' || provider === 'anthropic') providerBadgeClass = 'bg-info';
      else if (provider === 'huggingface') providerBadgeClass = 'bg-warning';

      // Roles badges
      const roles = Array.isArray(model.roles)
        ? model.roles
        : (typeof model.roles === "string" ? model.roles.split(",").map(r => r.trim()).filter(Boolean) : []);
      const rolesBadges = roles.map(role =>
        `<span class="badge bg-info text-white">${role}</span>`
      ).join(' ');

      html += `
        <div class="list-group-item">
          <div class="d-flex justify-content-between align-items-center">
            <div class="flex-grow-1">
              <div class="d-flex align-items-center gap-2 mb-1">
                <strong>${displayName}</strong>
                <span class="badge ${providerBadgeClass}">${provider}</span>
                ${rolesBadges}
                ${availabilityBadge}
              </div>
              <small class="text-muted">${modelId}</small>
            </div>
            <div class="d-flex align-items-center gap-2">
              ${enabledBadge}
              <div class="form-check form-switch mb-0">
                <input class="form-check-input" type="checkbox"
                       ${isEnabled ? 'checked' : ''}
                       onchange="toggleModel('${modelId}')"
                       title="Enable/Disable">
              </div>
              <button class="btn btn-sm btn-outline-primary" onclick="openTestModel('${modelId}')"
                      title="Test model">
                <i class="bi bi-play-circle"></i>
              </button>
              <button class="btn btn-sm btn-danger" onclick="deleteModel('${modelId}')"
                      title="Delete model">
                <i class="bi bi-trash"></i>
              </button>
            </div>
          </div>
        </div>
      `;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (error) {
    console.error("Error loading registered models:", error);
    document.getElementById("registeredModelsList").innerHTML = `
      <div class="alert alert-danger">
        Failed to load registered models: ${error.message}
        <br><small>Tip: restart the server after updating to the new API.</small>
      </div>
    `;
  }
}

// Load available Ollama models (NOT registered) - shown in "Add Ollama" tab
async function loadOllamaModels(forceRefresh) {
  try {
    // Load both Ollama models and registered models
    const [ollamaResponse, registeredResponse] = await Promise.all([
      fetchJson(
        `/api/models/discovered/ollama${forceRefresh ? "?refresh=true" : ""}`,
        null,
        "/api/models/ollama"
      ),
      fetchJson("/api/models/registry", null, ["/api/models/registered", "/api/models"])
    ]);

    const ollamaData = ollamaResponse;
    const registeredData = registeredResponse;
    const container = document.getElementById("ollamaModelsList");

    const discoveredModels = Array.isArray(ollamaData.models)
      ? ollamaData.models
      : (Array.isArray(ollamaData) ? ollamaData : []);

    if (discoveredModels.length === 0) {
      container.innerHTML = '<p class="text-muted">No Ollama models installed. Pull a model to get started.</p>';
      return;
    }

    // Get set of registered Ollama model names
    const registeredModels = Array.isArray(registeredData.models)
      ? registeredData.models
      : (Array.isArray(registeredData) ? registeredData : (registeredData.data || []));
    const registeredModelNames = new Set(
      registeredModels
        .filter(m => (m.provider_id || m.provider) === 'ollama')
        .map(m => m.model_name || m.model || m.name)
    );

    let html = '<div class="list-group">';
    discoveredModels.forEach(model => {
      const modelName = model.name || model.model_name || model.model || model.id;
      const isRegistered = registeredModelNames.has(modelName);
      const size = model.size_bytes || model.size || 0;

      const safeModelName = escapeHtmlAttr(modelName);
      html += `
        <div class="list-group-item d-flex justify-content-between align-items-center">
          <div>
            <strong>${modelName}</strong>
            <br>
            <small class="text-muted">${formatBytes(size)}</small>
          </div>
          <div>
            ${isRegistered
              ? '<span class="badge bg-success"><i class="bi bi-check-circle"></i> Registered</span>'
              : `<button class="btn btn-sm btn-primary js-register-ollama" data-model-name="${safeModelName}">
                   <i class="bi bi-plus-circle"></i> Register
                 </button>`
            }
          </div>
        </div>
      `;
    });
    html += '</div>';
    container.innerHTML = html;

    const registerButtons = container.querySelectorAll(".js-register-ollama");
    registerButtons.forEach(button => {
      button.addEventListener("click", () => {
        const name = button.getAttribute("data-model-name");
        openRegisterOllamaModal(name);
      });
    });
  } catch (error) {
    console.error("Error loading Ollama models:", error);
    document.getElementById("ollamaModelsList").innerHTML = `
      <div class="alert alert-danger">
        Failed to load Ollama models: ${error.message}
        <br><small>Make sure Ollama is running on http://localhost:11434 and the server was restarted after updates.</small>
      </div>
    `;
  }
}

// Load HuggingFace models
async function loadHuggingFaceModels() {
  try {
    // Load both presets and registered models
  const [presetsResponse, registeredResponse] = await Promise.all([
      fetchJson("/api/models/hf/presets"),
      fetchJson(
        "/api/models/registry?type=hf_local",
        null,
        ["/api/models/registered?type=hf_local", "/api/models?type=hf_local"]
      )
    ]);

    const presetsData = presetsResponse;
    const registeredData = registeredResponse;
    const container = document.getElementById("hfModelsList");

    if (!presetsData.presets || presetsData.presets.length === 0) {
      container.innerHTML = `
        <div class="text-center py-5">
          <i class="bi bi-inbox" style="font-size: 3rem; color: #ccc;"></i>
          <p class="text-muted mt-3">No HuggingFace presets available</p>
        </div>
      `;
      return;
    }

    // Get set of registered HF model IDs
    const registeredModels = Array.isArray(registeredData.models)
      ? registeredData.models
      : (Array.isArray(registeredData) ? registeredData : (registeredData.data || []));
    const registeredModelIds = new Set(
      registeredModels.map(m => m.model_name || m.hf_model_id || m.model_id)
    );

    let html = '<div class="list-group">';
    presetsData.presets.forEach(preset => {
      const isRegistered = registeredModelIds.has(preset.model_id);
      const presetId = preset.model_id.split('/').pop().replace(/[^a-z0-9]/gi, '_');

      // Role badges
      const rolesBadges = (preset.recommended_roles || []).map(role =>
        `<span class="badge bg-info text-white">${role}</span>`
      ).join(' ');

      html += `
        <div class="list-group-item">
          <div class="d-flex justify-content-between align-items-start">
            <div class="flex-grow-1">
              <div class="d-flex align-items-center gap-2 mb-2">
                <strong>${preset.name}</strong>
                ${rolesBadges}
              </div>
              <div class="mb-2">
                <small class="text-muted d-block"><i class="bi bi-cpu"></i> ${preset.model_id}</small>
                <small class="text-muted d-block"><i class="bi bi-tag"></i> ${preset.task_type}</small>
                <small class="text-muted d-block">${preset.description}</small>
              </div>
            </div>
            <div>
              ${isRegistered
                ? '<span class="badge bg-success"><i class="bi bi-check-circle"></i> Registered</span>'
                : `<button class="btn btn-sm btn-primary" onclick="openHFPresetModal('${presetId}')">
                     <i class="bi bi-plus-circle"></i> Configure & Register
                   </button>`
              }
            </div>
          </div>
        </div>
      `;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (error) {
    console.error("Error loading HuggingFace models:", error);
    document.getElementById("hfModelsList").innerHTML = `
      <div class="alert alert-danger">
        Failed to load HuggingFace presets: ${error.message}
        <br><small>Tip: restart the server after updating to the new API.</small>
      </div>
    `;
  }
}

// Pull and auto-register Ollama model
async function pullOllamaModel() {
  const modelName = document.getElementById("ollamaModelName").value;
  if (!modelName) {
    alert("Please enter a model name");
    return;
  }

  try {
    const response = await fetch("/api/models/ollama/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName })
    });
    const data = await response.json();

    if (response.ok && data.success) {
      alert(`Pull started/completed for ${modelName}. Refresh discovered models once finished.`);
      loadOllamaModels(true);
    } else {
      alert(`Could not pull automatically. Run:\n\nollama pull ${modelName}\n\nDetails: ${data.error || 'unknown error'}`);
    }
  } catch (error) {
    alert(`Failed to start pull: ${error.message}`);
  }

  const modal = bootstrap.Modal.getInstance(document.getElementById("pullOllamaModal"));
  modal?.hide();
}

// Register an existing Ollama model
function openRegisterOllamaModal(modelName) {
  const nameInput = document.getElementById("registerOllamaName");
  const roleDeep = document.getElementById("ollamaRoleDeep");
  const roleJudge = document.getElementById("ollamaRoleJudge");
  const roleTriage = document.getElementById("ollamaRoleTriage");
  const tempInput = document.getElementById("ollamaTemp");
  const maxTokensInput = document.getElementById("ollamaMaxTokens");
  const modalEl = document.getElementById("registerOllamaModal");

  if (!nameInput || !modalEl) {
    const ok = confirm(
      "Register modal not available. Register this model with default settings?"
    );
    if (ok) {
      registerOllamaQuick(modelName);
    }
    return;
  }

  nameInput.value = modelName;
  if (roleDeep) roleDeep.checked = true;
  if (roleJudge) roleJudge.checked = false;
  if (roleTriage) roleTriage.checked = false;
  if (tempInput) tempInput.value = "0.1";
  if (maxTokensInput) maxTokensInput.value = "2048";

  const modal = new bootstrap.Modal(modalEl);
  modal.show();
}

async function registerOllamaQuick(modelName) {
  const roles = ["deep_scan"];
  try {
    const base_url = "http://localhost:11434";
    const model_id = `ollama:${modelName}`;
    const payload = {
      model_id: model_id,
      model_type: "ollama_local",
      provider_id: "ollama",
      model_name: modelName,
      display_name: `Ollama - ${modelName}`,
      roles: roles,
      parser_id: "json_schema",
      settings: {
        base_url: base_url,
        temperature: 0.1,
        max_tokens: 2048
      }
    };

    let response = await fetch("/api/models/registry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response.status === 404 || response.status === 405) {
      response = await fetch("/api/models/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }
    if (response.ok && data.model) {
      alert(`Model "${modelName}" registered successfully!`);
      loadRegisteredModels();
      loadOllamaModels(false);
    } else {
      alert("Failed to register model: " + (data.error || response.statusText || "Unknown error"));
    }
  } catch (error) {
    alert("Failed to register model: " + error.message);
  }
}

async function registerOllamaFromModal() {
  const modelName = document.getElementById("registerOllamaName").value;
  const roles = [];
  if (document.getElementById("ollamaRoleDeep").checked) roles.push("deep_scan");
  if (document.getElementById("ollamaRoleJudge").checked) roles.push("judge");
  if (document.getElementById("ollamaRoleTriage").checked) roles.push("triage");

  if (roles.length === 0) {
    alert("Select at least one role.");
    return;
  }

  try {
    const base_url = "http://localhost:11434"; // TODO: Get from config
    const model_id = `ollama:${modelName}`;
    const temperature = parseFloat(document.getElementById("ollamaTemp").value || "0.1");
    const max_tokens = parseInt(document.getElementById("ollamaMaxTokens").value || "2048", 10);

    const payload = {
      model_id: model_id,
      model_type: "ollama_local",
      provider_id: "ollama",
      model_name: modelName,
      display_name: `Ollama - ${modelName}`,
      roles: roles,
      parser_id: "json_schema",
      settings: {
        base_url: base_url,
        temperature: temperature,
        max_tokens: max_tokens
      }
    };

    let response = await fetch("/api/models/registry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response.status === 404 || response.status === 405) {
      response = await fetch("/api/models/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }
    if (response.ok && data.model) {
      alert(`Model "${modelName}" registered successfully!`);
      loadRegisteredModels();
      loadOllamaModels(false);
    } else {
      alert("Failed to register model: " + (data.error || response.statusText || "Unknown error"));
    }
  } catch (error) {
    alert("Failed to register model: " + error.message);
  }

  const modal = bootstrap.Modal.getInstance(document.getElementById("registerOllamaModal"));
  modal?.hide();
}

// Toggle model enabled/disabled (for registered models)
async function toggleModel(modelId) {
  try {
    // Get current model to determine new status
    const listData = await fetchJson(
      "/api/models/registry",
      null,
      ["/api/models/registered", "/api/models"]
    );
    const models = Array.isArray(listData.models)
      ? listData.models
      : (Array.isArray(listData) ? listData : (Array.isArray(listData.data) ? listData.data : []));
    const currentModel = models.find(m => (m.model_id || m.id || m.modelId || m.model_name) === modelId);

    if (!currentModel) {
      alert('Model not found');
      return;
    }

    // Toggle status: registered <-> disabled
    const newStatus = currentModel.status === 'registered' ? 'disabled' : 'registered';

    let response = await fetch(`/api/models/${encodeURIComponent(modelId)}/status`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    });

    if ((response.status === 404 || response.status === 405)) {
      response = await fetch(`/api/models/${encodeURIComponent(modelId)}/toggle`, {
        method: 'POST'
      });
    }

    if (response.ok) {
      loadRegisteredModels();
    } else {
      let error = {};
      try {
        error = await response.json();
      } catch (parseError) {
        error = {};
      }
      alert('Failed to toggle model: ' + (error.error || response.statusText || 'Unknown error'));
    }
  } catch (error) {
    alert('Failed to toggle model: ' + error.message);
  }
}

// Delete registered model
async function deleteModel(modelId) {
  if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
    return;
  }

  try {
    const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`, {
      method: 'DELETE'
    });

    if (response.ok) {
      alert('Model deleted successfully');
      loadRegisteredModels();
      loadOllamaModels(); // Refresh Ollama list in case it was an Ollama model
    } else {
      let error = {};
      try {
        error = await response.json();
      } catch (parseError) {
        error = {};
      }
      alert('Failed to delete model: ' + (error.error || response.statusText || 'Unknown error'));
    }
  } catch (error) {
    alert('Failed to delete model: ' + error.message);
  }
}

// Cloud LLM functions - NOT YET IMPLEMENTED IN NEW API
async function addCloudModel() {
  alert("Cloud LLM registration not yet implemented in the new API.\n\nComing soon! You'll be able to register:\n- OpenAI models (GPT-4, GPT-3.5)\n- Anthropic models (Claude)\n- Azure OpenAI models");

  // Close modal
  const modal = bootstrap.Modal.getInstance(document.getElementById("addCloudModal"));
  if (modal) {
    modal.hide();
  }
}

// HuggingFace functions
async function registerHFPreset(presetId, options = {}) {
  try {
    const payload = { preset_id: presetId };
    if (options.settings) {
      payload.settings = options.settings;
    }
    const silent = options.silent === true;

    const response = await fetch("/api/models/hf/register_preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data = {};
    try {
      data = await response.json();
    } catch (parseError) {
      data = {};
    }
    if (response.ok && data.model) {
      if (!silent) {
        alert(`HuggingFace model registered successfully!`);
      }
      loadHuggingFaceModels();
      loadRegisteredModels();
    } else {
      if (!silent) {
        alert("Failed to register HF model: " + (data.error || response.statusText || "Unknown error"));
      }
    }
  } catch (error) {
    if (!silent) {
      alert("Failed to register HF model: " + error.message);
    }
  }
}

// Legacy function - now just shows message
async function seedBuiltinModels() {
  try {
    await registerHFPreset("codebert_insecure", { silent: true });
    await registerHFPreset("codeastra_7b", { silent: true });
    alert("Registered HuggingFace built-ins (CodeBERT + CodeAstra).");
  } catch (error) {
    alert("Failed to register built-ins: " + error.message);
  }
}

function openTestModel(modelId) {
  document.getElementById("testModelId").value = modelId;
  document.getElementById("testModelResult").classList.add("d-none");
  document.getElementById("testModelResultText").textContent = "";

  const modal = new bootstrap.Modal(document.getElementById("testModelModal"));
  modal.show();
}

async function runModelTest() {
  const modelId = document.getElementById("testModelId").value;
  const prompt = document.getElementById("testModelPrompt").value;
  const resultBox = document.getElementById("testModelResult");
  const resultText = document.getElementById("testModelResultText");

  try {
    const response = await fetch("/api/models/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId, prompt: prompt })
    });

    const data = await response.json();
    if (response.ok && data.success) {
      resultText.textContent = JSON.stringify(data.result, null, 2);
    } else {
      resultText.textContent = JSON.stringify(data, null, 2);
    }
    resultBox.classList.remove("d-none");
  } catch (error) {
    resultText.textContent = error.message;
    resultBox.classList.remove("d-none");
  }
}

// Utility functions
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtmlAttr(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function openHFPresetModal(presetId) {
  const modalEl = document.getElementById("hfPresetModal");
  if (!modalEl) {
    registerHFPreset(presetId);
    return;
  }
  document.getElementById("hfPresetId").value = presetId;
  document.getElementById("hfPresetDevice").value = "auto";
  document.getElementById("hfPresetQuant").value = "none";
  const modal = new bootstrap.Modal(modalEl);
  modal.show();
}

async function registerHFPresetFromModal() {
  const presetId = document.getElementById("hfPresetId").value;
  const device = document.getElementById("hfPresetDevice").value;
  const quant = document.getElementById("hfPresetQuant").value;

  const hf_kwargs = { device_map: device, trust_remote_code: true };
  if (quant === "4bit") hf_kwargs.load_in_4bit = true;
  if (quant === "8bit") hf_kwargs.load_in_8bit = true;

  await registerHFPreset(presetId, {
    settings: { hf_kwargs },
    silent: false
  });

  const modalEl = document.getElementById("hfPresetModal");
  const modal = bootstrap.Modal.getInstance(modalEl);
  modal?.hide();
}
