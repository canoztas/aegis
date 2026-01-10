// Models page JavaScript - Dark Theme Enhanced
document.addEventListener("DOMContentLoaded", function () {
  loadRegisteredModels();
  loadOllamaModels(false);
  loadHuggingFaceModels();

  // Attach globals
  window.loadRegisteredModels = loadRegisteredModels;
  window.loadOllamaModels = loadOllamaModels;
  window.pullOllamaModel = pullOllamaModel;
  window.seedBuiltinModels = seedBuiltinModels;
  window.addCloudModel = addCloudModel;
  window.registerHFPresetFromModal = registerHFPresetFromModal;
  window.openHFPresetModal = openHFPresetModal;
  window.openEditModelModal = openEditModelModal;
  window.saveRegisteredModel = saveRegisteredModel;
});

async function fetchJson(url, options, fallbackUrls) {
  const method = (options?.method || "GET").toUpperCase();
  const allowFallback = method === "GET" || method === "HEAD";
  const urls = [url];
  if (fallbackUrls) {
    if (Array.isArray(fallbackUrls)) urls.push(...fallbackUrls);
    else urls.push(fallbackUrls);
  }

  let lastError;
  for (let i = 0; i < urls.length; i += 1) {
    const response = await fetch(urls[i], options);
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");

    if (response.ok && isJson) return response.json();

    // Simple fallback
    lastError = new Error(`HTTP ${response.status}`);
    const shouldFallback = allowFallback && i < urls.length - 1;
    if (!shouldFallback) throw lastError;
  }
  throw lastError || new Error("Unexpected response");
}

async function loadRegisteredModels() {
  try {
    const data = await fetchJson("/api/models/registry", null, ["/api/models/registered"]);
    const container = document.getElementById("registeredModelsList");
    const models = Array.isArray(data.models) ? data.models : (data.data || []);
    window._registeredModelsCache = models;

    if (!models.length) {
      container.innerHTML = `
        <div class="text-center py-5">
          <i class="bi bi-inbox fs-1 text-secondary opacity-50"></i>
          <p class="text-secondary mt-3 small font-monospace">REGISTRY_EMPTY</p>
        </div>`;
      return;
    }

    let html = '<div class="list-group list-group-flush">';
    models.forEach(model => {
      const modelId = model.model_id || model.id || model.modelId;
      const displayName = model.display_name || model.name || modelId;
      const provider = model.provider_id || model.provider || 'unknown';
      const isEnabled = (model.status || 'registered') === 'registered';

      let providerBadge = 'bg-secondary';
      if (provider === 'ollama') providerBadge = 'bg-primary';
      else if (provider === 'openai') providerBadge = 'bg-info bg-opacity-75';
      else if (provider === 'huggingface') providerBadge = 'bg-warning text-dark';

      html += `
        <div class="list-group-item bg-panel border-subtle text-light p-4 transition-all">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <div class="d-flex align-items-center gap-2 mb-1">
                <span class="fw-bold font-monospace">${displayName}</span>
                <span class="badge ${providerBadge} rounded-0 font-monospace extra-small">${provider.toUpperCase()}</span>
                ${isEnabled ? '<span class="badge bg-success bg-opacity-10 text-success border border-success border-opacity-25 rounded-0 font-monospace extra-small">ACTIVE</span>' : ''}
              </div>
              <small class="text-secondary font-monospace extra-small">${modelId}</small>
            </div>
            <div class="d-flex gap-2">
               <button class="btn btn-sm btn-outline-primary border-0 hover-text-primary" onclick="openEditModelModal('${modelId}')">
                    <i class="bi bi-sliders"></i>
               </button>
               <button class="btn btn-sm btn-outline-danger border-0 hover-text-danger" onclick="deleteModel('${modelId}')">
                    <i class="bi bi-trash"></i>
               </button>
            </div>
          </div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (error) {
    console.error(error);
  }
}

async function loadOllamaModels(forceRefresh) {
  try {
    const [ollamaRes, regRes] = await Promise.all([
      fetchJson(`/api/models/discovered/ollama${forceRefresh ? "?refresh=true" : ""}`, null, "/api/models/ollama"),
      fetchJson("/api/models/registry", null, ["/api/models/registered"])
    ]);

    const container = document.getElementById("ollamaModelsList");
    const discovered = ollamaRes.models || [];

    if (discovered.length === 0) {
      container.innerHTML = '<p class="text-secondary text-center small font-monospace py-4">NO_SIGNALS_DETECTED_ON_LOCALHOST</p>';
      return;
    }

    const regModels = regRes.models || [];
    const regIds = new Set(regModels.filter(m => m.provider_id === 'ollama').map(m => m.model_name));

    let html = '<div class="list-group list-group-flush">';
    discovered.forEach(model => {
      const name = model.name || model.model;
      const size = formatBytes(model.size || 0);
      const isReg = regIds.has(name);

      html += `
            <div class="list-group-item bg-panel border-subtle text-light p-3">
              <div class="d-flex justify-content-between align-items-center">
                <div>
                  <div class="fw-bold font-monospace small text-primary">${name}</div>
                  <div class="text-secondary extra-small font-monospace">${size}</div>
                </div>
                <div>
                   ${isReg
          ? '<span class="text-success small font-monospace"><i class="bi bi-check2"></i> LINKED</span>'
          : `<button class="btn btn-sm btn-outline-primary rounded-0 font-monospace extra-small" onclick="openRegisterOllamaModal('${name}')">REGISTER</button>`}
                </div>
              </div>
            </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (e) {
    console.error(e);
  }
}

async function loadHuggingFaceModels() {
  try {
    const [presetsRes, regRes] = await Promise.all([
      fetchJson("/api/models/hf/presets"),
      fetchJson("/api/models/registry?type=hf_local", null, ["/api/models/registered"])
    ]);

    const container = document.getElementById("hfModelsList");
    const presets = presetsRes.presets || [];

    if (presets.length === 0) {
      container.innerHTML = `<div class="text-center py-5 text-secondary font-monospace">NO_PRESETS_AVAILABLE</div>`;
      return;
    }

    const regModels = regRes.models || [];
    const regIds = new Set(regModels.map(m => m.model_name || m.hf_model_id || m.model_id));

    let html = '<div class="list-group list-group-flush">';
    presets.forEach(preset => {
      const isReg = regIds.has(preset.model_id);
      const presetId = preset.model_id.split('/').pop().replace(/[^a-z0-9]/gi, '_');

      html += `
            <div class="list-group-item bg-panel border-subtle text-light p-4">
                <div class="d-flex justify-content-between align-items-start gap-3">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center flex-wrap gap-2 mb-2">
                             <span class="fw-bold text-white fs-5" style="letter-spacing: 0.05em;">${preset.name}</span>
                             ${(preset.recommended_roles || []).map(r => `<span class="badge bg-primary bg-opacity-25 text-primary-emphasis border border-primary border-opacity-25 rounded-0 small font-monospace">${r}</span>`).join(' ')}
                        </div>
                        <div class="text-light opacity-75 mb-2" style="font-size: 0.95rem; line-height: 1.5;">${preset.description}</div>
                        <div class="font-monospace small text-aegis-gold opacity-75 text-break bg-black bg-opacity-25 p-2 rounded-1 border border-subtle d-inline-block">
                            <i class="bi bi-box-seam me-2"></i>${preset.model_id}
                        </div>
                    </div>
                    <div class="flex-shrink-0 ms-3">
                        ${isReg
          ? '<span class="badge bg-success text-white border border-success rounded-0 p-2 font-monospace"><i class="bi bi-check2-circle me-2"></i>REGISTERED</span>'
          : `<button class="btn btn-warning text-dark fw-bold rounded-0 font-monospace px-4 py-2 shadow-sm" onclick="openHFPresetModal('${presetId}')"><i class="bi bi-download me-2"></i>INSTALL</button>`
        }
                    </div>
                </div>
            </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (e) {
    console.error("HF Error", e);
    document.getElementById("hfModelsList").innerHTML = `<div class="alert alert-danger font-monospace small">CONNECTION_ERROR: ${e.message}</div>`;
  }
}

// Modal Handlers
function openRegisterOllamaModal(name) {
  if (confirm(`Register local model: ${name}?`)) {
    registerOllamaQuick(name);
  }
}

function openEditModelModal(modelId) {
  const models = window._registeredModelsCache || [];
  const model = models.find(m => (m.model_id || m.id || m.modelId) === modelId);
  if (!model) {
    alert("Model not found. Refresh the registry list.");
    return;
  }

  const settings = model.settings || {};
  const runtime = settings.runtime || {};
  const gen = settings.generation_kwargs || {};

  document.getElementById("editModelId").value = modelId;
  document.getElementById("editDisplayName").value = model.display_name || "";
  document.getElementById("editModelName").value = model.model_name || "";
  document.getElementById("editStatus").value = model.status || "registered";
  document.getElementById("editParserId").value = model.parser_id || "";
  document.getElementById("editTaskType").value = settings.task_type || "";

  const roleSet = new Set((model.roles || []).map(r => String(r)));
  document.getElementById("roleTriage").checked = roleSet.has("triage");
  document.getElementById("roleDeep").checked = roleSet.has("deep_scan") || roleSet.has("scan");
  document.getElementById("roleJudge").checked = roleSet.has("judge");
  document.getElementById("roleExplain").checked = roleSet.has("explain");

  document.getElementById("editRuntimeDevice").value = runtime.device || settings.device || "";
  const pref = runtime.device_preference || settings.device_preference || "";
  document.getElementById("editRuntimeDevicePref").value = Array.isArray(pref) ? pref.join(",") : pref;
  document.getElementById("editRuntimeDeviceMap").value = runtime.device_map || settings.device_map || "";
  document.getElementById("editRuntimeDtype").value = runtime.dtype || settings.dtype || "";
  document.getElementById("editRuntimeQuantization").value = runtime.quantization || settings.quantization || "";
  document.getElementById("editRuntimeConcurrency").value = runtime.max_concurrency || settings.max_concurrency || "";
  document.getElementById("editRuntimeKeepAlive").value = runtime.keep_alive_seconds || settings.keep_alive_seconds || "";
  document.getElementById("editRuntimeAllowFallback").checked = runtime.allow_fallback !== undefined ? !!runtime.allow_fallback : true;
  document.getElementById("editRuntimeRequireDevice").value = runtime.require_device || settings.require_device || "";

  document.getElementById("editTemperature").value = gen.temperature ?? settings.temperature ?? "";
  document.getElementById("editMaxTokens").value = settings.max_tokens ?? "";
  document.getElementById("editMaxNewTokens").value = gen.max_new_tokens ?? "";
  document.getElementById("editMinNewTokens").value = gen.min_new_tokens ?? "";
  document.getElementById("editTopP").value = gen.top_p ?? settings.top_p ?? "";
  document.getElementById("editDoSample").checked = gen.do_sample ?? settings.do_sample ?? false;

  document.getElementById("editAdapterId").value = settings.adapter_id || "";
  document.getElementById("editBaseModelId").value = settings.base_model_id || "";
  document.getElementById("editPromptTemplate").value = settings.prompt_template || "";

  document.getElementById("editHfKwargs").value = settings.hf_kwargs ? JSON.stringify(settings.hf_kwargs, null, 2) : "";
  document.getElementById("editOllamaOptions").value = settings.options ? JSON.stringify(settings.options, null, 2) : "";

  const modal = new bootstrap.Modal(document.getElementById("editModelModal"));
  modal.show();
}

function parseJsonField(value, fieldName) {
  if (!value || !value.trim()) return null;
  try {
    return JSON.parse(value);
  } catch (e) {
    throw new Error(`Invalid JSON in ${fieldName}`);
  }
}

function readNumber(value) {
  if (value === null || value === undefined) return null;
  const str = String(value).trim();
  if (!str) return null;
  const num = Number(str);
  return Number.isNaN(num) ? null : num;
}

async function saveRegisteredModel() {
  const modelId = document.getElementById("editModelId").value;
  if (!modelId) return;

  const roles = [];
  if (document.getElementById("roleTriage").checked) roles.push("triage");
  if (document.getElementById("roleDeep").checked) roles.push("deep_scan");
  if (document.getElementById("roleJudge").checked) roles.push("judge");
  if (document.getElementById("roleExplain").checked) roles.push("explain");

  if (!roles.length) {
    alert("At least one role is required.");
    return;
  }

  let hfKwargs = null;
  let ollamaOptions = null;
  try {
    hfKwargs = parseJsonField(document.getElementById("editHfKwargs").value, "HF_KWARGS_JSON");
    ollamaOptions = parseJsonField(document.getElementById("editOllamaOptions").value, "OPTIONS_JSON");
  } catch (e) {
    alert(e.message);
    return;
  }

  const runtime = {};
  const device = document.getElementById("editRuntimeDevice").value.trim();
  const devicePref = document.getElementById("editRuntimeDevicePref").value.trim();
  const deviceMap = document.getElementById("editRuntimeDeviceMap").value.trim();
  const dtype = document.getElementById("editRuntimeDtype").value;
  const quant = document.getElementById("editRuntimeQuantization").value;
  const concurrency = readNumber(document.getElementById("editRuntimeConcurrency").value);
  const keepAlive = readNumber(document.getElementById("editRuntimeKeepAlive").value);
  const allowFallback = document.getElementById("editRuntimeAllowFallback").checked;
  const requireDevice = document.getElementById("editRuntimeRequireDevice").value.trim();

  if (device) runtime.device = device;
  if (devicePref) runtime.device_preference = devicePref.split(",").map(s => s.trim()).filter(Boolean);
  if (deviceMap) runtime.device_map = deviceMap;
  if (dtype) runtime.dtype = dtype;
  if (quant) runtime.quantization = quant;
  if (concurrency !== null) runtime.max_concurrency = concurrency;
  if (keepAlive !== null) runtime.keep_alive_seconds = keepAlive;
  runtime.allow_fallback = allowFallback;
  if (requireDevice) runtime.require_device = requireDevice;

  const gen = {};
  const temperature = readNumber(document.getElementById("editTemperature").value);
  const maxTokens = readNumber(document.getElementById("editMaxTokens").value);
  const maxNewTokens = readNumber(document.getElementById("editMaxNewTokens").value);
  const minNewTokens = readNumber(document.getElementById("editMinNewTokens").value);
  const topP = readNumber(document.getElementById("editTopP").value);
  const doSample = document.getElementById("editDoSample").checked;

  if (temperature !== null) gen.temperature = temperature;
  if (maxNewTokens !== null) gen.max_new_tokens = maxNewTokens;
  if (minNewTokens !== null) gen.min_new_tokens = minNewTokens;
  if (topP !== null) gen.top_p = topP;
  gen.do_sample = doSample;

  const settings = {};
  if (Object.keys(runtime).length) settings.runtime = runtime;
  if (Object.keys(gen).length) settings.generation_kwargs = gen;
  if (temperature !== null) settings.temperature = temperature;
  if (maxTokens !== null) settings.max_tokens = maxTokens;
  if (hfKwargs) settings.hf_kwargs = hfKwargs;
  if (ollamaOptions) settings.options = ollamaOptions;

  const taskType = document.getElementById("editTaskType").value.trim();
  if (taskType) settings.task_type = taskType;

  const adapterId = document.getElementById("editAdapterId").value.trim();
  if (adapterId) settings.adapter_id = adapterId;

  const baseModelId = document.getElementById("editBaseModelId").value.trim();
  if (baseModelId) settings.base_model_id = baseModelId;

  const promptTemplate = document.getElementById("editPromptTemplate").value.trim();
  if (promptTemplate) settings.prompt_template = promptTemplate;

  const payload = {
    display_name: document.getElementById("editDisplayName").value.trim() || undefined,
    model_name: document.getElementById("editModelName").value.trim() || undefined,
    roles: roles,
    parser_id: document.getElementById("editParserId").value.trim() || undefined,
    status: document.getElementById("editStatus").value,
    settings: settings,
    merge_settings: true
  };

  try {
    const response = await fetch(`/api/models/registry/${encodeURIComponent(modelId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Update failed");
    }
    const modal = bootstrap.Modal.getInstance(document.getElementById("editModelModal"));
    if (modal) modal.hide();
    loadRegisteredModels();
  } catch (e) {
    alert(`Failed to update model: ${e.message}`);
  }
}

async function registerOllamaQuick(name) {
  try {
    const payload = {
      model_id: `ollama:${name}`,
      provider_id: 'ollama',
      model_name: name,
      display_name: name,
      roles: ['deep_scan']
    };
    await fetch("/api/models/registry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    alert("Model Registered");
    loadRegisteredModels();
    loadOllamaModels(false);
  } catch (e) {
    alert("Error: " + e.message);
  }
}

async function deleteModel(id) {
  if (!confirm("CONFIRM_DELETION?")) return;
  await fetch(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
  loadRegisteredModels();
  loadOllamaModels(false);
}

// Utils
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i)) + ' ' + sizes[i];
}

async function pullOllamaModel() {
  const name = document.getElementById("ollamaModelName").value;
  if (!name) return;
  document.getElementById("pullProgress").classList.remove("d-none");

  try {
    await fetch("/api/models/ollama/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: name })
    });
    alert("Download Started. Check server logs or wait for completion.");
    loadOllamaModels(true);
  } catch (e) {
    alert("Error: " + e.message);
  }
  bootstrap.Modal.getInstance(document.getElementById("pullOllamaModal")).hide();
}

async function addCloudModel() {
  alert("Function disabled in tactical mode.");
}

function seedBuiltinModels() {
  // Legacy support
  loadHuggingFaceModels();
}

// HF Modal Helpers
async function registerHFPreset(presetId, options = {}) {
  // Mock implementation for demo logic or real call if backend supports
  alert("Installing " + presetId + "... (This may take a while)");
}

function openHFPresetModal(presetId) {
  // Direct install for now to keep it simple
  if (confirm(`Install HuggingFace model ${presetId}? This will download large files.`)) {
    // In real app, call API. For now, we simulate success or call existing endpoint
    // Assuming backend exists:
    fetch("/api/models/hf/register_preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: presetId })
    }).then(r => {
      if (r.ok) { alert("Install started!"); loadHuggingFaceModels(); loadRegisteredModels(); }
      else alert("Failed to start install.");
    });
  }
}

function registerHFPresetFromModal() { } // formatting
