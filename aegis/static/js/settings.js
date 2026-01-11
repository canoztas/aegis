/**
 * Settings page JavaScript - Credential management and usage tracking
 */

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadCredentialStatus();
  loadUsageData();
  initializeEventListeners();
});

function initializeEventListeners() {
  // Credential form submissions
  document.querySelectorAll(".credential-form").forEach((form) => {
    form.addEventListener("submit", handleCredentialSave);
  });

  // Validate buttons
  document.querySelectorAll(".validate-btn").forEach((btn) => {
    btn.addEventListener("click", handleCredentialValidate);
  });

  // Budget form
  const budgetForm = document.getElementById("budgetForm");
  if (budgetForm) {
    budgetForm.addEventListener("submit", handleBudgetSave);
  }

  // Alert threshold slider
  const alertSlider = document.getElementById("alertThreshold");
  if (alertSlider) {
    alertSlider.addEventListener("input", (e) => {
      document.getElementById("thresholdValue").textContent = e.target.value + "%";
    });
  }

  // Tab change events
  document.querySelectorAll('button[data-bs-toggle="tab"]').forEach((tab) => {
    tab.addEventListener("shown.bs.tab", (e) => {
      if (e.target.id === "usage-tab") {
        loadUsageData();
      }
    });
  });
}

// Toggle password visibility
function togglePasswordVisibility(inputId) {
  const input = document.getElementById(inputId);
  const button = input.nextElementSibling;
  const icon = button.querySelector("i");

  if (input.type === "password") {
    input.type = "text";
    icon.classList.remove("bi-eye");
    icon.classList.add("bi-eye-slash");
  } else {
    input.type = "password";
    icon.classList.remove("bi-eye-slash");
    icon.classList.add("bi-eye");
  }
}

// Load credential status for all providers
async function loadCredentialStatus() {
  try {
    const response = await fetch("/api/credentials");
    if (!response.ok) throw new Error("Failed to load credentials");

    const data = await response.json();
    const credentials = data.credentials || [];

    // Update status for each provider
    ["openai", "anthropic", "google"].forEach((provider) => {
      const cred = credentials.find((c) => c.provider === provider);
      const statusDiv = document.getElementById(`${provider}-status`);

      if (cred) {
        statusDiv.innerHTML = `
          <small class="text-muted font-monospace">
            Status: <span class="text-success"><i class="bi bi-check-circle-fill me-1"></i>Configured</span>
          </small>
          <div class="mt-2">
            <button class="btn btn-sm btn-outline-danger font-monospace" onclick="deleteCredential('${provider}', 'api_key')">
              <i class="bi bi-trash me-1"></i> Delete
            </button>
          </div>
        `;
      }
    });
  } catch (error) {
    console.error("Error loading credential status:", error);
  }
}

// Handle credential save
async function handleCredentialSave(e) {
  e.preventDefault();
  const form = e.target;
  const provider = form.dataset.provider;
  const inputId = `${provider}ApiKey`;
  const apiKey = document.getElementById(inputId).value.trim();

  if (!apiKey) {
    showToast("Error", `Please enter ${provider} API key`, "danger");
    return;
  }

  const button = form.querySelector('button[type="submit"]');
  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="bi bi-hourglass-split me-1"></i> Saving...';

  try {
    const response = await fetch("/api/credentials", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provider,
        key_name: "api_key",
        key_value: apiKey,
        encrypt: true,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to save credential");
    }

    showToast("Success", `${provider} API key saved successfully`, "success");
    document.getElementById(inputId).value = "";
    loadCredentialStatus();
  } catch (error) {
    showToast("Error", error.message, "danger");
  } finally {
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Handle credential validation
async function handleCredentialValidate(e) {
  const button = e.target.closest("button");
  const provider = button.dataset.provider;
  const inputId = `${provider}ApiKey`;
  const apiKey = document.getElementById(inputId).value.trim();

  if (!apiKey) {
    showToast("Error", `Please enter ${provider} API key`, "warning");
    return;
  }

  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="bi bi-hourglass-split me-1"></i> Validating...';

  try {
    const response = await fetch("/api/credentials/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provider,
        api_key: apiKey,
      }),
    });

    const data = await response.json();

    if (data.valid) {
      showToast(
        "Valid API Key",
        data.message || `${provider} API key is valid`,
        "success"
      );
    } else {
      throw new Error(data.error || "Invalid API key");
    }
  } catch (error) {
    showToast("Validation Failed", error.message, "danger");
  } finally {
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Delete credential
async function deleteCredential(provider, keyName) {
  if (!confirm(`Delete ${provider} API key?`)) return;

  try {
    const response = await fetch(`/api/credentials/${provider}/${keyName}`, {
      method: "DELETE",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to delete credential");
    }

    showToast("Success", `${provider} credential deleted`, "success");
    loadCredentialStatus();
  } catch (error) {
    showToast("Error", error.message, "danger");
  }
}

// Load usage data
async function loadUsageData() {
  try {
    // Calculate date 30 days ago
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);

    const params = new URLSearchParams({
      start_date: startDate.toISOString().split("T")[0],
      end_date: endDate.toISOString().split("T")[0],
    });

    const response = await fetch(`/api/credentials/usage?${params}`);
    if (!response.ok) throw new Error("Failed to load usage data");

    const data = await response.json();

    // Update summary cards
    document.getElementById("totalCost").textContent =
      "$" + (data.total_cost_usd || 0).toFixed(4);
    document.getElementById("totalRequests").textContent =
      data.total_requests || 0;
    document.getElementById("totalTokens").textContent = (
      data.total_tokens || 0
    ).toLocaleString();

    // Update usage table
    const tbody = document.getElementById("usageTableBody");
    const byProvider = data.by_provider || data.models || [];
    if (!byProvider || byProvider.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="5" class="text-center text-muted py-4">
            <i class="bi bi-inbox me-2"></i>No usage data available
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = byProvider
      .map(
        (item) => `
      <tr>
        <td class="ps-4">
          <span class="badge bg-${getProviderColor(
            item.provider
          )} bg-opacity-10 text-${getProviderColor(
          item.provider
        )} border border-${getProviderColor(item.provider)} border-opacity-25">
            ${item.provider.toUpperCase()}
          </span>
        </td>
        <td class="text-light">${item.model_name}</td>
        <td class="text-end">${item.requests || item.request_count || 0}</td>
        <td class="text-end">${(item.total_tokens || 0).toLocaleString()}</td>
        <td class="text-end pe-4 text-success fw-bold">$${(
          item.cost_usd || 0
        ).toFixed(4)}</td>
      </tr>
    `
      )
      .join("");
  } catch (error) {
    console.error("Error loading usage data:", error);
    document.getElementById("usageTableBody").innerHTML = `
      <tr>
        <td colspan="5" class="text-center text-danger py-4">
          <i class="bi bi-exclamation-triangle me-2"></i>Error loading usage data
        </td>
      </tr>
    `;
  }
}

// Handle budget save
async function handleBudgetSave(e) {
  e.preventDefault();

  const budgetAmount = parseFloat(
    document.getElementById("budgetAmount").value
  );
  const alertThreshold = parseInt(
    document.getElementById("alertThreshold").value
  );

  if (!budgetAmount || budgetAmount <= 0) {
    showToast("Error", "Please enter a valid budget amount", "warning");
    return;
  }

  const button = e.target.querySelector('button[type="submit"]');
  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="bi bi-hourglass-split me-1"></i> Saving...';

  try {
    // Calculate start date (beginning of current month)
    const now = new Date();
    const startDate = new Date(now.getFullYear(), now.getMonth(), 1);

    const response = await fetch("/api/credentials/budget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        budget_usd: budgetAmount,
        start_date: startDate.toISOString().split("T")[0],
        alert_threshold: alertThreshold,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to check budget");
    }

    // Update budget status display
    const budgetStatus = document.getElementById("budgetStatus");
    budgetStatus.style.display = "block";

    const currentUsage = data.current_cost || 0;
    const percentage = (currentUsage / budgetAmount) * 100;

    document.getElementById("currentUsage").textContent =
      "$" + currentUsage.toFixed(4);

    const progressBar = document.getElementById("budgetProgress");
    progressBar.style.width = percentage + "%";

    if (percentage >= 100) {
      progressBar.classList.remove("bg-primary", "bg-warning");
      progressBar.classList.add("bg-danger");
      document.getElementById("budgetMessage").textContent =
        "Budget exceeded! Consider increasing budget or reducing usage.";
      document.getElementById("budgetMessage").className =
        "text-danger font-monospace fw-bold";
    } else if (percentage >= alertThreshold) {
      progressBar.classList.remove("bg-primary", "bg-danger");
      progressBar.classList.add("bg-warning");
      document.getElementById("budgetMessage").textContent =
        `Warning: ${percentage.toFixed(
          1
        )}% of budget used. Approaching threshold.`;
      document.getElementById("budgetMessage").className =
        "text-warning font-monospace fw-bold";
    } else {
      progressBar.classList.remove("bg-warning", "bg-danger");
      progressBar.classList.add("bg-primary");
      document.getElementById("budgetMessage").textContent =
        `Budget on track. ${(100 - percentage).toFixed(
          1
        )}% remaining this month.`;
      document.getElementById("budgetMessage").className =
        "text-muted font-monospace";
    }

    showToast("Success", "Budget configuration saved", "success");
  } catch (error) {
    showToast("Error", error.message, "danger");
  } finally {
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Utility: Get provider color
function getProviderColor(provider) {
  const colors = {
    openai: "success",
    anthropic: "primary",
    google: "info",
  };
  return colors[provider] || "secondary";
}

// Show toast notification
function showToast(title, message, type = "info") {
  // Create toast container if it doesn't exist
  let container = document.getElementById("toastContainer");
  if (!container) {
    container = document.createElement("div");
    container.id = "toastContainer";
    container.className = "toast-container position-fixed top-0 end-0 p-3";
    container.style.zIndex = "9999";
    document.body.appendChild(container);
  }

  // Create toast
  const toast = document.createElement("div");
  toast.className = `toast align-items-center text-bg-${type} border-0`;
  toast.setAttribute("role", "alert");
  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body font-monospace">
        <strong>${title}:</strong> ${message}
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  `;

  container.appendChild(toast);
  const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
  bsToast.show();

  // Remove from DOM after hide
  toast.addEventListener("hidden.bs.toast", () => {
    toast.remove();
  });
}
