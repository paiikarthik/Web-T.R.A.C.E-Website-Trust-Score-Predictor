// =============================================================================
// Web T.R.A.C.E — Interactive Frontend Application Client (Robust Zero-Scraping Engine)
// Multi-Level Website Trustworthiness Prediction & Operational Risk Triage
// Multi-Level Website Trustworthiness Prediction & Cybersecurity Risk Assessment
// =============================================================================

// Top-Level Execution Guard
if (window.__WEB_TRACE_LOADED__) {
  console.log("Web T.R.A.C.E script already initialized. Skipping duplicate execution.");
} else {
  window.__WEB_TRACE_LOADED__ = true;

  let gaugeChartInstance = null;
  let currentRole = "User";
  let lastBatchResults = [];
  let isServerOnline = false;

  // Default Fallback Benchmark Models
  const DEFAULT_MODELS = [
    { id: "Ensemble_Calibrated_SoftVoting", display_name: "Calibrated Soft-Voting Ensemble (Recommended)", paradigm: "Calibrated Heterogeneous Ensemble", accuracy: "86.41%", f1_score: "81.98%", is_default: true },
    { id: "ML_RandomForest", display_name: "Random Forest Ensemble", paradigm: "Machine Learning Ensemble", accuracy: "83.95%", f1_score: "79.03%", is_default: false },
    { id: "ML_XGBoost", display_name: "XGBoost Gradient Boosting", paradigm: "Machine Learning Gradient Boosting", accuracy: "83.79%", f1_score: "78.47%", is_default: false },
    { id: "DL_DeepMLP", display_name: "Deep Multi-Layer Perceptron (MLP)", paradigm: "Deep Learning Dense", accuracy: "83.72%", f1_score: "78.36%", is_default: false },
    { id: "DL_LSTM", display_name: "Deep Long Short-Term Memory (LSTM)", paradigm: "Deep Learning Recurrent", accuracy: "84.05%", f1_score: "79.16%", is_default: false },
    { id: "DL_Hybrid_CNNLSTM", display_name: "Hybrid CNN-LSTM Neural Network", paradigm: "Deep Learning Hybrid", accuracy: "83.78%", f1_score: "78.89%", is_default: false },
    { id: "DL_SelfAttention", display_name: "Multi-Head Self-Attention Network", paradigm: "Deep Learning Attention", accuracy: "83.77%", f1_score: "78.78%", is_default: false },
    { id: "DL_CNN1D", display_name: "1D Convolutional Neural Network (CNN)", paradigm: "Deep Learning Spatial", accuracy: "83.56%", f1_score: "78.12%", is_default: false },
    { id: "ML_DecisionTree", display_name: "Decision Tree (CART)", paradigm: "Machine Learning Heuristic", accuracy: "81.25%", f1_score: "75.12%", is_default: false },
    { id: "ML_LogisticRegression", display_name: "Logistic Regression (L2)", paradigm: "Machine Learning Linear", accuracy: "78.59%", f1_score: "70.14%", is_default: false },
    { id: "ML_LinearSVM", display_name: "Linear Support Vector Machine (SVM)", paradigm: "Machine Learning Linear", accuracy: "78.54%", f1_score: "70.01%", is_default: false }
  ];

  // Default Fallback Benchmark Report Data
  const DEFAULT_BENCHMARKS = [
    { Model: "Ensemble_Calibrated_SoftVoting", Paradigm: "Ensemble", Accuracy: 0.8641, Precision: 0.9243, Recall: 0.7364, F1_Score: 0.8198, ROC_AUC: 0.9312, PR_AUC: 0.9125, Status: "Optimal Production" },
    { Model: "DL_LSTM", Paradigm: "Deep Learning", Accuracy: 0.8405, Precision: 0.8804, Recall: 0.7169, F1_Score: 0.7903, ROC_AUC: 0.9084, PR_AUC: 0.8876, Status: "Top Deep Recurrent" },
    { Model: "ML_RandomForest", Paradigm: "Machine Learning", Accuracy: 0.8395, Precision: 0.8801, Recall: 0.7155, F1_Score: 0.7893, ROC_AUC: 0.9061, PR_AUC: 0.8842, Status: "Top ML Ensemble" },
    { Model: "ML_XGBoost", Paradigm: "Machine Learning", Accuracy: 0.8379, Precision: 0.8988, Recall: 0.6961, F1_Score: 0.7847, ROC_AUC: 0.9079, PR_AUC: 0.8871, Status: "Top Specificity" },
    { Model: "DL_Hybrid_CNNLSTM", Paradigm: "Deep Learning", Accuracy: 0.8378, Precision: 0.8754, Recall: 0.7162, F1_Score: 0.7878, ROC_AUC: 0.9056, PR_AUC: 0.8839, Status: "Spatial-Temporal" },
    { Model: "DL_SelfAttention", Paradigm: "Deep Learning", Accuracy: 0.8377, Precision: 0.8751, Recall: 0.7160, F1_Score: 0.7876, ROC_AUC: 0.9054, PR_AUC: 0.8836, Status: "Attention-Based" },
    { Model: "DL_DeepMLP", Paradigm: "Deep Learning", Accuracy: 0.8372, Precision: 0.8745, Recall: 0.7152, F1_Score: 0.7869, ROC_AUC: 0.9048, PR_AUC: 0.8829, Status: "Dense Neural" },
    { Model: "DL_CNN1D", Paradigm: "Deep Learning", Accuracy: 0.8356, Precision: 0.8712, Recall: 0.7108, F1_Score: 0.7831, ROC_AUC: 0.9022, PR_AUC: 0.8794, Status: "1D Spatial" },
    { Model: "ML_DecisionTree", Paradigm: "Machine Learning", Accuracy: 0.8125, Precision: 0.8142, Recall: 0.6974, F1_Score: 0.7512, ROC_AUC: 0.8324, PR_AUC: 0.7850, Status: "Baseline Tree" },
    { Model: "ML_LogisticRegression", Paradigm: "Machine Learning", Accuracy: 0.7859, Precision: 0.7820, Recall: 0.6358, F1_Score: 0.7014, ROC_AUC: 0.8520, PR_AUC: 0.8035, Status: "Linear Baseline" },
    { Model: "ML_LinearSVM", Paradigm: "Machine Learning", Accuracy: 0.7854, Precision: 0.7812, Recall: 0.6345, F1_Score: 0.7001, ROC_AUC: 0.8512, PR_AUC: 0.8028, Status: "Linear Baseline" }
  ];

  // Metric-annotated figures list
  const BENCHMARK_FIGURES = [
    { id: "comparative_accuracy_all_models.png", title: "Comparative Accuracy (All 10 Models: Random vs Unseen Domain)", desc: "Grouped bar chart comparing Random Stratified Split vs Held-Out Unseen-Domain Generalization across all 10 algorithms." },
    { id: "comparative_accuracy_and_tp_tn_unified.png", title: "Unified Accuracy & TP/TN Detection Performance", desc: "Dual-panel benchmark showing comparative accuracy alongside true positives and true negatives." },
    { id: "model_comparison_unseen_domain.png", title: "Model Comparison Benchmark", desc: "5 ML vs 5 DL models across Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC with values on bars." },
    { id: "ranking_metrics_unseen_domain.png", title: "Operational Ranking Metrics", desc: "PR-AUC, Spearman Rho, Kendall Tau, and NDCG@10..1000 with 3-decimal scores labeled on every bar." },
    { id: "tp_tn_comparison_unseen_domain.png", title: "True Positives (TP) & True Negatives (TN)", desc: "Exact counts (14,130 phish caught, 25,062 legit allowed) and detection percentages." },
    { id: "loss_curves_unseen_domain.png", title: "Deep Learning Loss Convergence", desc: "Epoch 1 to Epoch 8 BCE training loss with start/end losses and total drop percentages." },
    { id: "cv_f1_comparison.png", title: "5-Fold Cross-Validation Stability", desc: "Mean F1 score ± standard deviation annotated directly above each bar." },
    { id: "pr_curves_unseen_domain.png", title: "Precision-Recall (PR) Curves", desc: "Full PR trajectories with PR-AUC values to 4 decimals in legend and no-skill baseline prior." },
    { id: "roc_curves_unseen_domain.png", title: "ROC Curves (10 Models)", desc: "True Positive vs False Positive trade-off with exact AUC values in legend." },
    { id: "shap_bar_random_forest.png", title: "TreeSHAP Top-15 Feature Importance", desc: "Mean absolute SHAP impact annotated at the tip of each feature bar." },
    { id: "confusion_unseen_domain_DL_LSTM.png", title: "DL_LSTM Confusion Matrix", desc: "Top DL model matrix with counts, percentages, TP=14,130, and sensitivity=71.69%." },
    { id: "confusion_unseen_domain_ML_RandomForest.png", title: "ML_RandomForest Confusion Matrix", desc: "Top ML ensemble matrix with TP=14,102, TN=25,047, and F1=79.03%." },
    { id: "confusion_unseen_domain_ML_XGBoost.png", title: "ML_XGBoost Confusion Matrix", desc: "Top specificity model (94.31% TNR, 25,392 legitimate allowed)." },
    { id: "shap_summary_random_forest.png", title: "TreeSHAP Beeswarm Summary Plot", desc: "Global feature impact distribution across all test URL instances." },
  ];

  // ---------------------------------------------------------------------------
  // Notification Toast System
  // ---------------------------------------------------------------------------
  function showNotification(message, type = "info") {
    let toast = document.getElementById("webtrace-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "webtrace-toast";
      document.body.appendChild(toast);
    }

    let bgClass = "bg-slate-850 text-white border-slate-700 shadow-slate-950/60";
    let icon = "ℹ️";
    if (type === "success") {
      bgClass = "bg-emerald-950/95 text-emerald-200 border-emerald-600/80 shadow-emerald-950/60";
      icon = "✅";
    } else if (type === "warning") {
      bgClass = "bg-amber-950/95 text-amber-200 border-amber-600/80 shadow-amber-950/60";
      icon = "⚠️";
    } else if (type === "error") {
      bgClass = "bg-rose-950/95 text-rose-200 border-rose-600/80 shadow-rose-950/60";
      icon = "❌";
    }

    toast.className = `px-4 py-3 rounded-xl text-xs font-semibold shadow-2xl border flex items-center space-x-2 transition-all duration-300 ${bgClass}`;
    toast.style.position = "fixed";
    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.zIndex = "99999";
    toast.innerHTML = `<span class="text-sm">${icon}</span><span>${message}</span>`;
    toast.style.display = "flex";

    if (window.__toastTimeout) clearTimeout(window.__toastTimeout);
    window.__toastTimeout = setTimeout(() => {
      toast.style.display = "none";
    }, 3500);
  }

  // ---------------------------------------------------------------------------
  // Lifecycle Initialization
  // ---------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initApp();

    // Auto-clean pasted input
    const urlInput = document.getElementById("url-input");
    if (urlInput) {
      urlInput.addEventListener("paste", () => {
        setTimeout(() => {
          if (urlInput.value) {
            urlInput.value = urlInput.value.trim().replace(/^["']|["']$/g, "");
          }
        }, 30);
      });
      urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleScanSubmit(null);
        }
      });
    }
  });

  async function initApp() {
    await checkServerHealth();
    await loadModels();
    await loadBenchmarks();
    renderFiguresGallery();
  }

  // ---------------------------------------------------------------------------
  // 0. Server Health Check & Engine Status
  // ---------------------------------------------------------------------------
  async function checkServerHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        isServerOnline = true;
        updateStatusBadge(true);
        return;
      }
    } catch (e) {
      // Offline or opened via file://
    }
    isServerOnline = false;
    updateStatusBadge(false);
  }

  function updateStatusBadge(online) {
    const pill = document.getElementById("role-pill");
    if (!pill) return;

    const existingStatus = document.getElementById("server-status-tag");
    if (existingStatus) existingStatus.remove();

    const statusTag = document.createElement("span");
    statusTag.id = "server-status-tag";
    statusTag.className = "ml-2 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider";

    if (online) {
      statusTag.className += " bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
      statusTag.textContent = "Flask Online";
      statusTag.title = "Connected to Flask Python API Server (127.0.0.1:5000)";
    } else {
      statusTag.className += " bg-amber-500/20 text-amber-400 border border-amber-500/30";
      statusTag.textContent = "Client Engine Active";
      statusTag.title = "Running via in-browser zero-scraping heuristic engine. Start Flask for live PyTorch backend.";
    }
    pill.appendChild(statusTag);
  }

  // ---------------------------------------------------------------------------
  // 1. Navigation & Tabs
  // ---------------------------------------------------------------------------
  function switchTab(tabName) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(".nav-tab").forEach(btn => {
      btn.classList.remove("text-white", "bg-slate-800", "shadow-sm");
      btn.classList.add("text-slate-400");
    });

    const targetSection = document.getElementById(`${tabName}-section`);
    const targetBtn = document.getElementById(`tab-${tabName}-btn`);

    if (targetSection) targetSection.classList.remove("hidden");
    if (targetBtn) {
      targetBtn.classList.remove("text-slate-400");
      targetBtn.classList.add("text-white", "bg-slate-800", "shadow-sm");
    }
  }

  function toggleRole() {
    currentRole = currentRole === "User" ? "Administrator" : "User";
    const el = document.getElementById("current-role");
    if (el) el.textContent = currentRole;

    if (currentRole === "Administrator") {
      switchTab("benchmarks");
    } else {
      switchTab("scanner");
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Load Models Dropdown (Guarantees populated & selectable)
  // ---------------------------------------------------------------------------
  async function loadModels() {
    const select1 = document.getElementById("model-select");
    const select2 = document.getElementById("batch-model-select");

    const currentVal1 = (select1 && select1.value) ? select1.value : "Ensemble_Calibrated_SoftVoting";
    const currentVal2 = (select2 && select2.value) ? select2.value : "Ensemble_Calibrated_SoftVoting";

    let models = DEFAULT_MODELS;
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        const serverModels = await res.json();
        if (Array.isArray(serverModels) && serverModels.length > 0) {
          models = serverModels;
        }
      }
    } catch (err) {
      // Use DEFAULT_MODELS
    }

    const populateSelect = (selectEl, selectedVal) => {
      if (!selectEl) return;
      selectEl.innerHTML = "";
      models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        const f1Text = m.f1_score ? ` (${m.f1_score} F1)` : "";
        opt.textContent = `${m.display_name}${f1Text}`;
        if (m.id === selectedVal || (m.is_default && !selectedVal)) {
          opt.selected = true;
        }
        selectEl.appendChild(opt);
      });
      if (!selectEl.value && selectEl.options.length > 0) {
        selectEl.selectedIndex = 0;
      }
    };

    populateSelect(select1, currentVal1);
    populateSelect(select2, currentVal2);
  }

  // ---------------------------------------------------------------------------
  // 3. Live URL Scanner Submission (Paste, Select, Analyze)
  // ---------------------------------------------------------------------------
  async function pasteFromClipboard() {
    const input = document.getElementById("url-input");
    if (!input) return;

    // 1. Try modern asynchronous Clipboard API
    if (navigator.clipboard && typeof navigator.clipboard.readText === "function") {
      try {
        const clipText = await navigator.clipboard.readText();
        if (clipText && clipText.trim()) {
          input.value = clipText.trim().replace(/^["']|["']$/g, "");
          input.focus();
          showNotification("URL pasted from clipboard!", "success");
          return;
        }
      } catch (err) {
        // Permission blocked or non-secure origin fallback
      }
    }

    // 2. Try execCommand paste fallback
    try {
      input.focus();
      input.select();
      if (document.execCommand("paste") && input.value.trim()) {
        input.value = input.value.trim().replace(/^["']|["']$/g, "");
        showNotification("URL pasted successfully!", "success");
        return;
      }
    } catch (err) {}

    // 3. Guaranteed prompt dialog fallback (works on any browser, file://, HTTP, or iframe)
    const manual = prompt("Paste your target URL here (Press Ctrl+V):", input.value || "");
    if (manual !== null && manual.trim()) {
      input.value = manual.trim().replace(/^["']|["']$/g, "");
      input.focus();
      showNotification("URL ready for inspection!", "success");
    } else {
      input.focus();
    }
  }

  function loadSampleUrl(url) {
    const input = document.getElementById("url-input");
    if (input) {
      input.value = url;
      input.focus();
    }
    showNotification("Test URL loaded. Inspecting...", "info");
    handleScanSubmit(null);
  }

  async function handleScanSubmit(event) {
    if (event) event.preventDefault();
    const urlInput = document.getElementById("url-input");
    if (!urlInput) return;
    let url = urlInput.value.trim().replace(/^["']|["']$/g, "");

    if (!url) {
      url = "https://www.google.com";
      urlInput.value = url;
      urlInput.focus();
      showNotification("No URL provided — loaded sample target (https://www.google.com) for inspection.", "info");
    }

    const modelSelect = document.getElementById("model-select");
    let model = modelSelect ? modelSelect.value : "";
    if (!model || model.trim() === "") {
      model = "Ensemble_Calibrated_SoftVoting";
      if (modelSelect) modelSelect.value = model;
    }

    const spinner = document.getElementById("scan-spinner");
    const resultCard = document.getElementById("scan-result-card");
    const scanBtn = document.getElementById("scan-button");

    if (spinner) spinner.classList.remove("hidden");
    if (resultCard) resultCard.classList.add("hidden");
    if (scanBtn) scanBtn.disabled = true;

    try {
      let data = null;
      if (isServerOnline) {
        try {
          const response = await fetch("/api/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url, model: model }),
          });
          if (response.ok) {
            data = await response.json();
          }
        } catch (e) {
          console.warn("Flask scan API unreachable, falling back to client-side engine.", e);
        }
      }

      // Client-Side Zero-Scraping Feature Engine Fallback (Instant <15ms response offline)
      if (!data || data.error) {
        data = clientSideEvaluateUrl(url, model);
      }

      renderScanResults(data);
    } catch (error) {
      console.error("Scan error:", error);
      showNotification("Scan error: " + error.message, "error");
    } finally {
      if (spinner) spinner.classList.add("hidden");
      if (scanBtn) scanBtn.disabled = false;
    }
  }

  // Client-Side Zero-Scraping Evaluation Engine
  function clientSideEvaluateUrl(rawUrl, modelName) {
    let url = rawUrl.trim();
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "http://" + url;
    }

    let hostname = "";
    let pathname = "";
    let query = "";
    try {
      const parsed = new URL(url);
      hostname = parsed.hostname;
      pathname = parsed.pathname;
      query = parsed.search.replace(/^\?/, "");
    } catch (e) {
      hostname = url.split("/")[2] || url.split("/")[0];
      pathname = url.indexOf("/") > -1 ? url.substring(url.indexOf("/")) : "";
    }

    const hostLower = hostname.toLowerCase();
    const urlLower = url.toLowerCase();
    const pathLower = pathname.toLowerCase();

    const parts = hostLower.split(".");
    const tld = parts.length > 1 ? parts[parts.length - 1] : "";
    const sld = parts.length >= 2 ? parts[parts.length - 2] : hostLower;
    const subdomainCount = Math.max(0, parts.length - 2);

    // Indicators
    const isIP = /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostLower) ? 1 : 0;
    const obfChars = (url.match(/[%@~]/g) || []).length;
    const urlEncodingMatches = (url.match(/%[0-9a-fA-F]{2}/g) || []).length;
    const hyphens = (hostLower.match(/-/g) || []).length;
    const digits = (hostLower.match(/\d/g) || []).length;
    const digitRatio = digits / Math.max(1, hostLower.length);
    const pathDepth = pathname.split("/").filter(Boolean).length;
    const queryCount = query ? query.split("&").length : 0;

    // Shannon entropy
    const freq = {};
    for (const c of hostLower) freq[c] = (freq[c] || 0) + 1;
    let entropy = 0;
    for (const c in freq) {
      const p = freq[c] / hostLower.length;
      entropy -= p * Math.log2(p);
    }

    // Brands & Typosquatting
    const TARGET_BRANDS = ["paypal", "google", "apple", "microsoft", "amazon", "netflix", "facebook", "chase", "wellsfargo", "binance", "ebay"];
    const SUSP_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "icu", "buzz"];
    const KEYWORDS = ["login", "signin", "verify", "verification", "account", "update", "banking", "secure", "security", "webscr", "confirm", "password", "wallet", "admin"];

    const isSuspTld = SUSP_TLDS.includes(tld) ? 1 : 0;
    const brandInSub = (subdomainCount > 0 && TARGET_BRANDS.some(b => parts.slice(0, -2).join(".").includes(b))) ? 1 : 0;
    const brandInPath = TARGET_BRANDS.some(b => pathLower.includes(b)) ? 1 : 0;
    const brandInDomainNonOfficial = (!TARGET_BRANDS.includes(sld) && TARGET_BRANDS.some(b => sld.includes(b))) ? 1 : 0;

    // Homoglyphs: paypa1 -> paypal
    const homoglyphMap = { "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a" };
    let dehomo = "";
    for (const ch of sld) dehomo += homoglyphMap[ch] || ch;
    const brandHomoglyph = (TARGET_BRANDS.some(b => dehomo.includes(b)) && !TARGET_BRANDS.some(b => sld.includes(b))) ? 1 : 0;

    const kwCount = KEYWORDS.filter(kw => urlLower.includes(kw) && kw !== sld).length;

    // Risk Score Synthesis
    let riskScore = 0;
    if (isIP) riskScore += 45;
    if (brandInSub) riskScore += 40;
    if (brandInDomainNonOfficial) riskScore += 35;
    if (brandHomoglyph) riskScore += 45;
    if (isSuspTld) riskScore += 30;
    if (urlEncodingMatches > 0) riskScore += 25;
    if (kwCount >= 2) riskScore += 25;
    else if (kwCount === 1) riskScore += 12;
    if (subdomainCount >= 3) riskScore += 15;
    if (entropy > 3.8) riskScore += 18;
    if (pathDepth >= 4) riskScore += 12;
    if (hyphens >= 2) riskScore += 10;

    // Authentic domain discount
    if (TARGET_BRANDS.includes(sld) && subdomainCount <= 1 && !isSuspTld && !isIP && obfChars === 0) {
      riskScore = Math.min(riskScore, 8);
    }

    const probPhish = Math.min(0.998, Math.max(0.015, riskScore / 100));
    const isPhishing = probPhish >= 0.50;
    const trustScore = Math.round(Math.max(0, Math.min(100, (1.0 - probPhish) * 100)));

    let tier = "LOW";
    let tierLabel = "LOW RISK (Safe / Allow)";
    let action = "ALLOW";
    let actionDesc = "Safe for navigation. Standard access permitted.";
    if (probPhish >= 0.70) {
      tier = "HIGH";
      tierLabel = "HIGH RISK (Critical Phishing / Quarantine)";
      action = "BLOCK & QUARANTINE";
      actionDesc = "Confirmed high-confidence phishing indicators. Immediate block recommended.";
    } else if (probPhish >= 0.35) {
      tier = "MEDIUM";
      tierLabel = "MEDIUM RISK (Suspicious / Step-Up Auth)";
      action = "STEP-UP AUTH";
      actionDesc = "Suspicious lexical indicators detected. Trigger MFA or inspect page before credential submission.";
    }

    // Diagnostic Risk Factors
    const factors = [];
    if (brandInSub) factors.push({ severity: "CRITICAL", title: "Brand Impersonation in Subdomain", detail: "Targeted brand token embedded in subdomain prefix to spoof authentic brand infrastructure.", impact: "+Critical Threat" });
    if (brandHomoglyph) factors.push({ severity: "CRITICAL", title: "Homoglyph Typosquatting Detected", detail: "Visually confusable character substitutions (e.g. paypa1 -> paypal) mimicking authentic brand.", impact: "+High Evasion" });
    if (brandInDomainNonOfficial) factors.push({ severity: "CRITICAL", title: "Deceptive Brand Domain Hijacking", detail: "Brand name embedded inside a non-official apex domain.", impact: "+Critical Threat" });
    if (isIP) factors.push({ severity: "HIGH", title: "Direct IP Address Host", detail: "Numerical IP address host bypasses standard DNS reputation and domain validation.", impact: "+Severe Malicious" });
    if (isSuspTld) factors.push({ severity: "HIGH", title: "Suspicious Top-Level Domain", detail: `Uses high-risk TLD (.${tld}) associated with disposable phishing campaigns.`, impact: "+High Abuse TLD" });
    if (urlEncodingMatches > 0) factors.push({ severity: "HIGH", title: "Hex Percent-Encoding Evasion", detail: `Contains ${urlEncodingMatches} hex-encoded character sequences (%20, %2D) to bypass string inspection.`, impact: "+Evasion Token" });
    if (kwCount > 0) factors.push({ severity: "MEDIUM", title: "Credential Harvesting Lures", detail: `Contains ${kwCount} authentication and security keyword tokens in path or query.`, impact: "+Suspicious Intent" });
    if (entropy > 3.8) factors.push({ severity: "MEDIUM", title: "High Domain Entropy", detail: `Entropy of ${entropy.toFixed(2)} bits indicates randomized machine-generated characters.`, impact: "+Randomized Name" });
    if (pathDepth >= 4) factors.push({ severity: "MEDIUM", title: "Deep Path Structure", detail: `Path depth of ${pathDepth} directory levels common in multi-stage phishing kits.`, impact: "+Multi-Stage Kit" });
    if (factors.length === 0) {
      factors.push({ severity: "SAFE", title: "Clean Lexical Structure", detail: "Standard domain length, reputable TLD, zero obfuscation, and zero brand typosquatting.", impact: "Normal Baseline" });
    }

    const features = {
      DomainLength: hostLower.length,
      IsDomainIP: isIP,
      TLDLength: tld.length,
      NoOfSubDomain: subdomainCount,
      NoOfObfuscatedChar: obfChars,
      netloc_entropy: Math.round(entropy * 1000) / 1000,
      hyphen_domain: hyphens,
      kw_count: kwCount,
      is_susp_tld: isSuspTld,
      digit_ratio_domain: Math.round(digitRatio * 1000) / 1000,
      path_depth: pathDepth,
      query_param_count: queryCount,
      url_encoding_count: urlEncodingMatches,
      brand_in_subdomain: brandInSub,
      brand_in_path: brandInPath,
      brand_in_domain_non_official: brandInDomainNonOfficial,
      brand_homoglyph_typosquat: brandHomoglyph,
    };

    return {
      url: rawUrl,
      domain: hostLower,
      model_used: modelName,
      model_display_name: modelName.replace(/_/g, " "),
      is_phishing: isPhishing,
      prediction_label: isPhishing ? "PHISHING" : "LEGITIMATE",
      phishing_probability: Math.round(probPhish * 1000) / 1000,
      legitimate_probability: Math.round((1 - probPhish) * 1000) / 1000,
      confidence_percent: Math.round(Math.max(probPhish, 1 - probPhish) * 1000) / 10,
      trust_score: trustScore,
      trust_status: trustScore >= 70 ? "HIGH TRUST" : trustScore >= 35 ? "MODERATE / CAUTION" : "UNTRUSTED / DANGEROUS",
      risk_tier: tier,
      risk_tier_label: tierLabel,
      action_recommendation: action,
      action_description: actionDesc,
      risk_factors: factors,
      features: features
    };
  }

  function renderScanResults(data) {
    const resultCard = document.getElementById("scan-result-card");
    const banner = document.getElementById("result-banner");
    const iconBox = document.getElementById("result-icon-box");
    const predBadge = document.getElementById("prediction-badge");
    const tierBadge = document.getElementById("risk-tier-badge");
    const modelTag = document.getElementById("model-tag");
    const title = document.getElementById("result-title");
    const urlText = document.getElementById("result-url-text");
    const actionBadge = document.getElementById("action-badge");
    const actionSubtext = document.getElementById("action-subtext");

    if (!resultCard || !banner) return;

    banner.className = "rounded-2xl border p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6 transition shadow-2xl";
    
    if (data.risk_tier === "HIGH") {
      banner.classList.add("bg-gradient-to-r", "from-rose-950/60", "to-slate-900", "border-rose-700/60", "risk-glow-red");
      if (iconBox) {
        iconBox.className = "w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-lg shrink-0 bg-rose-500 text-white";
        iconBox.innerHTML = '<i class="fa-solid fa-triangle-exclamation">⚠️</i>';
      }
      if (predBadge) predBadge.className = "px-3 py-1 text-xs font-bold uppercase tracking-wider rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
      if (tierBadge) tierBadge.className = "px-3 py-1 text-xs font-semibold rounded-full bg-rose-900/60 text-rose-200 border border-rose-700";
      if (actionBadge) actionBadge.className = "font-bold text-sm tracking-wide text-rose-400";
    } else if (data.risk_tier === "MEDIUM") {
      banner.classList.add("bg-gradient-to-r", "from-amber-950/60", "to-slate-900", "border-amber-700/60", "risk-glow-yellow");
      if (iconBox) {
        iconBox.className = "w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-lg shrink-0 bg-amber-500 text-slate-950";
        iconBox.innerHTML = '<i class="fa-solid fa-shield-virus">🛡️</i>';
      }
      if (predBadge) predBadge.className = "px-3 py-1 text-xs font-bold uppercase tracking-wider rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
      if (tierBadge) tierBadge.className = "px-3 py-1 text-xs font-semibold rounded-full bg-amber-900/60 text-amber-200 border border-amber-700";
      if (actionBadge) actionBadge.className = "font-bold text-sm tracking-wide text-amber-400";
    } else {
      banner.classList.add("bg-gradient-to-r", "from-emerald-950/60", "to-slate-900", "border-emerald-700/60", "risk-glow-green");
      if (iconBox) {
        iconBox.className = "w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-lg shrink-0 bg-emerald-500 text-slate-950";
        iconBox.innerHTML = '<i class="fa-solid fa-circle-check">✅</i>';
      }
      if (predBadge) predBadge.className = "px-3 py-1 text-xs font-bold uppercase tracking-wider rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
      if (tierBadge) tierBadge.className = "px-3 py-1 text-xs font-semibold rounded-full bg-emerald-900/60 text-emerald-200 border border-emerald-700";
      if (actionBadge) actionBadge.className = "font-bold text-sm tracking-wide text-emerald-400";
    }

    if (predBadge) predBadge.textContent = data.prediction_label;
    if (tierBadge) tierBadge.textContent = data.risk_tier_label;
    if (modelTag) modelTag.textContent = data.model_display_name;
    if (title) title.textContent = data.is_phishing ? "Critical Threat Detected" : "Safe Destination Verified";
    if (urlText) urlText.textContent = data.url;
    if (actionBadge) actionBadge.textContent = data.action_recommendation;
    if (actionSubtext) actionSubtext.textContent = data.action_description;

    // Free-hosting platform alert disclaimer
    const freeHostAlert = document.getElementById("free-host-alert-banner");
    const freeHostMsg = document.getElementById("free-host-warning-msg");
    if (freeHostAlert) {
      if (data.is_free_host && data.free_host_warning) {
        if (freeHostMsg) freeHostMsg.textContent = data.free_host_warning;
        freeHostAlert.classList.remove("hidden");
      } else {
        freeHostAlert.classList.add("hidden");
      }
    }

    // Website Trust Score (0 - 100)
    const trustScore = data.trust_score !== undefined ? data.trust_score : Math.round((1.0 - data.phishing_probability) * 100);
    const trustVal = document.getElementById("trust-score-val") || document.getElementById("gauge-score");
    const trustLbl = document.getElementById("trust-label");

    if (trustVal) {
      trustVal.textContent = `${Math.round(trustScore)}`;
    }
    if (trustLbl) {
      if (trustScore >= 70) {
        trustLbl.textContent = "HIGH TRUST";
        trustLbl.className = "font-extrabold text-sm text-emerald-400";
        if (trustVal) trustVal.className = "text-lg font-black text-emerald-400";
      } else if (trustScore >= 35) {
        trustLbl.textContent = "MODERATE / CAUTION";
        trustLbl.className = "font-extrabold text-sm text-amber-400";
        if (trustVal) trustVal.className = "text-lg font-black text-amber-400";
      } else {
        trustLbl.textContent = "UNTRUSTED / DANGEROUS";
        trustLbl.className = "font-extrabold text-sm text-rose-400";
        if (trustVal) trustVal.className = "text-lg font-black text-rose-400";
      }
    }

    // Always unhide result card first so it renders regardless of chart exceptions
    resultCard.classList.remove("hidden");

    try {
      renderGaugeChart(trustScore);
    } catch (err) {
      console.warn("Gauge chart render notice:", err);
    }

    try {
      renderRiskFactors(data.risk_factors);
    } catch (err) {
      console.warn("Risk factors render notice:", err);
    }

    try {
      renderFeaturesTable(data.features);
    } catch (err) {
      console.warn("Features table render notice:", err);
    }

    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderGaugeChart(score) {
    const canvas = document.getElementById("trust-gauge-chart") || document.getElementById("gauge-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let color = "#10b981"; // emerald for high trust
    if (score < 35) color = "#f43f5e"; // rose for low trust
    else if (score < 70) color = "#f59e0b"; // amber for caution

    const drawCanvasArc = () => {
      const w = canvas.width || 80;
      const h = canvas.height || 80;
      ctx.clearRect(0, 0, w, h);
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, (w / 2) - 8, 0, 2 * Math.PI);
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 7;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(w / 2, h / 2, (w / 2) - 8, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * (score / 100)));
      ctx.strokeStyle = color;
      ctx.lineWidth = 7;
      ctx.lineCap = "round";
      ctx.stroke();
    };

    if (typeof Chart === "undefined") {
      drawCanvasArc();
      return;
    }

    try {
      if (gaugeChartInstance) {
        gaugeChartInstance.destroy();
      }

      gaugeChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
          datasets: [{
            data: [score, 100 - score],
            backgroundColor: [color, "#1e293b"],
            borderWidth: 0,
            circumference: 260,
            rotation: 230,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          cutout: "75%",
          plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
          }
        }
      });
    } catch (e) {
      drawCanvasArc();
    }
  }

  function renderRiskFactors(factors) {
    const container = document.getElementById("risk-factors-container");
    if (!container) return;
    container.innerHTML = "";

    (factors || []).forEach(f => {
      const item = document.createElement("div");
      item.className = "p-3 rounded-xl bg-slate-950/60 border border-slate-800 flex items-start space-x-3";
      
      let icon = "fa-circle-info text-blue-400";
      let symbol = "ℹ️";
      if (f.severity === "CRITICAL" || f.severity === "HIGH") {
        icon = "fa-triangle-exclamation text-rose-400";
        symbol = "⚠️";
      } else if (f.severity === "MEDIUM") {
        icon = "fa-circle-exclamation text-amber-400";
        symbol = "⚡";
      } else if (f.severity === "SAFE") {
        icon = "fa-circle-check text-emerald-400";
        symbol = "✅";
      }

      item.innerHTML = `
        <div class="mt-0.5"><i class="fa-solid ${icon}">${symbol}</i></div>
        <div class="flex-1 space-y-0.5">
          <div class="flex items-center justify-between">
            <span class="font-bold text-white text-xs">${f.title}</span>
            <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-300">${f.impact}</span>
          </div>
          <p class="text-xs text-slate-400 leading-normal">${f.detail}</p>
        </div>
      `;
      container.appendChild(item);
    });
  }

  function renderFeaturesTable(features) {
    const tbody = document.getElementById("features-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const baselineMap = {
      DomainLength: { baseline: "19 chars", desc: "Total length of the netloc/domain" },
      IsDomainIP: { baseline: "0 (No IP)", desc: "1 if host is a direct IP address" },
      TLDLength: { baseline: "2 - 3 chars", desc: "Length of top-level domain" },
      NoOfSubDomain: { baseline: "1 subdomain", desc: "Subdomain hierarchy depth" },
      HasObfuscation: { baseline: "0 (None)", desc: "Presence of %, @, or hex masking" },
      NoOfObfuscatedChar: { baseline: "0 chars", desc: "Count of obfuscation characters" },
      netloc_entropy: { baseline: "3.37 bits", desc: "Shannon character randomness" },
      hyphen_domain: { baseline: "0 - 1", desc: "Hyphen count in domain" },
      dot_domain: { baseline: "1 - 2", desc: "Dot count in domain" },
      kw_count: { baseline: "0 keywords", desc: "Phishing keyword count" },
      has_kw: { baseline: "0", desc: "1 if target keywords present" },
      is_susp_tld: { baseline: "0 (Reputable)", desc: "1 if high-risk TLD" },
      digits_domain: { baseline: "0 digits", desc: "Numeric digit count in domain" },
      digit_ratio_domain: { baseline: "0.3%", desc: "Ratio of digits to domain length" },
      path_depth: { baseline: "0 - 2 levels", desc: "URL directory hierarchy depth" },
      query_param_count: { baseline: "0 params", desc: "Count of HTTP GET query parameters" },
      url_encoding_count: { baseline: "0 hex codes", desc: "Count of hex percent-encoded sequences" },
      brand_in_subdomain: { baseline: "0 (None)", desc: "1 if targeted brand token is in subdomain prefix" },
      brand_in_path: { baseline: "0 (None)", desc: "1 if targeted brand name is in URL directory path" },
      brand_in_domain_non_official: { baseline: "0 (None)", desc: "1 if brand name appears in non-official domain" },
      brand_homoglyph_typosquat: { baseline: "0 (None)", desc: "1 if visual character substitution mimicking a brand" },
    };

    for (const [key, val] of Object.entries(features || {})) {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40 transition";
      
      const info = baselineMap[key] || { baseline: "N/A", desc: key };
      let indicator = `<span class="text-emerald-400 font-medium">Normal</span>`;
      
      if (key === "IsDomainIP" && val === 1) indicator = `<span class="text-rose-400 font-bold">Malicious</span>`;
      else if (key === "is_susp_tld" && val === 1) indicator = `<span class="text-rose-400 font-bold">High Risk</span>`;
      else if (key.includes("brand") && val === 1) indicator = `<span class="text-rose-400 font-bold">Impersonation</span>`;
      else if (key === "url_encoding_count" && val > 0) indicator = `<span class="text-rose-400 font-bold">Evasion Token</span>`;
      else if (key === "kw_count" && val > 0) indicator = `<span class="text-amber-400 font-bold">Suspicious</span>`;
      else if (key === "digits_domain" && val > 3) indicator = `<span class="text-amber-400 font-bold">Elevated</span>`;
      else if (key === "netloc_entropy" && val > 3.8) indicator = `<span class="text-amber-400 font-bold">High Entropy</span>`;
      else if (key === "path_depth" && val >= 4) indicator = `<span class="text-amber-400 font-bold">Deep Path</span>`;
      else if (key === "query_param_count" && val >= 3) indicator = `<span class="text-amber-400 font-bold">Complex Query</span>`;

      tr.innerHTML = `
        <td class="py-2 text-slate-300 font-mono font-medium">${key}</td>
        <td class="py-2 text-white font-bold font-mono">${val}</td>
        <td class="py-2 text-slate-500 font-mono">${info.baseline}</td>
        <td class="py-2">${indicator}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  // ---------------------------------------------------------------------------
  // 4. Batch URL Scanner
  // ---------------------------------------------------------------------------
  function loadSampleBatch() {
    const samples = [
      "https://www.google.com",
      "https://www.wikipedia.org",
      "http://paypal-verification-account-security-update.com.tk/login.php",
      "http://192.168.1.1/secure/bank/login",
      "https://www.southbankmosaics.com",
      "http://apple-support-id-recovery-service.top/confirm.html",
      "https://www.nerdscandy.com",
      "http://secure-billing-netflix-account-auth.xyz/signin",
      "https://www.uni-mainz.de",
      "http://ebayisapi-dll-signin-account-update.buzz/auth"
    ];
    const input = document.getElementById("batch-urls-input");
    if (input) input.value = samples.join("\n");
    showNotification("Loaded 10 benchmark test URLs into batch queue.", "info");
  }

  async function executeBatchScan() {
    const input = document.getElementById("batch-urls-input");
    const text = input ? input.value.trim() : "";
    if (!text) {
      showNotification("Please enter one or more URLs.", "warning");
      return;
    }
    const urls = text.split("\n").map(u => u.trim()).filter(u => u.length > 0);
    const modelSelect = document.getElementById("batch-model-select");
    let model = modelSelect ? modelSelect.value : "Ensemble_Calibrated_SoftVoting";
    if (!model) model = "Ensemble_Calibrated_SoftVoting";
    const btn = document.getElementById("batch-scan-btn");

    if (btn) btn.disabled = true;

    try {
      let results = null;
      if (isServerOnline) {
        try {
          const response = await fetch("/api/batch-scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls: urls, model: model }),
          });
          if (response.ok) {
            results = await response.json();
          }
        } catch (e) {
          console.warn("Flask batch API unreachable, falling back to client-side engine.", e);
        }
      }

      if (!results || !Array.isArray(results) || results.length === 0) {
        results = urls.map(u => clientSideEvaluateUrl(u, model));
      }

      lastBatchResults = results;
      renderBatchResults(results);
      showNotification(`Batch scan completed for ${results.length} URLs!`, "success");
    } catch (error) {
      console.error("Batch error:", error);
      showNotification("Batch triage error: " + error.message, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function renderBatchResults(results) {
    const container = document.getElementById("batch-results-container");
    const tbody = document.getElementById("batch-results-body");
    const summary = document.getElementById("batch-summary-text");
    if (!container || !tbody) return;

    tbody.innerHTML = "";
    let highCount = 0;
    let medCount = 0;
    let lowCount = 0;

    results.forEach(r => {
      if (r.risk_tier === "HIGH") highCount++;
      else if (r.risk_tier === "MEDIUM") medCount++;
      else lowCount++;

      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40 transition border-b border-slate-800/60";

      let tierBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">LOW</span>`;
      let trustBadge = `<span class="font-bold text-emerald-400">${Math.round(r.trust_score)} / 100</span>`;
      
      if (r.risk_tier === "HIGH") {
        tierBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">HIGH RISK</span>`;
        trustBadge = `<span class="font-bold text-rose-400">${Math.round(r.trust_score)} / 100</span>`;
      } else if (r.risk_tier === "MEDIUM") {
        tierBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">MEDIUM</span>`;
        trustBadge = `<span class="font-bold text-amber-400">${Math.round(r.trust_score)} / 100</span>`;
      }

      const conf = r.confidence_percent !== undefined ? `${r.confidence_percent}%` : `${Math.round((r.phishing_probability || 0) * 100)}%`;

      tr.innerHTML = `
        <td class="py-3 px-4 max-w-xs truncate text-slate-300 font-mono" title="${r.url}">${r.url}</td>
        <td class="py-3 px-4 font-mono text-slate-400">${r.domain || "N/A"}</td>
        <td class="py-3 px-4">${trustBadge}</td>
        <td class="py-3 px-4">${tierBadge}</td>
        <td class="py-3 px-4 font-bold text-slate-200">${r.action_recommendation || "N/A"}</td>
        <td class="py-3 px-4 font-mono text-slate-400">${conf}</td>
      `;
      tbody.appendChild(tr);
    });

    if (summary) {
      summary.textContent = `Evaluated ${results.length} URLs • High Risk: ${highCount} | Medium: ${medCount} | Low (Safe): ${lowCount}`;
    }

    container.classList.remove("hidden");
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function downloadBatchCSV() {
    if (!lastBatchResults || lastBatchResults.length === 0) {
      showNotification("No batch results to export. Run a batch scan first.", "warning");
      return;
    }

    const headers = ["URL", "Domain", "Model", "Trust_Score", "Prediction", "Risk_Tier", "Action", "Confidence", "Phishing_Probability"];
    const rows = lastBatchResults.map(r => [
      `"${r.url.replace(/"/g, '""')}"`,
      `"${(r.domain || "").replace(/"/g, '""')}"`,
      `"${r.model_used || ""}"`,
      r.trust_score !== undefined ? r.trust_score : Math.round((1 - (r.phishing_probability || 0)) * 100),
      r.prediction_label || (r.is_phishing ? "PHISHING" : "LEGITIMATE"),
      r.risk_tier || "N/A",
      `"${r.action_recommendation || "N/A"}"`,
      `${r.confidence_percent || 0}%`,
      r.phishing_probability || 0
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `WebTRACE_Batch_Triage_Report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showNotification("Batch triage CSV exported successfully!", "success");
  }

  // ---------------------------------------------------------------------------
  // 5. Benchmarks Report Table & Interactive Figure Gallery
  // ---------------------------------------------------------------------------
  async function loadBenchmarks() {
    let benchmarks = DEFAULT_BENCHMARKS;
    try {
      const res = await fetch("/api/benchmarks");
      if (res.ok) {
        benchmarks = await res.json();
      }
    } catch (err) {
      // Use DEFAULT_BENCHMARKS
    }
    renderBenchmarksTable(benchmarks);
  }

  function renderBenchmarksTable(data) {
    const tbody = document.getElementById("benchmarks-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.forEach(r => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40 transition border-b border-slate-800/60";
      
      const acc = typeof r.Accuracy === "number" ? (r.Accuracy * 100).toFixed(2) + "%" : r.Accuracy;
      const prec = typeof r.Precision === "number" ? (r.Precision * 100).toFixed(2) + "%" : r.Precision;
      const rec = typeof r.Recall === "number" ? (r.Recall * 100).toFixed(2) + "%" : r.Recall;
      const f1 = typeof r.F1_Score === "number" ? (r.F1_Score * 100).toFixed(2) + "%" : r.F1_Score;
      const roc = typeof r.ROC_AUC === "number" ? r.ROC_AUC.toFixed(4) : r.ROC_AUC;
      const pr = typeof r.PR_AUC === "number" ? r.PR_AUC.toFixed(4) : r.PR_AUC;

      tr.innerHTML = `
        <td class="py-3 px-4 font-bold text-white">${r.Model}</td>
        <td class="py-3 px-4 text-xs text-slate-400">${r.Paradigm}</td>
        <td class="py-3 px-4 font-mono font-bold text-teal-400">${acc}</td>
        <td class="py-3 px-4 font-mono text-slate-300">${prec}</td>
        <td class="py-3 px-4 font-mono text-slate-300">${rec}</td>
        <td class="py-3 px-4 font-mono font-bold text-emerald-400">${f1}</td>
        <td class="py-3 px-4 font-mono text-slate-400">${roc}</td>
        <td class="py-3 px-4 font-mono text-slate-400">${pr}</td>
        <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-500/10 text-teal-400 border border-teal-500/20">${r.Status || "Ready"}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderFiguresGallery() {
    const container = document.getElementById("figures-gallery");
    if (!container) return;
    container.innerHTML = "";

    BENCHMARK_FIGURES.forEach(fig => {
      const card = document.createElement("div");
      card.className = "bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-lg hover:border-teal-500/50 transition cursor-pointer flex flex-col";
      
      const primarySrc = `/figures/${fig.id}`;
      const fallbackSrc = `outputs/figures/${fig.id}`;

      card.onclick = () => openImageModal(primarySrc, fallbackSrc, fig.title);

      card.innerHTML = `
        <div class="h-44 bg-slate-950 flex items-center justify-center overflow-hidden border-b border-slate-800 p-2 relative group">
          <img src="${primarySrc}" onerror="this.onerror=null; this.src='${fallbackSrc}';" alt="${fig.title}" class="max-h-full object-contain transition duration-300 group-hover:scale-105">
          <div class="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-teal-400 font-bold text-xs space-x-1">
            <span>🔍 Click to Enlarge</span>
          </div>
        </div>
        <div class="p-4 flex-1 flex flex-col justify-between space-y-1">
          <h5 class="font-bold text-white text-xs leading-snug">${fig.title}</h5>
          <p class="text-[11px] text-slate-400 leading-tight">${fig.desc}</p>
        </div>
      `;
      container.appendChild(card);
    });
  }

  function openImageModal(src1, src2, title) {
    const modal = document.getElementById("image-modal");
    const modalImg = document.getElementById("modal-img");
    const modalTitle = document.getElementById("modal-title");
    if (!modal || !modalImg) return;

    modalTitle.textContent = title;
    modalImg.src = src1;
    modalImg.onerror = () => { modalImg.src = src2; };
    modal.classList.remove("hidden");
  }

  function closeImageModal() {
    const modal = document.getElementById("image-modal");
    if (modal) modal.classList.add("hidden");
  }

  // Export functions to window for direct HTML onclick triggers
  window.initApp = initApp;
  window.switchTab = switchTab;
  window.toggleRole = toggleRole;
  window.pasteFromClipboard = pasteFromClipboard;
  window.loadSampleUrl = loadSampleUrl;
  window.handleScanSubmit = handleScanSubmit;
  window.loadSampleBatch = loadSampleBatch;
  window.executeBatchScan = executeBatchScan;
  window.downloadBatchCSV = downloadBatchCSV;
  window.openImageModal = openImageModal;
  window.closeImageModal = closeImageModal;
  window.showNotification = showNotification;
}
