/**
 * PhiShield AI - Main Interactive Frontend Logic
 * Supports: Live URL Feature Analyzer, Benchmark Viewer, Gallery Lightbox, XAI Bars, Session State
 */

document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initUserSessionState();
  initBenchmarkTable();
  initShapBars();
  initLiveScanner();
  initGalleryModal();
  initFaqAccordion();
});

/* ==========================================================================
   Navbar & Mobile Menu
   ========================================================================== */
function initNavbar() {
  const toggleBtn = document.querySelector(".mobile-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener("click", () => {
      navMenu.classList.toggle("open");
      const isOpen = navMenu.classList.contains("open");
      toggleBtn.setAttribute("aria-expanded", isOpen);
      toggleBtn.innerHTML = isOpen ? "✕" : "☰";
    });

    // Close on link click
    document.querySelectorAll(".nav-link").forEach(link => {
      link.addEventListener("click", () => {
        navMenu.classList.remove("open");
        if (toggleBtn) toggleBtn.innerHTML = "☰";
      });
    });
  }

  // Active section highlighting on scroll
  const sections = document.querySelectorAll("section[id]");
  window.addEventListener("scroll", () => {
    const scrollY = window.pageYOffset;
    sections.forEach(current => {
      const sectionHeight = current.offsetHeight;
      const sectionTop = current.offsetTop - 120;
      const sectionId = current.getAttribute("id");
      const navLink = document.querySelector(`.nav-link[href*="${sectionId}"]`);
      if (navLink) {
        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
          navLink.classList.add("active");
        } else {
          navLink.classList.remove("active");
        }
      }
    });
  });
}

/* ==========================================================================
   User Session State (LocalStorage Simulated Auth)
   ========================================================================== */
function initUserSessionState() {
  const userActionsContainer = document.getElementById("navUserActions");
  if (!userActionsContainer) return;

  const sessionJson = localStorage.getItem("phishield_session");
  if (sessionJson) {
    try {
      const user = JSON.parse(sessionJson);
      const initials = (user.name || "User").split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2);
      userActionsContainer.innerHTML = `
        <div class="user-profile-badge" title="Logged in as ${escapeHtml(user.email)} (${escapeHtml(user.role || 'Member')})">
          <span class="user-avatar-tiny">${initials}</span>
          <span style="font-weight:600; color: #fff;">${escapeHtml(user.name.split(" ")[0])}</span>
          <span style="font-size:0.75rem; color: var(--accent-cyan); opacity:0.8;">• ${escapeHtml(user.role || 'Analyst')}</span>
        </div>
        <button id="logoutBtn" class="btn btn-ghost btn-sm" title="Log Out of Session">
          Log Out
        </button>
      `;

      document.getElementById("logoutBtn")?.addEventListener("click", () => {
        localStorage.removeItem("phishield_session");
        showToast("Logged out successfully.", "info");
        setTimeout(() => window.location.reload(), 600);
      });
      return;
    } catch (e) {
      console.warn("Invalid session token in localStorage", e);
    }
  }

  // If no user session, show Login & Sign Up buttons
  userActionsContainer.innerHTML = `
    <a href="login.html" class="btn btn-ghost btn-sm">Log In</a>
    <a href="signup.html" class="btn btn-primary btn-sm">Get Started</a>
  `;
}

/* ==========================================================================
   Benchmark Table Rendering & Split Toggling
   ========================================================================== */
let currentSplit = "random"; // 'random' or 'unseen'

function initBenchmarkTable() {
  const tableBody = document.getElementById("benchmarkTableBody");
  const tabRandom = document.getElementById("tabRandomSplit");
  const tabUnseen = document.getElementById("tabUnseenSplit");

  if (!tableBody) return;

  function renderTable(splitKey) {
    if (!window.BENCHMARK_MODELS) return;
    
    // Sort descending by Accuracy for current split
    const sorted = [...window.BENCHMARK_MODELS].sort((a, b) => b[splitKey].accuracy - a[splitKey].accuracy);

    let html = "";
    sorted.forEach((item, index) => {
      const metrics = item[splitKey];
      const accPct = (metrics.accuracy * 100).toFixed(2);
      const precPct = (metrics.precision * 100).toFixed(2);
      const recPct = (metrics.recall * 100).toFixed(2);
      const f1Pct = (metrics.f1 * 100).toFixed(2);
      const aucVal = metrics.auc.toFixed(4);

      let badgeHtml = "";
      if (item.badge) {
        badgeHtml = `<span class="badge-tag ${item.badgeType}">${item.badge}</span>`;
      }

      html += `
        <tr>
          <td style="font-weight: 700; color: var(--text-muted); width: 40px;">#${index + 1}</td>
          <td>
            <div class="model-name-cell">
              <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
                <strong>${escapeHtml(item.name)}</strong>
                ${badgeHtml}
              </div>
              <small>${escapeHtml(item.description)}</small>
            </div>
          </td>
          <td>
            <span class="badge-tag ${item.paradigm === 'Deep Learning' ? 'featured' : 'standard'}">
              ${item.paradigm}
            </span>
          </td>
          <td>
            <div class="metric-bar-cell">
              <div class="metric-bar-track">
                <div class="metric-bar-fill" style="width: ${accPct}%"></div>
              </div>
              <strong>${accPct}%</strong>
            </div>
          </td>
          <td>${precPct}%</td>
          <td>${recPct}%</td>
          <td><strong style="color:var(--accent-cyan);">${f1Pct}%</strong></td>
          <td>${aucVal}</td>
        </tr>
      `;
    });

    tableBody.innerHTML = html;
  }

  // Initial render
  renderTable("random");

  if (tabRandom && tabUnseen) {
    tabRandom.addEventListener("click", () => {
      tabRandom.classList.add("active");
      tabUnseen.classList.remove("active");
      currentSplit = "random";
      renderTable("random");
      showToast("Displaying 80/20 Random Split benchmark results", "info");
    });

    tabUnseen.addEventListener("click", () => {
      tabUnseen.classList.add("active");
      tabRandom.classList.remove("active");
      currentSplit = "unseen";
      renderTable("unseen");
      showToast("Displaying Unseen Domain (Zero-Day) evaluation results", "info");
    });
  }
}

/* ==========================================================================
   SHAP Global Feature Importance Bars
   ========================================================================== */
function initShapBars() {
  const container = document.getElementById("shapBarsList");
  if (!container || !window.SHAP_TOP_FEATURES) return;

  const maxVal = Math.max(...window.SHAP_TOP_FEATURES.map(f => f.importance));

  let html = "";
  window.SHAP_TOP_FEATURES.forEach(item => {
    const widthPct = ((item.importance / maxVal) * 100).toFixed(1);
    html += `
      <div class="shap-row" title="${escapeHtml(item.description)}">
        <div class="shap-row-header">
          <span class="shap-feature-name">${escapeHtml(item.feature)}</span>
          <span class="shap-importance-val">|SHAP|: ${item.importance.toFixed(4)}</span>
        </div>
        <div class="shap-track">
          <div class="shap-fill" style="width: ${widthPct}%;"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

/* ==========================================================================
   Live URL Scanner & Feature Extractor (Client-Side Simulation)
   ========================================================================== */
function initLiveScanner() {
  const scanInput = document.getElementById("liveScanUrlInput");
  const scanBtn = document.getElementById("liveScanBtn");
  const presetChips = document.querySelectorAll(".preset-chip");

  if (!scanInput || !scanBtn) return;

  // Preset button click
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const targetUrl = chip.getAttribute("data-url");
      if (targetUrl) {
        scanInput.value = targetUrl;
        analyzeUrl(targetUrl);
      }
    });
  });

  // Scan button click
  scanBtn.addEventListener("click", () => {
    const val = scanInput.value.trim();
    if (!val) {
      showToast("Please enter a URL to inspect", "error");
      return;
    }
    analyzeUrl(val);
  });

  // Enter key in input
  scanInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      scanBtn.click();
    }
  });

  // Run initial scan on placeholder URL
  if (scanInput.value) {
    analyzeUrl(scanInput.value);
  }
}

/**
 * Shannon Entropy Calculation: H = -sum( p_i * log2(p_i) )
 */
function calculateShannonEntropy(str) {
  if (!str || str.length === 0) return 0;
  const freq = {};
  for (let char of str) {
    freq[char] = (freq[char] || 0) + 1;
  }
  const len = str.length;
  let entropy = 0;
  for (let char in freq) {
    const p = freq[char] / len;
    entropy -= p * Math.log2(p);
  }
  return parseFloat(entropy.toFixed(3));
}

/**
 * Extract genuine lexical and structural features in JavaScript
 */
function extractUrlFeatures(rawUrl) {
  let url = rawUrl.trim();
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    url = "http://" + url;
  }

  let parsed;
  try {
    parsed = new URL(url);
  } catch (e) {
    parsed = {
      hostname: url.split("/")[0] || "",
      pathname: "/" + (url.split("/").slice(1).join("/") || ""),
      search: "",
      protocol: "http:"
    };
  }

  const hostname = parsed.hostname || "";
  const pathname = parsed.pathname || "";
  const search = parsed.search || "";
  const fullUrl = url;

  // 1. Domain length & path length
  const domainLength = hostname.length;
  const pathLength = pathname.length;
  const urlLength = fullUrl.length;

  // 2. IP in hostname check
  const isDomainIP = /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname) || hostname.includes(":");

  // 3. Slashes and depth
  const slashCount = (fullUrl.match(/\//g) || []).length;
  const pathDepth = pathname.split("/").filter(p => p.length > 0).length;

  // 4. Entropy
  const urlEntropy = calculateShannonEntropy(fullUrl);
  const pathEntropy = calculateShannonEntropy(pathname);
  const netlocEntropy = calculateShannonEntropy(hostname);

  // 5. Special chars and digits
  const digitsInUrl = (fullUrl.match(/\d/g) || []).length;
  const digitRatioInUrl = urlLength > 0 ? parseFloat((digitsInUrl / urlLength).toFixed(3)) : 0;
  const digitsInDomain = (hostname.match(/\d/g) || []).length;
  const digitRatioDomain = domainLength > 0 ? parseFloat((digitsInDomain / domainLength).toFixed(3)) : 0;

  const specialChars = (fullUrl.match(/[^a-zA-Z0-9]/g) || []).length;
  const specialCharRatio = urlLength > 0 ? parseFloat((specialChars / urlLength).toFixed(3)) : 0;

  // 6. Suspicious TLDs
  const suspiciousTLDs = [".xyz", ".top", ".ru", ".cc", ".work", ".cn", ".club", ".buzz", ".info", ".tk", ".ml", ".ga", ".cf", ".gq"];
  const isSuspiciousTLD = suspiciousTLDs.some(tld => hostname.toLowerCase().endsWith(tld));
  const tldParts = hostname.split(".");
  const tldLength = tldParts.length > 1 ? tldParts[tldParts.length - 1].length : 0;

  // 7. Security Keywords
  const securityKeywords = [
    "login", "signin", "verify", "verification", "secure", "security", 
    "account", "banking", "update", "service", "apple-id", "paypal", 
    "netflix", "confirm", "wallet", "password", "auth", "token"
  ];
  
  const foundKeywords = securityKeywords.filter(kw => fullUrl.toLowerCase().includes(kw));
  const kwInPath = securityKeywords.some(kw => pathname.toLowerCase().includes(kw));

  // 8. Obfuscation indicators
  const hasDoubleSlashInPath = pathname.includes("//");
  const hasAtSymbol = fullUrl.includes("@");
  const hyphenCountDomain = (hostname.match(/-/g) || []).length;

  return {
    rawUrl,
    domainLength,
    pathLength,
    isDomainIP,
    slashCount,
    pathDepth,
    urlEntropy,
    pathEntropy,
    netlocEntropy,
    digitsInUrl,
    digitRatioInUrl,
    digitRatioDomain,
    specialCharRatio,
    isSuspiciousTLD,
    tldLength,
    foundKeywords,
    kwInPath,
    hasDoubleSlashInPath,
    hasAtSymbol,
    hyphenCountDomain
  };
}

/**
 * Compute Operational Risk Assessment mirroring the model scoring
 */
function analyzeUrl(url) {
  const feat = extractUrlFeatures(url);

  let riskScore = 0.05; // Base prior
  const riskFactors = [];

  // IP Host
  if (feat.isDomainIP) {
    riskScore += 0.40;
    riskFactors.push("Direct IP address used as hostname instead of a domain name (+40%)");
  }

  // Suspicious TLD
  if (feat.isSuspiciousTLD) {
    riskScore += 0.25;
    riskFactors.push("High-abuse Top-Level Domain detected (+25%)");
  }

  // Security Keywords
  if (feat.foundKeywords.length > 0) {
    const penalty = Math.min(0.35, feat.foundKeywords.length * 0.12);
    riskScore += penalty;
    riskFactors.push(`Sensitive keywords detected: "${feat.foundKeywords.slice(0, 3).join(', ')}" (+${Math.round(penalty*100)}%)`);
  }

  // High entropy / random string
  if (feat.urlEntropy > 4.4) {
    riskScore += 0.20;
    riskFactors.push(`Abnormally high Shannon entropy (${feat.urlEntropy} bits) indicates token randomization (+20%)`);
  }

  // High Special Char Ratio
  if (feat.specialCharRatio > 0.18) {
    riskScore += 0.15;
    riskFactors.push(`Excessive special character density (${(feat.specialCharRatio*100).toFixed(1)}%) (+15%)`);
  }

  // High digit density in domain or URL
  if (feat.digitRatioDomain > 0.15 || feat.digitRatioInUrl > 0.20) {
    riskScore += 0.12;
    riskFactors.push("High digit proportion indicating algorithmically generated identifiers (+12%)");
  }

  // Multiple hyphens in domain (brand impersonation)
  if (feat.hyphenCountDomain >= 2) {
    riskScore += 0.15;
    riskFactors.push(`Multiple hyphens (${feat.hyphenCountDomain}) in domain name indicate phishing typosquatting (+15%)`);
  }

  // Slashes & deep nesting
  if (feat.slashCount >= 5 || feat.pathDepth >= 4) {
    riskScore += 0.10;
    riskFactors.push(`Deep folder path nesting (${feat.slashCount} slashes) used for evasion (+10%)`);
  }

  // Well known benign domain discount
  const knownLegit = ["google.com", "github.com", "wikipedia.org", "coursera.org", "microsoft.com", "apple.com", "amazon.com"];
  if (knownLegit.some(d => url.includes(d)) && !feat.isDomainIP && !feat.isSuspiciousTLD) {
    riskScore = Math.max(0.01, riskScore * 0.15);
  }

  // Cap between 0.01 and 0.99
  riskScore = Math.min(0.99, Math.max(0.01, riskScore));
  const riskPct = Math.round(riskScore * 100);

  // Update UI Elements
  updateScannerUI(feat, riskPct, riskFactors);
}

function updateScannerUI(feat, riskPct, riskFactors) {
  // Dial gauge
  const dialBar = document.getElementById("dialBar");
  const dialPct = document.getElementById("dialPct");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictNotes = document.getElementById("verdictNotes");

  if (dialPct) dialPct.innerText = `${riskPct}%`;

  // Stroke Dashoffset (circle circumference is ~ 440)
  if (dialBar) {
    const offset = 440 - (440 * (riskPct / 100));
    dialBar.style.strokeDashoffset = offset;

    if (riskPct < 35) {
      dialBar.style.stroke = "var(--accent-emerald)";
    } else if (riskPct < 70) {
      dialBar.style.stroke = "var(--accent-amber)";
    } else {
      dialBar.style.stroke = "var(--accent-danger)";
    }
  }

  // Verdict Badge
  if (verdictBadge) {
    if (riskPct < 35) {
      verdictBadge.className = "verdict-badge safe";
      verdictBadge.innerHTML = `🛡️ Safe (Low Risk)`;
      if (verdictNotes) verdictNotes.innerText = "No deceptive lexical signals detected. Domain matches benign natural distribution.";
    } else if (riskPct < 70) {
      verdictBadge.className = "verdict-badge suspicious";
      verdictBadge.innerHTML = `⚠️ Suspicious (Review)`;
      if (verdictNotes) verdictNotes.innerText = "Elevated lexical anomalies found. User caution or sandbox quarantine advised.";
    } else {
      verdictBadge.className = "verdict-badge malicious";
      verdictBadge.innerHTML = `🚫 Malicious (Phishing)`;
      if (verdictNotes) verdictNotes.innerText = "Critical threat signals detected: potential credential theft or domain spoofing.";
    }
  }

  // Update feature numbers
  setElemText("featDomainLen", feat.domainLength);
  setElemText("featPathLen", feat.pathLength);
  setElemText("featEntropy", `${feat.urlEntropy} bits`);
  setElemText("featSlashCount", feat.slashCount);
  setElemText("featSpecialRatio", `${(feat.specialCharRatio * 100).toFixed(1)}%`);
  setElemText("featDigitRatio", `${(feat.digitRatioInUrl * 100).toFixed(1)}%`);

  // Indicators checklist
  const indIp = document.getElementById("indIsIp");
  if (indIp) {
    indIp.className = `indicator-val ${feat.isDomainIP ? 'bad' : 'ok'}`;
    indIp.innerText = feat.isDomainIP ? "YES (Warning)" : "NO (Domain Name)";
  }

  const indTld = document.getElementById("indSuspTld");
  if (indTld) {
    indTld.className = `indicator-val ${feat.isSuspiciousTLD ? 'bad' : 'ok'}`;
    indTld.innerText = feat.isSuspiciousTLD ? "YES (High Risk)" : "Standard";
  }

  const indKw = document.getElementById("indKeywords");
  if (indKw) {
    if (feat.foundKeywords.length > 0) {
      indKw.className = "indicator-val bad";
      indKw.innerText = `${feat.foundKeywords.length} Detected (${feat.foundKeywords.slice(0, 2).join(", ")})`;
    } else {
      indKw.className = "indicator-val ok";
      indKw.innerText = "None (Clean)";
    }
  }

  // Contributing factors list
  const factorsList = document.getElementById("contributingFactorsList");
  if (factorsList) {
    if (riskFactors.length === 0) {
      factorsList.innerHTML = `<li style="color:var(--accent-emerald);">✓ Standard lexical profile with zero anomalous features detected.</li>`;
    } else {
      factorsList.innerHTML = riskFactors.map(f => `<li>• ${escapeHtml(f)}</li>`).join("");
    }
  }
}

function setElemText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

/* ==========================================================================
   Gallery Lightbox Modal
   ========================================================================== */
function initGalleryModal() {
  const modal = document.getElementById("galleryModal");
  const modalImg = document.getElementById("modalImg");
  const modalTitle = document.getElementById("modalTitle");
  const modalDesc = document.getElementById("modalDesc");
  const closeBtn = document.getElementById("modalCloseBtn");
  const figureCards = document.querySelectorAll(".figure-card");

  if (!modal || !modalImg) return;

  figureCards.forEach(card => {
    card.addEventListener("click", () => {
      const src = card.getAttribute("data-img-src");
      const title = card.getAttribute("data-title") || "Research Artifact";
      const desc = card.getAttribute("data-desc") || "";

      modalImg.src = src;
      modalImg.alt = title;
      if (modalTitle) modalTitle.innerText = title;
      if (modalDesc) modalDesc.innerText = desc;

      modal.classList.add("active");
      document.body.style.overflow = "hidden";
    });
  });

  function closeModal() {
    modal.classList.remove("active");
    document.body.style.overflow = "auto";
  }

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeModal();
    }
  });
}

/* ==========================================================================
   FAQ Accordion
   ========================================================================== */
function initFaqAccordion() {
  const faqItems = document.querySelectorAll(".faq-item");
  faqItems.forEach(item => {
    const question = item.querySelector(".faq-question");
    if (question) {
      question.addEventListener("click", () => {
        const isActive = item.classList.contains("active");
        faqItems.forEach(other => other.classList.remove("active"));
        if (!isActive) item.classList.add("active");
      });
    }
  });
}

/* ==========================================================================
   Toast Notification Utility
   ========================================================================== */
function showToast(message, type = "info") {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: "✓",
    error: "✕",
    info: "ℹ"
  };

  toast.innerHTML = `
    <span style="font-weight: 700; color: var(--accent-cyan);">${iconMap[type] || "•"}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 350);
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

window.showToast = showToast;
