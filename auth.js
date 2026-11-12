/**
 * PhiShield AI - Authentication Client Controller
 * Pure Vanilla JavaScript Client-Side Auth Management with LocalStorage
 */

const DEFAULT_DEMO_USERS = [
  {
    name: "Security Analyst",
    email: "analyst@phishield.ai",
    password: "Analyst2026!",
    role: "Lead Security Analyst",
    organization: "Cyber Threat Intelligence Unit"
  },
  {
    name: "Dr. Elena Vance",
    email: "researcher@phishield.ai",
    password: "Research2026!",
    role: "ML Researcher",
    organization: "AI Security Lab"
  }
];

document.addEventListener("DOMContentLoaded", () => {
  seedDefaultUsers();
  initPasswordToggles();
  initLoginForm();
  initSignupForm();
  initDemoAccountFillers();
  initForgotPasswordModal();
});

/**
 * Seed initial mock accounts into LocalStorage if not present
 */
function seedDefaultUsers() {
  const existingUsers = localStorage.getItem("phishield_users");
  if (!existingUsers) {
    localStorage.setItem("phishield_users", JSON.stringify(DEFAULT_DEMO_USERS));
  }
}

/**
 * Toggle password visibility between text and password type
 */
function initPasswordToggles() {
  const toggleBtns = document.querySelectorAll(".password-toggle-btn");
  toggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const input = btn.parentElement.querySelector("input");
      if (!input) return;

      if (input.type === "password") {
        input.type = "text";
        btn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
            <line x1="1" y1="1" x2="23" y2="23"></line>
          </svg>
        `;
      } else {
        input.type = "password";
        btn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
        `;
      }
    });
  });
}

/**
 * Login Form Logic
 */
function initLoginForm() {
  const loginForm = document.getElementById("loginForm");
  if (!loginForm) return;

  const emailInput = document.getElementById("loginEmail");
  const passwordInput = document.getElementById("loginPassword");
  const submitBtn = document.getElementById("loginSubmitBtn");
  const errorAlert = document.getElementById("loginErrorAlert");

  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const email = emailInput.value.trim().toLowerCase();
    const password = passwordInput.value;

    // Reset errors
    if (errorAlert) errorAlert.style.display = "none";
    loginForm.classList.remove("shake");

    if (!email || !password) {
      showError("Please enter both email and password.");
      return;
    }

    // Set loading state
    setLoadingState(submitBtn, true, "Authenticating...");

    setTimeout(() => {
      const users = JSON.parse(localStorage.getItem("phishield_users") || "[]");
      const foundUser = users.find(u => u.email.toLowerCase() === email && u.password === password);

      if (foundUser) {
        // Success: Store active session
        const sessionPayload = {
          name: foundUser.name,
          email: foundUser.email,
          role: foundUser.role,
          loggedInAt: new Date().toISOString()
        };
        localStorage.setItem("phishield_session", JSON.stringify(sessionPayload));

        if (window.showToast) {
          window.showToast(`Welcome back, ${foundUser.name}!`, "success");
        }

        submitBtn.innerHTML = `✓ Verified! Redirecting...`;
        submitBtn.style.background = "var(--grad-emerald-cyan)";

        setTimeout(() => {
          window.location.href = "index.html";
        }, 800);
      } else {
        setLoadingState(submitBtn, false, "Sign In to Terminal");
        showError("Invalid email or password. You can use the Demo Accounts below to test.");
        loginForm.classList.add("shake");
      }
    }, 600);
  });

  function showError(msg) {
    if (errorAlert) {
      errorAlert.innerText = msg;
      errorAlert.style.display = "block";
    } else if (window.showToast) {
      window.showToast(msg, "error");
    }
  }
}

/**
 * Signup Form Logic & Password Strength Calculation
 */
function initSignupForm() {
  const signupForm = document.getElementById("signupForm");
  if (!signupForm) return;

  const nameInput = document.getElementById("signupName");
  const emailInput = document.getElementById("signupEmail");
  const roleSelect = document.getElementById("signupRole");
  const passwordInput = document.getElementById("signupPassword");
  const confirmPasswordInput = document.getElementById("signupConfirmPassword");
  const termsCheckbox = document.getElementById("signupTerms");
  const submitBtn = document.getElementById("signupSubmitBtn");
  const strengthMeter = document.getElementById("passwordStrengthMeter");
  const strengthLabel = document.getElementById("strengthLabel");
  const errorAlert = document.getElementById("signupErrorAlert");

  // Checklist items
  const reqLen = document.getElementById("reqLen");
  const reqUpper = document.getElementById("reqUpper");
  const reqLower = document.getElementById("reqLower");
  const reqNumber = document.getElementById("reqNumber");
  const reqSpecial = document.getElementById("reqSpecial");

  // Password Strength Evaluation
  if (passwordInput && strengthMeter) {
    passwordInput.addEventListener("input", () => {
      const val = passwordInput.value;
      const metrics = checkPasswordRequirements(val);

      // Update checklists
      updateReqItem(reqLen, metrics.hasLen);
      updateReqItem(reqUpper, metrics.hasUpper);
      updateReqItem(reqLower, metrics.hasLower);
      updateReqItem(reqNumber, metrics.hasNumber);
      updateReqItem(reqSpecial, metrics.hasSpecial);

      // Score 0-4
      let score = 0;
      if (metrics.hasLen) score++;
      if (metrics.hasUpper && metrics.hasLower) score++;
      if (metrics.hasNumber) score++;
      if (metrics.hasSpecial) score++;

      strengthMeter.className = "strength-meter";
      if (!val) {
        strengthLabel.innerText = "Empty";
        strengthLabel.style.color = "var(--text-muted)";
      } else if (score <= 1) {
        strengthMeter.classList.add("strength-weak");
        strengthLabel.innerText = "Weak";
        strengthLabel.style.color = "var(--accent-rose)";
      } else if (score === 2) {
        strengthMeter.classList.add("strength-fair");
        strengthLabel.innerText = "Fair";
        strengthLabel.style.color = "var(--accent-amber)";
      } else if (score === 3) {
        strengthMeter.classList.add("strength-good");
        strengthLabel.innerText = "Good";
        strengthLabel.style.color = "var(--accent-blue)";
      } else {
        strengthMeter.classList.add("strength-strong");
        strengthLabel.innerText = "Strong";
        strengthLabel.style.color = "var(--accent-emerald)";
      }
    });
  }

  function updateReqItem(elem, isMet) {
    if (!elem) return;
    if (isMet) {
      elem.classList.add("met");
      elem.querySelector("span:first-child").innerText = "✓";
    } else {
      elem.classList.remove("met");
      elem.querySelector("span:first-child").innerText = "•";
    }
  }

  signupForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const name = nameInput.value.trim();
    const email = emailInput.value.trim().toLowerCase();
    const role = roleSelect ? roleSelect.value : "Security Analyst";
    const password = passwordInput.value;
    const confirmPassword = confirmPasswordInput.value;

    if (errorAlert) errorAlert.style.display = "none";
    signupForm.classList.remove("shake");

    if (!name || !email || !password) {
      showError("Please fill out all required fields.");
      return;
    }

    if (password.length < 8) {
      showError("Password must be at least 8 characters long.");
      return;
    }

    if (password !== confirmPassword) {
      showError("Passwords do not match. Please re-enter.");
      confirmPasswordInput.focus();
      return;
    }

    if (termsCheckbox && !termsCheckbox.checked) {
      showError("Please acknowledge the Research & Usage Guidelines.");
      return;
    }

    // Check if email already registered
    const users = JSON.parse(localStorage.getItem("phishield_users") || "[]");
    if (users.some(u => u.email.toLowerCase() === email)) {
      showError("An account with this email already exists. Try signing in.");
      return;
    }

    setLoadingState(submitBtn, true, "Registering Identity...");

    setTimeout(() => {
      // Save new user
      const newUser = {
        name,
        email,
        password,
        role,
        registeredAt: new Date().toISOString()
      };
      users.push(newUser);
      localStorage.setItem("phishield_users", JSON.stringify(users));

      // Auto sign-in or redirect to login
      const sessionPayload = {
        name: newUser.name,
        email: newUser.email,
        role: newUser.role,
        loggedInAt: new Date().toISOString()
      };
      localStorage.setItem("phishield_session", JSON.stringify(sessionPayload));

      submitBtn.innerHTML = `✓ Account Created! Launching...`;
      submitBtn.style.background = "var(--grad-emerald-cyan)";

      if (window.showToast) {
        window.showToast("Account created successfully! Welcome to PhiShield.", "success");
      }

      setTimeout(() => {
        window.location.href = "index.html";
      }, 1000);
    }, 750);
  });

  function showError(msg) {
    if (errorAlert) {
      errorAlert.innerText = msg;
      errorAlert.style.display = "block";
    } else if (window.showToast) {
      window.showToast(msg, "error");
    }
  }
}

function checkPasswordRequirements(pwd) {
  return {
    hasLen: pwd.length >= 8,
    hasUpper: /[A-Z]/.test(pwd),
    hasLower: /[a-z]/.test(pwd),
    hasNumber: /[0-9]/.test(pwd),
    hasSpecial: /[^A-Za-z0-9]/.test(pwd)
  };
}

/**
 * 1-Click Demo Account Quick-Fill Buttons on Login
 */
function initDemoAccountFillers() {
  const fillAnalystBtn = document.getElementById("fillAnalystBtn");
  const fillResearcherBtn = document.getElementById("fillResearcherBtn");
  const emailInput = document.getElementById("loginEmail");
  const passwordInput = document.getElementById("loginPassword");

  if (fillAnalystBtn && emailInput && passwordInput) {
    fillAnalystBtn.addEventListener("click", () => {
      emailInput.value = "analyst@phishield.ai";
      passwordInput.value = "Analyst2026!";
      if (window.showToast) {
        window.showToast("Analyst credentials populated! Click Sign In.", "info");
      }
    });
  }

  if (fillResearcherBtn && emailInput && passwordInput) {
    fillResearcherBtn.addEventListener("click", () => {
      emailInput.value = "researcher@phishield.ai";
      passwordInput.value = "Research2026!";
      if (window.showToast) {
        window.showToast("Researcher credentials populated! Click Sign In.", "info");
      }
    });
  }
}

/**
 * Forgot Password Interactive Dialog Simulator
 */
function initForgotPasswordModal() {
  const trigger = document.getElementById("forgotPasswordLink");
  if (!trigger) return;

  trigger.addEventListener("click", (e) => {
    e.preventDefault();
    const emailPrompt = prompt("Enter your registered email address to receive password recovery instructions:", "analyst@phishield.ai");
    if (emailPrompt) {
      if (window.showToast) {
        window.showToast(`Recovery instructions simulated for ${emailPrompt}. Check demo console!`, "success");
      } else {
        alert(`Recovery instructions simulated for ${emailPrompt}. For demo accounts, credentials are Analyst2026! or Research2026!`);
      }
    }
  });
}

function setLoadingState(btn, isLoading, loadingText = "Loading...") {
  if (!btn) return;
  if (isLoading) {
    btn.disabled = true;
    btn.dataset.originalHtml = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span> ${loadingText}`;
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.originalHtml || "Submit";
  }
}
