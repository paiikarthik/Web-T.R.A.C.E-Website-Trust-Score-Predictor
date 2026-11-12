/**
 * PhiShield AI - Project Benchmark Data & Knowledge Base
 * Extracted directly from Major Project training runs on PhiUSIIL Phishing Dataset
 */

const PROJECT_CONFIG = {
  name: "PhiShield AI",
  subtitle: "Intelligent Phishing URL Detection & Explainable AI Benchmark",
  dataset: "PhiUSIIL Phishing URL Dataset",
  totalSamples: "235,370",
  testSamples: "47,074",
  featuresCount: 33,
  modelsCount: 10,
  peakAccuracy: "99.73%",
  peakF1: "99.68%",
  avgInferenceTime: "< 1.8ms",
  paradigm: "Zero-Scraping Pure-URL Architecture",
  splits: ["Random 80/20 Split", "Unseen Domain Split (Zero-Day Evaluation)"],
  cvFolds: 5
};

const BENCHMARK_MODELS = [
  {
    id: "DL_DeepMLP",
    name: "Deep MLP (Multi-Layer Perceptron)",
    paradigm: "Deep Learning",
    badge: "Champion DL",
    badgeType: "champion",
    description: "4-layer feedforward network with BatchNorm, Dropout (0.2), and GELU activation.",
    random: {
      accuracy: 0.9973,
      precision: 0.9997,
      recall: 0.9940,
      f1: 0.9969,
      auc: 0.9989
    },
    unseen: {
      accuracy: 0.9956,
      precision: 0.9990,
      recall: 0.9907,
      f1: 0.9948,
      auc: 0.9989
    }
  },
  {
    id: "ML_XGBoost",
    name: "Extreme Gradient Boosting (XGBoost)",
    paradigm: "Machine Learning",
    badge: "Champion ML",
    badgeType: "champion",
    description: "Histogram-based gradient boosted decision trees with L2 regularization and tree subsampling.",
    random: {
      accuracy: 0.9964,
      precision: 0.9974,
      recall: 0.9943,
      f1: 0.9958,
      auc: 0.9988
    },
    unseen: {
      accuracy: 0.9963,
      precision: 0.9970,
      recall: 0.9942,
      f1: 0.9956,
      auc: 0.9988
    }
  },
  {
    id: "DL_Hybrid_CNNLSTM",
    name: "Hybrid CNN-BiLSTM",
    paradigm: "Deep Learning",
    badge: "Hybrid Architecture",
    badgeType: "featured",
    description: "1D-CNN temporal feature extractor coupled with a Bidirectional LSTM recurrent sequence head.",
    random: {
      accuracy: 0.9930,
      precision: 0.9905,
      recall: 0.9932,
      f1: 0.9918,
      auc: 0.9989
    },
    unseen: {
      accuracy: 0.9481,
      precision: 0.9980,
      recall: 0.8791,
      f1: 0.9348,
      auc: 0.9770
    }
  },
  {
    id: "DL_SelfAttention",
    name: "Self-Attention Tabular Net",
    paradigm: "Deep Learning",
    badge: "Transformer Based",
    badgeType: "featured",
    description: "Multi-head scaled dot-product self-attention capturing non-linear cross-feature relationships.",
    random: {
      accuracy: 0.9919,
      precision: 0.9960,
      recall: 0.9851,
      f1: 0.9905,
      auc: 0.9985
    },
    unseen: {
      accuracy: 0.9925,
      precision: 0.9910,
      recall: 0.9913,
      f1: 0.9911,
      auc: 0.9983
    }
  },
  {
    id: "ML_DecisionTree",
    name: "Decision Tree (CART)",
    paradigm: "Machine Learning",
    badge: "Interpretable",
    badgeType: "standard",
    description: "Optimized classification tree split via Gini impurity criterion with minimal depth tuning.",
    random: {
      accuracy: 0.9854,
      precision: 0.9923,
      recall: 0.9732,
      f1: 0.9827,
      auc: 0.9959
    },
    unseen: {
      accuracy: 0.9866,
      precision: 0.9928,
      recall: 0.9754,
      f1: 0.9840,
      auc: 0.9963
    }
  },
  {
    id: "ML_RandomForest",
    name: "Random Forest (100 Trees)",
    paradigm: "Machine Learning",
    badge: "Ensemble Baseline",
    badgeType: "standard",
    description: "Bagged ensemble of decorrelated decision trees with bootstrap aggregation.",
    random: {
      accuracy: 0.9591,
      precision: 0.9891,
      recall: 0.9142,
      f1: 0.9502,
      auc: 0.9969
    },
    unseen: {
      accuracy: 0.9558,
      precision: 0.9883,
      recall: 0.9061,
      f1: 0.9454,
      auc: 0.9971
    }
  },
  {
    id: "DL_LSTM",
    name: "Bidirectional LSTM",
    paradigm: "Deep Learning",
    badge: "Recurrent",
    badgeType: "standard",
    description: "Bidirectional Recurrent Neural Network modeling sequential character/token transitions.",
    random: {
      accuracy: 0.9583,
      precision: 0.9768,
      recall: 0.9243,
      f1: 0.9499,
      auc: 0.9911
    },
    unseen: {
      accuracy: 0.9448,
      precision: 0.9471,
      recall: 0.9208,
      f1: 0.9338,
      auc: 0.9779
    }
  },
  {
    id: "DL_CNN1D",
    name: "1D Convolutional Neural Network",
    paradigm: "Deep Learning",
    badge: "Convolutional",
    badgeType: "standard",
    description: "Multi-filter 1D CNN kernel extracting local spatial n-gram patterns across URL representations.",
    random: {
      accuracy: 0.7008,
      precision: 0.5882,
      recall: 0.9987,
      f1: 0.7403,
      auc: 0.9975
    },
    unseen: {
      accuracy: 0.9510,
      precision: 0.9960,
      recall: 0.8877,
      f1: 0.9388,
      auc: 0.9957
    }
  },
  {
    id: "ML_LogisticRegression",
    name: "Logistic Regression (L2)",
    paradigm: "Machine Learning",
    badge: "Linear Baseline",
    badgeType: "standard",
    description: "Standard regularized linear classifier trained on standardized feature vectors.",
    random: {
      accuracy: 0.9135,
      precision: 0.9467,
      recall: 0.8450,
      f1: 0.8929,
      auc: 0.9556
    },
    unseen: {
      accuracy: 0.9154,
      precision: 0.9485,
      recall: 0.8459,
      f1: 0.8943,
      auc: 0.9566
    }
  },
  {
    id: "ML_LinearSVM",
    name: "Linear Support Vector Machine",
    paradigm: "Machine Learning",
    badge: "Max-Margin",
    badgeType: "standard",
    description: "Linear maximum margin hyperplane classifier with squared hinge loss.",
    random: {
      accuracy: 0.9067,
      precision: 0.9505,
      recall: 0.8246,
      f1: 0.8831,
      auc: 0.9551
    },
    unseen: {
      accuracy: 0.9077,
      precision: 0.9502,
      recall: 0.8249,
      f1: 0.8831,
      auc: 0.9558
    }
  }
];

const SHAP_TOP_FEATURES = [
  { feature: "SpacialCharRatioInURL", importance: 0.0831, description: "Ratio of non-alphanumeric symbols to total URL length" },
  { feature: "LetterRatioInURL", importance: 0.0829, description: "Proportion of standard alphabetic characters" },
  { feature: "slash_count", importance: 0.0792, description: "Total forward slashes indicative of nested evasion folders" },
  { feature: "path_len", importance: 0.0624, description: "Length of the target path string past domain" },
  { feature: "NoOfOtherSpecialCharsInURL", importance: 0.0344, description: "Uncommon delimiters and hex escapes like %, _, =" },
  { feature: "is_susp_tld", importance: 0.0225, description: "High-abuse Top-Level Domains (.top, .xyz, .ru, .cc, etc.)" },
  { feature: "DegitRatioInURL", importance: 0.0221, description: "Percentage of numeric digits across the entire URL" },
  { feature: "NoOfDegitsInURL", importance: 0.0199, description: "Absolute digit count in netloc and parameters" },
  { feature: "TLDLength", importance: 0.0190, description: "Character length of the active TLD suffix" },
  { feature: "digit_ratio_domain", importance: 0.0161, description: "Digit density strictly within the hostname" },
  { feature: "DomainLength", importance: 0.0136, description: "Length of registered second-level plus subdomains" },
  { feature: "url_entropy", importance: 0.0134, description: "Shannon entropy measuring randomness and token obfuscation" }
];

const PRESET_SAMPLE_URLS = [
  {
    url: "https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn",
    category: "Legitimate (Benign)",
    expectedRisk: "Safe",
    score: 0.03,
    notes: "Standard reputable OAuth endpoint, natural entropy, well-formed domain, no spoofing keywords in weird positions."
  },
  {
    url: "http://paypal-security-account-verification-alert.ru/signin/login.php?token=93821098&ssl=true",
    category: "Phishing (Credential Theft)",
    expectedRisk: "Malicious",
    score: 0.98,
    notes: "Suspicious TLD (.ru), domain brand spoofing (paypal in subdomain), nested credential path, abnormal special char ratio."
  },
  {
    url: "http://192.168.10.45/apple-id/update-billing-information/verify.html?client_id=8923",
    category: "Phishing (Direct IP Host)",
    expectedRisk: "Malicious",
    score: 0.96,
    notes: "Raw IPv4 address in netloc (IsDomainIP=1), keyword stacking ('apple-id', 'update-billing', 'verify'), high digit ratio."
  },
  {
    url: "https://secure-login.chase-online-banking-auth.xyz/web/auth/verify?session=x83kf8923jk1l029",
    category: "Phishing (Evasion & Token Randomness)",
    expectedRisk: "Malicious",
    score: 0.94,
    notes: "Abusive .xyz TLD, high Shannon entropy in query token, compound bank brand keyword in host, hyphens in domain."
  },
  {
    url: "https://en.wikipedia.org/wiki/Machine_learning",
    category: "Legitimate (Educational)",
    expectedRisk: "Safe",
    score: 0.01,
    notes: "Low domain length, clean path, 0 obfuscation, 0 suspicious TLDs, natural vocabulary distribution."
  }
];

// Export to window for vanilla JS accessibility
window.PROJECT_CONFIG = PROJECT_CONFIG;
window.BENCHMARK_MODELS = BENCHMARK_MODELS;
window.SHAP_TOP_FEATURES = SHAP_TOP_FEATURES;
window.PRESET_SAMPLE_URLS = PRESET_SAMPLE_URLS;
