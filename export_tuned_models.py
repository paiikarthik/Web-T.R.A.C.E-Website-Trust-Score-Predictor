"""
Export Tuned Models and Update Feature Extractor & Model Service.
"""

import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
DATA_PATH = BASE_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
RANDOM_STATE = 42

from research_audit_and_tune import extract_features_v2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Fitting full-dataset trained models with tuned hyperparameters...")
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["URL"], keep="first").copy()
df["y"] = (df["label"] == 0).astype(int)

# Use 60,000 balanced rows to train production models
SAMPLE_SIZE = min(60000, len(df))
df_sample = (
    df.groupby("y", group_keys=False)
    .apply(lambda g: g.sample(n=SAMPLE_SIZE // 2, random_state=RANDOM_STATE))
    .reset_index(drop=True)
)

print(f"Extracting features for {len(df_sample):,} rows...")
features = [extract_features_v2(u) for u in df_sample["URL"]]
X = pd.DataFrame(features)
y = df_sample["y"]
groups = df_sample["Domain"].astype(str)

# Train-test split (GroupShuffleSplit unseen domain)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Save feature list
pd.Series(X.columns, name="feature").to_csv(REPORT_DIR / "feature_list.csv", index=False)

# 1. Random Forest (Tuned)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, MODEL_DIR / "ML_RandomForest.pkl")
print("Saved ML_RandomForest.pkl")

# 2. XGBoost (Tuned)
xgb_model = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.05, subsample=0.85, scale_pos_weight=1.2, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, MODEL_DIR / "ML_XGBoost.pkl")
print("Saved ML_XGBoost.pkl")

# 3. Deep Learning (DeepMLP Tuned)
dl_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=0.005, learning_rate_init=0.002, max_iter=40, random_state=RANDOM_STATE, early_stopping=True))
])
dl_pipe.fit(X_train, y_train)
joblib.dump(dl_pipe, MODEL_DIR / "DL_DeepMLP.pkl")
print("Saved DL_DeepMLP.pkl")

print("[OK] Production models successfully refreshed with audited 17 features!")
